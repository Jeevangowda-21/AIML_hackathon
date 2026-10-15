import io
import json
import os
import sqlite3
import time
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DB_PATH = os.getenv("FOODRESCUE_DB", "foodrescue.db")
MAX_SIDE = 1568
MAX_BYTES = 10 * 1024 * 1024
AI_PER_MIN = int(os.getenv("AI_PER_MIN", "6"))
AI_GLOBAL_PER_MIN = int(os.getenv("AI_GLOBAL_PER_MIN", "30"))
WRITE_PER_MIN = int(os.getenv("WRITE_PER_MIN", "20"))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

Category = Literal[
    "cooked_rice", "curry_or_dal", "bread_or_roti", "dairy", "meat_or_egg",
    "fruit_or_veg", "snack_or_fried", "sweet", "packaged", "other",
]


class FoodItem(BaseModel):
    name: str = Field(description="Specific dish name, e.g. 'veg biryani', 'dal tadka', 'idli'")
    category: Category
    quantity_note: str = Field(description="Rough amount, e.g. 'large tray, about half full'; empty string if unclear")


class Issue(BaseModel):
    type: Literal[
        "mold", "discoloration", "slime_or_film", "contamination",
        "dried_out", "separation_or_curdling", "pests", "bad_packaging", "other",
    ]
    description: str
    severity: Literal["minor", "major"]


class VisionResult(BaseModel):
    is_food: bool
    photo_quality: Literal["good", "blurry", "too_dark", "too_far", "obstructed", "not_food"]
    items: list[FoodItem]
    veg_status: Literal["veg", "non_veg", "mixed", "unclear"]
    visible_issues: list[Issue]
    looks_fresh: bool = Field(description="True only if NO visible signs of spoilage")
    estimated_servings: int = Field(description="Approx. number of adult servings, 0 if not food")
    container: Literal["open_tray", "covered", "sealed_packaging", "unclear"]
    confidence: float = Field(description="Overall confidence between 0 and 1")
    needs_retake: bool
    notes: str = Field(description="Short remark for the donor/NGO; empty string if none")

    @field_validator("confidence")
    @classmethod
    def _clamp_conf(cls, v):
        return min(1.0, max(0.0, v))

    @field_validator("estimated_servings")
    @classmethod
    def _non_negative(cls, v):
        return max(0, v)


SYSTEM_PROMPT = """You are the vision component of a food-donation platform in India. \
Donors (colleges, hostels, caterers, event organisers) photograph leftover food, and NGOs \
decide whether to collect it.

Your job: identify the food and report ONLY what is visibly observable.

Rules:
- Be specific with dish names and prefer common Indian names (dal tadka, jeera rice, \
paneer butter masala, chapati, idli, sambar, gulab jamun) when appropriate.
- List every distinct dish you can see as a separate item.
- Report visible_issues only if you actually see them: mold, unusual discoloration, \
slimy or glossy film, foreign objects or contamination, dried-out/crusted surfaces, \
curdled or separated gravies, pests, torn or bulging packaging.
- You cannot detect bacteria, smell, or temperature. Never claim food is 'safe'; \
looks_fresh only means 'no visible signs of spoilage'.
- If the image is blurry, too dark, too far away, obstructed, or not food, set \
photo_quality accordingly, needs_retake=true, and lower confidence.
- estimated_servings: estimate adult servings from the visible quantity. If unsure, give \
a conservative number.
- confidence reflects how sure you are of the whole analysis (0 to 1)."""


def preprocess_image(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _call_model(jpeg_bytes: bytes) -> VisionResult:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            "Analyze this photo of donated food.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=VisionResult,
            temperature=0.2,
        ),
    )
    if isinstance(resp.parsed, VisionResult):
        return resp.parsed
    return VisionResult.model_validate_json(resp.text)


def analyze_food_image(image_bytes: bytes, retries: int = 1) -> VisionResult:
    try:
        jpeg = preprocess_image(image_bytes)
    except Exception as e:
        raise ValueError(f"Could not read image: {e}") from e
    last_err = None
    for _ in range(retries + 1):
        try:
            return _call_model(jpeg)
        except (ValidationError, ValueError, TypeError) as e:
            last_err = e
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Vision analysis failed after retries: {last_err}")


RISK = {
    "cooked_rice": "high", "curry_or_dal": "high", "dairy": "high", "meat_or_egg": "high",
    "fruit_or_veg": "medium", "bread_or_roti": "medium", "sweet": "medium", "other": "medium",
    "snack_or_fried": "low", "packaged": "low",
}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

LIMIT_HOURS = {
    "room_temp": {"high": 2, "medium": 4, "low": 6},
    "hot_held": {"high": 4, "medium": 6, "low": 8},
    "refrigerated": {"high": 12, "medium": 24, "low": 48},
}
STORAGE_LABEL = {"room_temp": "at room temperature", "hot_held": "kept hot", "refrigerated": "refrigerated"}
URGENT_BELOW_HOURS = 1.5

DISCLAIMER = ("Automated first-pass screening from a photo and general food-safety time limits. "
              "It cannot detect bacteria or guarantee safety - the NGO must inspect the food on pickup.")


class Verdict(BaseModel):
    verdict: Literal["accept", "urgent_pickup", "reject", "retake"]
    reasons: list[str]
    risk_level: Literal["low", "medium", "high"]
    pickup_within_hours: float
    vision: VisionResult
    disclaimer: str


def evaluate(v: VisionResult, hours: float, storage: str) -> Verdict:
    reasons: list[str] = []
    risk = max((RISK[i.category] for i in v.items), key=RISK_ORDER.get, default="medium")
    limit = LIMIT_HOURS[storage][risk]
    remaining = round(limit - hours, 1)

    def out(verdict: str) -> Verdict:
        window = max(0.0, remaining) if verdict in ("accept", "urgent_pickup") else 0.0
        return Verdict(verdict=verdict, reasons=reasons, risk_level=risk,
                       pickup_within_hours=window, vision=v, disclaimer=DISCLAIMER)

    if v.needs_retake or not v.is_food:
        reasons.append(v.notes or f"Photo not usable ({v.photo_quality}). Please retake it clearly.")
        return out("retake")

    major = [i for i in v.visible_issues if i.severity == "major"]
    if major or not v.looks_fresh:
        for i in major:
            reasons.append(f"Visible problem: {i.type.replace('_', ' ')} - {i.description}")
        if not major:
            reasons.append("Food does not look fresh in the photo.")
        return out("reject")

    if remaining <= 0:
        reasons.append(f"{hours:g}h since cooking, {STORAGE_LABEL[storage]}; "
                       f"limit is {limit}h for {risk}-risk food.")
        return out("reject")

    if v.confidence < 0.5:
        reasons.append("The AI is not confident about this photo. Please retake it in better light.")
        return out("retake")

    verdict = "accept"
    for i in v.visible_issues:
        reasons.append(f"Minor note: {i.type.replace('_', ' ')} - {i.description}. NGO should inspect closely.")
        verdict = "urgent_pickup"
    if remaining < URGENT_BELOW_HOURS:
        verdict = "urgent_pickup"
        reasons.append(f"Only {remaining:g}h left in the safe window.")

    reasons.append(f"{hours:g}h since cooking, {STORAGE_LABEL[storage]}; "
                   f"limit {limit}h for {risk}-risk food.")
    return out(verdict)


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS listings ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, items TEXT NOT NULL, "
            "servings INTEGER NOT NULL, veg TEXT NOT NULL, place TEXT NOT NULL, km REAL, "
            "total_hours REAL NOT NULL, expires_at REAL NOT NULL, claimed INTEGER NOT NULL DEFAULT 0, "
            "created_at REAL NOT NULL)"
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
        for col, ddl in (("donor_token", "TEXT"), ("claimed_by", "TEXT"),
                         ("claimed_contact", "TEXT"), ("claimed_at", "REAL")):
            if col not in cols:
                conn.execute(f"ALTER TABLE listings ADD COLUMN {col} {ddl}")
        if conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 0:
            now = time.time()
            seed = [
                ("Hostel mess, Block B", ["Dal tadka", "Jeera rice"], 40, "veg", "Koramangala", 2.1, 3.5, 4),
                ("Aroma Caterers", ["Paneer butter masala", "Chapati"], 60, "veg", "Indiranagar", 4.8, 1, 6),
                ("Campus canteen", ["Idli", "Sambar"], 30, "veg", "Jayanagar", 3.4, 5, 6),
                ("Orchid Banquet Hall", ["Veg biryani", "Raita", "Chicken curry"], 80, "mixed", "Whitefield", 9.6, 2, 4),
                ("Daily Bread Bakery", ["Packaged buns"], 25, "veg", "HSR Layout", 5.2, 7, 8),
            ]
            for title, items, servings, veg, place, km, left, total in seed:
                conn.execute(
                    "INSERT INTO listings (title, items, servings, veg, place, km, total_hours, expires_at, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (title, json.dumps(items), servings, veg, place, km, total, now + left * 3600, now),
                )


def row_to_listing(r: sqlite3.Row) -> dict:
    left = round(max(0.0, (r["expires_at"] - time.time()) / 3600), 1)
    return {
        "id": r["id"], "title": r["title"], "items": json.loads(r["items"]),
        "servings": r["servings"], "veg": r["veg"], "place": r["place"], "km": r["km"],
        "left": left, "total": r["total_hours"], "claimed": bool(r["claimed"]),
    }


def row_to_mine(r: sqlite3.Row) -> dict:
    d = row_to_listing(r)
    d["claimed_by"] = r["claimed_by"]
    d["claimed_contact"] = r["claimed_contact"]
    d["claimed_at"] = r["claimed_at"]
    d["status"] = "claimed" if d["claimed"] else ("expired" if d["left"] <= 0 else "open")
    return d


class ListingIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    items: list[str] = Field(min_length=1, max_length=20)
    servings: int = Field(ge=1, le=100000)
    veg: Literal["veg", "non_veg", "mixed", "unclear"]
    pickup_within_hours: float = Field(gt=0, le=72)
    total_hours: float = Field(gt=0, le=96)
    place: str = Field(default="Your location", max_length=120)
    donor_token: str = Field(default="", max_length=64)


class ClaimIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    contact: str = Field(min_length=3, max_length=60)


_hits: dict[str, list[float]] = {}


def _allow(key: str, limit: int, window: float = 60.0) -> bool:
    now = time.time()
    if len(_hits) > 5000:
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > window]:
            _hits.pop(k, None)
    q = [t for t in _hits.get(key, []) if now - t < window]
    allowed = len(q) < limit
    if allowed:
        q.append(now)
    _hits[key] = q
    return allowed


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")


def limit_ai(request: Request):
    if not _allow("ai:" + client_ip(request), AI_PER_MIN) or not _allow("ai:all", AI_GLOBAL_PER_MIN):
        raise HTTPException(429, "Too many photo checks right now. Wait a minute and try again.")


def limit_write(request: Request):
    if not _allow("w:" + client_ip(request), WRITE_PER_MIN):
        raise HTTPException(429, "Too many requests. Slow down and try again.")


app = FastAPI(title="FoodRescue AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
init_db()


async def _read_and_analyze(image: UploadFile) -> VisionResult:
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")
    try:
        return analyze_food_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/vision", response_model=VisionResult, dependencies=[Depends(limit_ai)])
async def vision(image: UploadFile = File(...)):
    return await _read_and_analyze(image)


@app.post("/analyze", response_model=Verdict, dependencies=[Depends(limit_ai)])
async def analyze(image: UploadFile = File(...),
                  hours_since_cooked: float = Form(...),
                  storage: str = Form("room_temp")):
    if hours_since_cooked < 0:
        raise HTTPException(400, "hours_since_cooked cannot be negative")
    if storage not in LIMIT_HOURS:
        raise HTTPException(400, f"storage must be one of {list(LIMIT_HOURS)}")
    v = await _read_and_analyze(image)
    return evaluate(v, hours_since_cooked, storage)


@app.get("/listings")
def list_listings():
    with db() as conn:
        rows = conn.execute("SELECT * FROM listings ORDER BY expires_at ASC").fetchall()
    out = [row_to_listing(r) for r in rows]
    return [l for l in out if l["claimed"] or l["left"] > 0]


@app.post("/listings", status_code=201, dependencies=[Depends(limit_write)])
def create_listing(body: ListingIn):
    now = time.time()
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO listings (title, items, servings, veg, place, km, total_hours, expires_at, created_at, donor_token) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (body.title.strip(), json.dumps(body.items), body.servings, body.veg, body.place.strip(),
             None, max(body.total_hours, body.pickup_within_hours),
             now + body.pickup_within_hours * 3600, now, body.donor_token),
        )
        row = conn.execute("SELECT * FROM listings WHERE id=?", (cur.lastrowid,)).fetchone()
    return row_to_listing(row)


@app.post("/listings/{listing_id}/claim", dependencies=[Depends(limit_write)])
def claim_listing(listing_id: int, body: ClaimIn):
    now = time.time()
    with db() as conn:
        cur = conn.execute(
            "UPDATE listings SET claimed=1, claimed_by=?, claimed_contact=?, claimed_at=? "
            "WHERE id=? AND claimed=0 AND expires_at>?",
            (body.name.strip(), body.contact.strip(), now, listing_id, now),
        )
        won = cur.rowcount == 1
        row = conn.execute("SELECT * FROM listings WHERE id=?", (listing_id,)).fetchone()
    if won:
        return row_to_listing(row)
    if row is None:
        raise HTTPException(404, "Listing not found")
    if row["claimed"]:
        raise HTTPException(409, "Another NGO just claimed this pickup")
    raise HTTPException(410, "This listing has expired")


@app.get("/my-listings")
def my_listings(token: str):
    if not 16 <= len(token) <= 64:
        raise HTTPException(400, "Invalid donor token")
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM listings WHERE donor_token=? ORDER BY created_at DESC", (token,)
        ).fetchall()
    return [row_to_mine(r) for r in rows]


@app.get("/stats")
def stats():
    with db() as conn:
        rows = conn.execute("SELECT * FROM listings").fetchall()
    now = time.time()
    return {
        "servings_listed": sum(r["servings"] for r in rows),
        "servings_claimed": sum(r["servings"] for r in rows if r["claimed"]),
        "open_listings": sum(1 for r in rows if not r["claimed"] and r["expires_at"] > now),
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FoodRescue AI</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;700;800&display=swap">
<style>
@property --p{syntax:'<number>';inherits:false;initial-value:0}
:root{--bg:#F2F4EF;--card:#fff;--ink:#17231D;--muted:#5E6E65;--line:#D9DFD8;--steel:#E3E8E2;--leaf:#2E6B40;--tur:#F4B31D;--chili:#CF3A27;--grey:#6B7A72}
@media(prefers-color-scheme:dark){:root{--bg:#0F1612;--card:#18211C;--ink:#EAF0EB;--muted:#97A89D;--line:#2B3830;--steel:#222D26;--leaf:#4C9A62;--grey:#8A9A90}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
h1,h2,h3,.brand,button,.num{font-family:"Bricolage Grotesque",system-ui,sans-serif}
header{background:var(--tur);color:#17231D;padding:1.2rem 1rem 4.5rem}
.hd{max-width:1000px;margin:0 auto}
.top{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:.5rem;font-weight:800;font-size:1.1rem}
nav{display:flex;gap:.4rem;background:rgba(23,35,29,.1);padding:.3rem;border-radius:999px}
nav button{width:auto;background:transparent;color:#17231D;border-radius:999px;padding:.5rem 1.1rem;font-size:.95rem;font-weight:700}
nav button[aria-current=page]{background:#17231D;color:#fff}
h1{font-size:clamp(1.9rem,5vw,3rem);line-height:1.05;margin:1.6rem 0 .6rem;font-weight:800;max-width:15ch}
header p{margin:0;max-width:48ch;font-size:1.05rem}
main{max-width:1000px;margin:-3rem auto 3rem;padding:0 1rem}
.split{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:1.5rem;align-items:start}
.panel{background:var(--card);border:1px solid var(--line);border-radius:28px;padding:1.5rem}
.plate{display:block;width:min(260px,72vw);margin:0 auto 1.4rem;cursor:pointer;border-radius:50%}
.ring{--p:0;--vc:var(--line);width:100%;aspect-ratio:1;border-radius:50%;padding:12px;background:conic-gradient(var(--vc) calc(var(--p)*1%),var(--steel) 0);transition:--p .9s ease-out}
.ring.accept{--vc:var(--leaf)}.ring.urgent_pickup{--vc:var(--tur)}.ring.reject{--vc:var(--chili)}.ring.retake{--vc:var(--grey)}
.dish{width:100%;height:100%;border-radius:50%;background:var(--card) center/cover no-repeat;display:grid;place-items:center;text-align:center;box-shadow:inset 0 0 0 7px var(--card),inset 0 0 0 8px var(--line);padding:2rem}
.dish svg{color:var(--leaf)}.dish b{display:block;font-family:"Bricolage Grotesque",system-ui,sans-serif;font-size:1.15rem}
.dish span{display:block;color:var(--muted);font-size:.85rem}
.plate.over .dish{box-shadow:inset 0 0 0 7px var(--card),inset 0 0 0 10px var(--tur)}
.plate:focus-within .ring{outline:3px solid var(--tur);outline-offset:3px}
.field{margin-bottom:1.2rem}
.row{display:flex;justify-content:space-between;align-items:baseline}
label.t,legend{font-weight:700;padding:0}
output{font-family:"Bricolage Grotesque",system-ui,sans-serif;font-weight:800;font-size:1.3rem}
input[type=range]{width:100%;accent-color:var(--leaf);margin:.5rem 0 0}
input[type=text],input[type=number]{width:100%;margin-top:.5rem;border:2px solid var(--line);border-radius:14px;padding:.7rem .9rem;font:inherit;background:var(--card);color:var(--ink)}
input[type=text]:focus-visible,input[type=number]:focus-visible{outline:3px solid var(--tur);outline-offset:2px}
fieldset{border:0;padding:0;margin:0 0 1.4rem}
.seg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:.5rem}
.seg label{position:relative}
.seg input{position:absolute;opacity:0;inset:0}
.seg span{display:block;text-align:center;border:2px solid var(--line);border-radius:14px;padding:.7rem .3rem;font-weight:600;font-size:.9rem;cursor:pointer}
.seg small{display:block;font-weight:400;color:var(--muted);font-size:.75rem}
.seg input:checked+span{border-color:var(--leaf);background:color-mix(in srgb,var(--leaf) 12%,transparent)}
.seg input:focus-visible+span{outline:3px solid var(--tur);outline-offset:2px}
button{width:100%;border:0;border-radius:14px;background:var(--leaf);color:#fff;font-size:1.05rem;font-weight:700;padding:.9rem;cursor:pointer}
button:disabled{opacity:.4;cursor:not-allowed}
button:focus-visible{outline:3px solid var(--tur);outline-offset:3px}
.empty{background:var(--card);border:2px dashed var(--line);border-radius:28px;padding:3rem 1.5rem;text-align:center;color:var(--muted)}
.empty b{display:block;color:var(--ink);font-family:"Bricolage Grotesque",system-ui,sans-serif;font-size:1.2rem;margin-bottom:.3rem}
.spin{width:34px;height:34px;border:4px solid var(--steel);border-top-color:var(--leaf);border-radius:50%;margin:0 auto 1rem;animation:s .8s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
.res{background:var(--card);border:1px solid var(--line);border-radius:28px;overflow:hidden}
.verdict{padding:1.4rem 1.5rem;border-left:10px solid var(--vc);background:color-mix(in srgb,var(--vc) 12%,var(--card))}
.verdict.accept{--vc:var(--leaf)}.verdict.urgent_pickup{--vc:var(--tur)}.verdict.reject{--vc:var(--chili)}.verdict.retake{--vc:var(--grey)}
.verdict h2{margin:0;font-size:1.7rem;line-height:1.1}
.verdict p{margin:.4rem 0 0}.verdict .big{font-size:1.15rem}.verdict .big b{font-size:1.6rem;font-family:"Bricolage Grotesque",system-ui,sans-serif}
.body{padding:1.2rem 1.5rem 1.5rem}
h3{font-size:1rem;margin:1.2rem 0 .4rem}.body>h3:first-child{margin-top:0}
ul{margin:0;padding-left:1.2rem}li{margin:.25rem 0}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;padding:0;list-style:none}
.chips li{margin:0;background:var(--steel);border-radius:999px;padding:.35rem .8rem;font-weight:600}
.chips small{color:var(--muted);font-weight:400;margin-left:.3rem}
.note{margin-top:1.2rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
.dot{display:inline-block;width:.7rem;height:.7rem;border-radius:50%;margin-right:.4rem;background:var(--grey)}
.dot.veg{background:var(--leaf)}.dot.non_veg{background:var(--chili)}.dot.mixed{background:var(--tur)}
.bar{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.2rem}
.pill{width:auto;background:var(--card);color:var(--ink);border:2px solid var(--line);border-radius:999px;padding:.45rem 1rem;font-size:.9rem}
.pill[aria-pressed=true]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:1rem}
.item{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:1.2rem;display:grid;grid-template-columns:84px 1fr;gap:1rem;align-items:start}
.item .ring{width:84px;padding:7px}
.item .ring div{width:100%;height:100%;border-radius:50%;background:var(--card);display:grid;place-items:center;text-align:center;line-height:1}
.item .num{font-size:1.2rem;font-weight:800}.item small.u{display:block;font-size:.7rem;color:var(--muted)}
.item h3{margin:0 0 .2rem;font-size:1.05rem}
.item p{margin:0;color:var(--muted);font-size:.88rem}
.item .meta{margin:.5rem 0 .8rem;font-size:.88rem}
.item button{grid-column:1/-1;padding:.7rem;font-size:.95rem;border-radius:12px}
.item button.done{background:var(--steel);color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}
.stat{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:1.4rem}
.stat .num{font-size:2.6rem;font-weight:800;line-height:1}
.stat p{margin:.4rem 0 0;color:var(--muted)}
.org{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.2rem;padding:1.2rem}
.org .field{margin:0}
.claim{margin-top:.7rem;padding:.7rem .9rem;border-radius:14px;border-left:6px solid var(--leaf);background:color-mix(in srgb,var(--leaf) 14%,var(--card));font-size:.9rem;line-height:1.5}
.item p.wait{margin-top:.7rem;font-weight:600;color:var(--ink)}
.toast{position:fixed;left:50%;bottom:1.5rem;transform:translate(-50%,160%);background:var(--ink);color:var(--bg);padding:.8rem 1.2rem;border-radius:14px;font-weight:600;transition:transform .25s;z-index:5}
.toast.on{transform:translate(-50%,0)}
[hidden]{display:none!important}
@media(max-width:780px){.split,.org{grid-template-columns:minmax(0,1fr)}}
@media(prefers-reduced-motion:reduce){.ring,.toast{transition:none}.spin{animation:none}}
</style></head><body>
<header><div class="hd">
 <div class="top">
  <div class="brand"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 14c0-3.2 2.2-5.4 6.4-5.4 0 4.2-2.2 6.2-6.4 5.4z"/></svg>FoodRescue AI</div>
  <nav aria-label="Sections">
   <button data-view="donate" aria-current="page">Donate food</button>
   <button data-view="mine">My donations</button>
   <button data-view="collect">Collect food</button>
   <button data-view="impact">Impact</button>
  </nav>
 </div>
 <h1 id="title">Leftover food, checked in seconds</h1>
 <p id="sub">Photograph the food, tell us when it was cooked, and see how long NGOs have to collect it.</p>
</div></header>
<main>
<section id="v-donate" class="split">
 <div class="panel">
  <label class="plate" id="plate" for="f"><div class="ring" id="ring"><div class="dish" id="dish"><div id="hint">
   <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>
   <b>Add a photo</b><span>Use the camera or choose a file</span></div></div></div></label>
  <input type="file" id="f" accept="image/*" capture="environment" hidden>
  <div class="field"><div class="row"><label class="t" for="hrs">Cooked how long ago?</label><output id="hv">1 hour</output></div>
   <input type="range" id="hrs" min="0" max="24" step="0.5" value="1"></div>
  <fieldset><legend>How was it kept?</legend><div class="seg">
   <label><input type="radio" name="stor" value="room_temp" checked><span>🌡️<br>Room temp<small>Left out</small></span></label>
   <label><input type="radio" name="stor" value="hot_held"><span>🔥<br>Kept hot<small>Hot box</small></span></label>
   <label><input type="radio" name="stor" value="refrigerated"><span>❄️<br>Fridge<small>Chilled</small></span></label>
  </div></fieldset>
  <div class="field"><label class="t" for="place">Pickup location</label><input type="text" id="place" placeholder="Hostel mess, Block B"></div>
  <button id="go" disabled>Check this food</button>
 </div>
 <section id="out" aria-live="polite"><div class="empty"><b>Your result shows up here</b>Add a photo of the food to see how long it can wait.</div></section>
</section>
<section id="v-mine" hidden>
 <div class="grid" id="mine"></div>
</section>
<section id="v-collect" hidden>
 <div class="panel org">
  <div class="field"><label class="t" for="org">Your organisation</label><input type="text" id="org" placeholder="Hope Foundation" autocomplete="organization"></div>
  <div class="field"><label class="t" for="ph">Phone number</label><input type="text" id="ph" placeholder="98xxxxxxxx" inputmode="tel" autocomplete="tel"></div>
 </div>
 <div class="bar" id="filters">
  <button class="pill" data-f="all" aria-pressed="true">All food</button>
  <button class="pill" data-f="veg" aria-pressed="false">Veg only</button>
  <button class="pill" data-f="urgent" aria-pressed="false">Collect soon</button>
 </div>
 <div class="grid" id="list"></div>
</section>
<section id="v-impact" hidden>
 <div class="stats" id="stats"></div>
</section>
</main>
<div class="toast" id="toast" role="status"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=s=>String(s).split('_').join(' ');
const META={
 accept:{t:'Good to list',s:'It passed the first check. NGOs can collect it.'},
 urgent_pickup:{t:'Collect soon',s:'It is fine for now, but the window is short.'},
 reject:{t:'Do not donate this',s:'It should not be collected.'},
 retake:{t:'Retake the photo',s:'The photo was not clear enough to check.'}};
const HEAD={
 donate:['Leftover food, checked in seconds','Photograph the food, tell us when it was cooked, and see how long NGOs have to collect it.'],
 mine:['Your donations','See who claimed your food and how long is left.'],
 collect:['Food waiting for pickup','Pick a listing and claim it before the ring runs out.'],
 impact:['What we have rescued together','Live totals from listings and pickups.']};
const state={listings:[],mine:[]};
let filter='all',view='donate',mineSeen=null;

function store(k,v){
 try{if(v===undefined)return localStorage.getItem(k)||'';localStorage.setItem(k,v);}catch(e){}
 return '';
}
let token=store('fr_donor');
if(!token){
 token=Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('');
 store('fr_donor',token);
}
$('org').value=store('fr_org');$('ph').value=store('fr_ph');

function toast(m){const t=$('toast');t.textContent=m;t.classList.add('on');clearTimeout(toast.h);toast.h=setTimeout(()=>t.classList.remove('on'),2400);}

async function api(path,opts){
 const r=await fetch(path,opts);
 let d=null;
 try{d=await r.json();}catch(e){}
 if(!r.ok)throw new Error(d&&typeof d.detail==='string'?d.detail:'Something went wrong. Try again.');
 return d;
}

function go(v){
 view=v;
 ['donate','mine','collect','impact'].forEach(x=>{$('v-'+x).hidden=x!==v;});
 document.querySelectorAll('nav button').forEach(b=>{if(b.dataset.view===v)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
 $('title').textContent=HEAD[v][0];$('sub').textContent=HEAD[v][1];
 if(v==='mine'){renderMine();loadMine();}
 if(v==='collect')loadList();
 if(v==='impact')loadStats();
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>go(b.dataset.view));

async function loadMine(){
 try{
  const rows=await api('/my-listings?token='+token);
  if(mineSeen!==null)rows.forEach(r=>{if(r.status==='claimed'&&!mineSeen.has(r.id))toast(r.claimed_by+' claimed your donation: '+r.title);});
  mineSeen=new Set(rows.filter(r=>r.status==='claimed').map(r=>r.id));
  state.mine=rows;
  if(view==='mine')renderMine();
 }catch(e){}
}

function renderMine(){
 const rows=state.mine;
 if(!rows.length){$('mine').innerHTML='<div class="empty" style="grid-column:1/-1"><b>No donations yet</b>Check some food on the Donate tab and list it for pickup.</div>';return;}
 $('mine').innerHTML=rows.map(l=>{
  const claimed=l.status==='claimed',open=l.status==='open';
  const pct=claimed?100:Math.max(6,Math.round(l.left/l.total*100));
  const tone=claimed?'accept':open?(l.left<1.5?'urgent_pickup':'accept'):'reject';
  const mid=claimed?'<span class="num">✓</span><small class="u">claimed</small>':open?'<span class="num">'+l.left+'</span><small class="u">hours left</small>':'<span class="num">0</span><small class="u">expired</small>';
  const st=claimed?'<div class="claim"><b>'+esc(l.claimed_by)+'</b> will collect this.<br>Contact: <b>'+esc(l.claimed_contact)+'</b></div>':open?'<p class="wait">Waiting for an NGO to claim it</p>':'<p class="wait">Expired before anyone collected it</p>';
  return '<article class="item"><div class="ring '+tone+'" style="--p:'+pct+'"><div><span>'+mid+'</span></div></div><div><h3>'+esc(l.title)+'</h3><p>'+esc(l.items.join(', '))+'</p><p class="meta"><i class="dot '+esc(l.veg)+'"></i>'+esc(pretty(l.veg))+' · <b>'+l.servings+' servings</b> · '+esc(l.place)+'</p>'+st+'</div></article>';}).join('');
}

async function loadList(){
 try{
  const d=await api('/listings'),k=JSON.stringify(d);
  if(k!==loadList.k){loadList.k=k;state.listings=d;renderList();}
 }
 catch(e){loadList.k=null;$('list').innerHTML='<div class="empty" style="grid-column:1/-1"><b>Could not load listings</b>'+esc(e.message)+'</div>';}
}

function renderList(){
 const rows=state.listings.filter(l=>filter==='veg'?l.veg==='veg':filter==='urgent'?(!l.claimed&&l.left<1.5):true);
 if(!rows.length){$('list').innerHTML='<div class="empty" style="grid-column:1/-1"><b>Nothing matches</b>Try another filter or check back soon.</div>';return;}
 $('list').innerHTML=rows.map(l=>{
  const pct=Math.max(6,Math.round(l.left/l.total*100)),tone=l.left<1.5?'urgent_pickup':'accept';
  return '<article class="item"><div class="ring '+tone+'" style="--p:'+pct+'"><div><span><span class="num">'+l.left+'</span><small class="u">hours left</small></span></div></div>'+
  '<div><h3>'+esc(l.title)+'</h3><p>'+esc(l.items.join(', '))+'</p><p class="meta"><i class="dot '+esc(l.veg)+'"></i>'+esc(pretty(l.veg))+' · <b>'+l.servings+' servings</b> · '+esc(l.place)+(l.km?' · '+l.km+' km':'')+'</p></div>'+
  '<button data-id="'+l.id+'" class="'+(l.claimed?'done':'')+'"'+(l.claimed?' disabled':'')+'>'+(l.claimed?'Pickup claimed':'Claim pickup')+'</button></article>';}).join('');
 $('list').querySelectorAll('button[data-id]').forEach(b=>b.onclick=async()=>{
  const name=$('org').value.trim(),contact=$('ph').value.trim();
  if(!name||contact.length<3){toast('Enter your organisation and phone number first');$('org').focus();return;}
  store('fr_org',name);store('fr_ph',contact);
  b.disabled=true;
  try{
   await api('/listings/'+b.dataset.id+'/claim',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,contact})});
   toast('Pickup claimed');
  }catch(e){toast(e.message);}
  loadList();
 });
}
$('filters').querySelectorAll('.pill').forEach(p=>p.onclick=()=>{
 filter=p.dataset.f;
 $('filters').querySelectorAll('.pill').forEach(x=>x.setAttribute('aria-pressed',x===p));
 renderList();
});

async function loadStats(){
 try{
  const s=await api('/stats');
  $('stats').innerHTML=[[s.servings_listed,'servings listed'],[s.servings_claimed,'servings claimed by NGOs'],[s.open_listings,'listings still waiting']].map(x=>'<div class="stat"><div class="num">'+x[0]+'</div><p>'+x[1]+'</p></div>').join('');
 }catch(e){$('stats').innerHTML='<div class="empty"><b>Could not load totals</b>'+esc(e.message)+'</div>';}
}

const f=$('f'),btn=$('go'),out=$('out'),ring=$('ring'),dish=$('dish'),hint=$('hint'),hrs=$('hrs'),hv=$('hv'),plate=$('plate');
hrs.oninput=()=>{const v=parseFloat(hrs.value);hv.textContent=v===1?'1 hour':v+' hours';};
function setFile(file){
 if(!file)return;
 dish.style.backgroundImage='url("'+URL.createObjectURL(file)+'")';
 hint.style.display='none';btn.disabled=false;
 ring.className='ring';ring.style.setProperty('--p',0);
}
f.onchange=()=>setFile(f.files[0]);
['dragenter','dragover'].forEach(n=>plate.addEventListener(n,e=>{e.preventDefault();plate.classList.add('over');}));
['dragleave','drop'].forEach(n=>plate.addEventListener(n,()=>plate.classList.remove('over')));
plate.addEventListener('drop',e=>{e.preventDefault();if(e.dataTransfer.files[0]){f.files=e.dataTransfer.files;setFile(f.files[0]);}});
const err=m=>'<div class="empty"><b>That did not work</b>'+esc(m)+'</div>';

btn.onclick=async()=>{
 out.innerHTML='<div class="empty"><div class="spin"></div>Looking at the food…</div>';
 btn.disabled=true;btn.textContent='Checking…';
 ring.className='ring';ring.style.setProperty('--p',0);
 const fd=new FormData();
 fd.append('image',f.files[0]);
 fd.append('hours_since_cooked',hrs.value);
 fd.append('storage',document.querySelector('input[name=stor]:checked').value);
 try{
  const d=await api('/analyze',{method:'POST',body:fd});
  show(d,parseFloat(hrs.value));
 }catch(e){out.innerHTML=err(e instanceof TypeError?'The server is not responding. Check that it is still running.':e.message);}
 btn.disabled=false;btn.textContent='Check this food';
 if(window.innerWidth<780)out.scrollIntoView({behavior:'smooth'});
};

function show(d,elapsed){
 const m=META[d.verdict],v=d.vision,left=d.pickup_within_hours;
 const live=d.verdict==='accept'||d.verdict==='urgent_pickup';
 const total=elapsed+left,pct=live&&total>0?Math.max(6,Math.round(left/total*100)):100;
 ring.className='ring '+d.verdict;
 requestAnimationFrame(()=>ring.style.setProperty('--p',pct));
 let h='<div class="res"><div class="verdict '+d.verdict+'"><h2>'+m.t+'</h2>';
 h+=live?'<p class="big">Collect within <b>'+left+' hours</b> · about <b>'+v.estimated_servings+' servings</b></p><p>'+m.s+'</p>':'<p>'+m.s+'</p>';
 h+='</div><div class="body"><h3>Why</h3><ul>'+d.reasons.map(r=>'<li>'+esc(r)+'</li>').join('')+'</ul>';
 if(d.verdict!=='retake'){
  h+='<h3>What we found</h3><ul class="chips">'+v.items.map(i=>'<li>'+esc(i.name)+'<small>'+esc(pretty(i.category))+'</small></li>').join('')+'</ul>';
  h+='<h3>Details</h3><ul><li>About '+v.estimated_servings+' servings</li><li><i class="dot '+esc(v.veg_status)+'"></i>'+esc(pretty(v.veg_status))+'</li><li>Container: '+esc(pretty(v.container))+'</li><li>AI confidence: '+Math.round(v.confidence*100)+'%</li></ul>';
  if(v.visible_issues.length)h+='<h3>Problems spotted</h3><ul>'+v.visible_issues.map(i=>'<li>'+esc(pretty(i.type))+' ('+esc(i.severity)+'): '+esc(i.description)+'</li>').join('')+'</ul>';
 }
 if(live)h+='<div style="margin-top:1.2rem"><div class="field"><label class="t" for="srv">Servings you are donating</label><input type="number" id="srv" min="1" max="100000" value="'+Math.max(1,v.estimated_servings)+'"></div><button id="list-it">List for pickup</button></div>';
 h+='<p class="note">'+esc(d.disclaimer)+'</p></div></div>';
 out.innerHTML=h;
 const b=$('list-it');
 if(b)b.onclick=async()=>{
  b.disabled=true;
  try{
   await api('/listings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    title:$('place').value.trim()||'New donation',
    items:v.items.map(i=>i.name),
    servings:Math.max(1,parseInt($('srv').value,10)||1),
    veg:v.veg_status,
    pickup_within_hours:left,
    total_hours:Math.max(left,total),
    place:$('place').value.trim()||'Your location',
    donor_token:token})});
   b.textContent='Listed for NGOs';toast('Listed for pickup');loadMine();
  }catch(e){b.disabled=false;toast(e.message);}
 };
}
setInterval(()=>{
 if(document.hidden)return;
 loadMine();
 if(view==='collect')loadList();
 if(view==='impact')loadStats();
},3000);
loadMine();
</script></body></html>"""


if __name__ == "__main__":
    import socket

    import uvicorn

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        lan_ip = probe.getsockname()[0]
    except OSError:
        lan_ip = "127.0.0.1"
    finally:
        probe.close()
    port = int(os.getenv("PORT", "8000"))
    print(f"Open on this laptop: http://localhost:{port}")
    print(f"Open on other devices (same Wi-Fi): http://{lan_ip}:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
