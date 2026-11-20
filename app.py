"""Flask API — Real Estate Agent System backend."""
from __future__ import annotations

import json
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Data models ────────────────────────────────────────────────────────────

@dataclass
class ClientProfile:
    client_id: str
    name: str
    min_beds: int = 2
    max_beds: int = 5
    min_price: float = 200_000
    max_price: float = 700_000
    zip_codes: list[str] = field(default_factory=list)
    preference_weights: dict[str, float] = field(default_factory=lambda: {
        "price_value":  0.30,
        "neighborhood": 0.25,
        "size_match":   0.20,
        "condition":    0.15,
        "listing_age":  0.10,
    })
    feedback_notes: list[str] = field(default_factory=list)


@dataclass
class Listing:
    mls_id: str
    address: str
    price: float
    beds: int
    baths: float
    sqft: int
    zip_code: str
    days_on_market: int
    condition: str
    listing_url: str
    description: str = ""
    relevance_score: float = 0.0


@dataclass
class EmailDraft:
    draft_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_id: str = ""
    subject: str = ""
    body: str = ""
    listings_included: list[str] = field(default_factory=list)
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class FeedbackRecord:
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_id: str = ""
    mls_id: str = ""
    raw_text: str = ""
    sentiment: str = "neutral"
    extracted_preferences: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# ── Stub MLS data ──────────────────────────────────────────────────────────

STUB_LISTINGS: list[Listing] = [
    Listing("MLS001", "12 Oak St, Springfield",     425_000, 3, 2.0, 1500, "62701",  5,  "move_in_ready", "https://mls.example/MLS001", "Bright open floor plan."),
    Listing("MLS002", "88 Elm Ave, Springfield",    380_000, 4, 2.5, 1900, "62702",  2,  "updated",       "https://mls.example/MLS002", "Recently remodelled kitchen."),
    Listing("MLS003", "5 Pine Rd, Shelbyville",     290_000, 3, 1.0, 1100, "62703", 30,  "fixer_upper",   "https://mls.example/MLS003", "Great bones, needs TLC."),
    Listing("MLS004", "200 Maple Ln, Springfield",  650_000, 5, 3.0, 2800, "62701",  1,  "move_in_ready", "https://mls.example/MLS004", "Luxury master suite."),
    Listing("MLS005", "44 Birch Blvd, Capital City",310_000, 3, 2.0, 1350, "62706", 10,  "updated",       "https://mls.example/MLS005", "Corner lot, large yard."),
]


class MLSClient:
    def search(self, profile: ClientProfile) -> list[Listing]:
        return [
            l for l in STUB_LISTINGS
            if profile.min_price <= l.price <= profile.max_price
            and profile.min_beds <= l.beds <= profile.max_beds
            and (not profile.zip_codes or l.zip_code in profile.zip_codes)
        ]


# ── Agent implementations ──────────────────────────────────────────────────

class Agent1_SearchCuration:
    IDEAL_SQFT = 1_600
    CONDITION_SCORE = {"move_in_ready": 1.0, "updated": 0.7, "fixer_upper": 0.3}
    MAX_DAYS = 60

    def __init__(self, mls: MLSClient):
        self.mls = mls

    def run(self, profile: ClientProfile, top_n: int = 3) -> list[Listing]:
        candidates = self.mls.search(profile)
        scored = sorted([self._score(l, profile) for l in candidates],
                        key=lambda l: l.relevance_score, reverse=True)
        return scored[:top_n]

    def _score(self, listing: Listing, profile: ClientProfile) -> Listing:
        w = profile.preference_weights
        mid    = (profile.min_price + profile.max_price) / 2
        spread = (profile.max_price - profile.min_price) / 2
        scores = {
            "price_value":  max(0.0, 1.0 - abs(listing.price - mid) / spread),
            "neighborhood": 1.0 if listing.zip_code in profile.zip_codes else 0.5,
            "size_match":   max(0.0, 1.0 - abs(listing.sqft - self.IDEAL_SQFT) / self.IDEAL_SQFT),
            "condition":    self.CONDITION_SCORE.get(listing.condition, 0.5),
            "listing_age":  max(0.0, 1.0 - listing.days_on_market / self.MAX_DAYS),
        }
        listing.relevance_score = round(sum(scores[d] * w.get(d, 0.0) for d in scores), 4)
        return listing


class Agent2_EmailDrafting:
    def __init__(self):
        self.queue: list[EmailDraft] = []

    def compose(self, profile: ClientProfile, listings: list[Listing]) -> EmailDraft:
        blocks = []
        for i, l in enumerate(listings, 1):
            blocks.append(
                f"{i}. {l.address}  |  ${l.price:,.0f}  |  {l.beds}bd/{l.baths}ba  |  {l.sqft:,} sqft\n"
                f"   Condition   : {l.condition.replace('_',' ').title()}\n"
                f"   Days Listed : {l.days_on_market}\n"
                f"   Description : {l.description}\n"
                f"   View Listing: {l.listing_url}\n"
                f"   Relevance   : {l.relevance_score:.2f}"
            )
        body = (
            f"Hi {profile.name},\n\n"
            "I've curated the following properties based on your search criteria.\n"
            "Please review and let me know your thoughts on each one!\n\n"
            + "\n\n".join(blocks) +
            "\n\nReply with any feedback — what you liked, what didn't fit, "
            "or if you'd like to schedule a showing.\n\nBest,\nSarah"
        )
        draft = EmailDraft(
            client_id=profile.client_id,
            subject=f"Your Personalised Property Picks — {datetime.now().strftime('%b %d')}",
            body=body,
            listings_included=[l.mls_id for l in listings],
        )
        self.queue.append(draft)
        return draft

    def approve(self, draft_id: str) -> EmailDraft:
        draft = self._find(draft_id)
        draft.status = "approved"
        return draft

    def reject(self, draft_id: str) -> EmailDraft:
        draft = self._find(draft_id)
        draft.status = "rejected"
        return draft

    def _find(self, draft_id: str) -> EmailDraft:
        for d in self.queue:
            if d.draft_id == draft_id:
                return d
        raise KeyError(draft_id)


class Agent3_FeedbackTuning:
    POSITIVE_SIGNALS = {
        r"\bprice\b|\baffordable\b|\bbargain\b":    "price_value",
        r"\bneighborhood\b|\blocation\b|\barea\b":  "neighborhood",
        r"\bspacious\b|\broom\b|\bsqft\b":          "size_match",
        r"\bmove.in\b|\bready\b|\bupdated\b":        "condition",
        r"\bnew listing\b|\bjust listed\b":           "listing_age",
    }
    NEGATIVE_SIGNALS = {
        r"\btoo expensive\b|\bover budget\b":        "price_value",
        r"\bbad area\b|\bwrong neighborhood\b":       "neighborhood",
        r"\btoo small\b|\btiny\b":                   "size_match",
        r"\bfixer\b|\bneeds work\b|\brun.?down\b":  "condition",
        r"\bold listing\b|\bsitting\b":               "listing_age",
    }
    NUDGE = 0.05; MIN = 0.05; MAX = 0.60

    def __init__(self):
        self.records: list[FeedbackRecord] = []

    def run(self, profile: ClientProfile, mls_id: str, text: str) -> FeedbackRecord:
        text_l = text.lower()
        pos = [d for p, d in self.POSITIVE_SIGNALS.items() if re.search(p, text_l)]
        neg = [d for p, d in self.NEGATIVE_SIGNALS.items() if re.search(p, text_l)]
        sentiment = "positive" if len(pos) > len(neg) else "negative" if len(neg) > len(pos) else "neutral"
        record = FeedbackRecord(
            client_id=profile.client_id, mls_id=mls_id, raw_text=text,
            sentiment=sentiment, extracted_preferences={"positive": pos, "negative": neg},
        )
        w = profile.preference_weights
        for d in pos: w[d] = min(self.MAX, round(w.get(d, 0.1) + self.NUDGE, 4))
        for d in neg: w[d] = max(self.MIN, round(w.get(d, 0.1) - self.NUDGE, 4))
        total = sum(w.values())
        profile.preference_weights = {k: round(v / total, 4) for k, v in w.items()}
        note = (
            f"[{record.timestamp}] MLS {mls_id} — {sentiment}. "
            f"Boosted: {pos}. Reduced: {neg}. Raw: \"{text[:100]}\""
        )
        profile.feedback_notes.append(note)
        self.records.append(record)
        return record


# ── In-memory session store ────────────────────────────────────────────────

_sessions: dict[str, dict] = {}   # client_id -> { profile, agent2, agent3 }


def _get_or_create(client_id: str, name: str = "") -> dict:
    if client_id not in _sessions:
        profile = ClientProfile(client_id=client_id, name=name or client_id)
        _sessions[client_id] = {
            "profile": profile,
            "agent2":  Agent2_EmailDrafting(),
            "agent3":  Agent3_FeedbackTuning(),
        }
    return _sessions[client_id]


def _profile_dict(p: ClientProfile) -> dict:
    return {
        "client_id": p.client_id, "name": p.name,
        "min_beds": p.min_beds, "max_beds": p.max_beds,
        "min_price": p.min_price, "max_price": p.max_price,
        "zip_codes": p.zip_codes,
        "preference_weights": p.preference_weights,
        "feedback_notes": p.feedback_notes,
    }


def _draft_dict(d: EmailDraft) -> dict:
    return {
        "draft_id": d.draft_id, "client_id": d.client_id,
        "subject": d.subject, "body": d.body,
        "listings_included": d.listings_included,
        "status": d.status, "created_at": d.created_at,
    }


def _listing_dict(l: Listing) -> dict:
    return {
        "mls_id": l.mls_id, "address": l.address, "price": l.price,
        "beds": l.beds, "baths": l.baths, "sqft": l.sqft,
        "zip_code": l.zip_code, "days_on_market": l.days_on_market,
        "condition": l.condition, "listing_url": l.listing_url,
        "description": l.description, "relevance_score": l.relevance_score,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    """Agent 1 + 2: create/update a client profile and run a search."""
    data = request.get_json(force=True)
    client_id = data.get("client_id") or str(uuid.uuid4())[:8]
    sess = _get_or_create(client_id, data.get("name", ""))
    p = sess["profile"]

    # Update profile fields from request
    p.name      = data.get("name", p.name)
    p.min_beds  = int(data.get("min_beds",  p.min_beds))
    p.max_beds  = int(data.get("max_beds",  p.max_beds))
    p.min_price = float(data.get("min_price", p.min_price))
    p.max_price = float(data.get("max_price", p.max_price))
    raw_zips    = data.get("zip_codes", "")
    p.zip_codes = [z.strip() for z in raw_zips.split(",") if z.strip()] if isinstance(raw_zips, str) else raw_zips
    top_n       = int(data.get("top_n", 3))

    mls    = MLSClient()
    agent1 = Agent1_SearchCuration(mls)
    agent2 = sess["agent2"]

    listings = agent1.run(p, top_n=top_n)
    draft    = agent2.compose(p, listings)

    return jsonify({
        "client_id": client_id,
        "profile":   _profile_dict(p),
        "listings":  [_listing_dict(l) for l in listings],
        "draft":     _draft_dict(draft),
    })


@app.route("/api/draft/<draft_id>/approve", methods=["POST"])
def api_approve(draft_id: str):
    data      = request.get_json(force=True)
    client_id = data.get("client_id", "")
    if client_id not in _sessions:
        return jsonify({"error": "session not found"}), 404
    draft = _sessions[client_id]["agent2"].approve(draft_id)
    return jsonify(_draft_dict(draft))


@app.route("/api/draft/<draft_id>/reject", methods=["POST"])
def api_reject(draft_id: str):
    data      = request.get_json(force=True)
    client_id = data.get("client_id", "")
    if client_id not in _sessions:
        return jsonify({"error": "session not found"}), 404
    draft = _sessions[client_id]["agent2"].reject(draft_id)
    return jsonify(_draft_dict(draft))


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Agent 3: parse buyer feedback and re-tune weights."""
    data      = request.get_json(force=True)
    client_id = data.get("client_id", "")
    if client_id not in _sessions:
        return jsonify({"error": "session not found"}), 404
    sess   = _sessions[client_id]
    record = sess["agent3"].run(sess["profile"], data["mls_id"], data["feedback_text"])
    return jsonify({
        "sentiment":  record.sentiment,
        "boosted":    record.extracted_preferences.get("positive", []),
        "reduced":    record.extracted_preferences.get("negative", []),
        "weights":    sess["profile"].preference_weights,
        "notes":      sess["profile"].feedback_notes,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
