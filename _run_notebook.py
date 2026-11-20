from __future__ import annotations

import json
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
        "price_value":    0.30,
        "neighborhood":  0.25,
        "size_match":    0.20,
        "condition":     0.15,
        "listing_age":   0.10,
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
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))


@dataclass
class FeedbackRecord:
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_id: str = ""
    mls_id: str = ""
    raw_text: str = ""
    sentiment: str = "neutral"
    extracted_preferences: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))


STUB_LISTINGS: list[Listing] = [
    Listing("MLS001", "12 Oak St, Springfield",    425_000, 3, 2.0, 1500, "62701", 5,  "move_in_ready", "https://mls.example/MLS001", "Bright open floor plan."),
    Listing("MLS002", "88 Elm Ave, Springfield",   380_000, 4, 2.5, 1900, "62702", 2,  "updated",       "https://mls.example/MLS002", "Recently remodelled kitchen."),
    Listing("MLS003", "5 Pine Rd, Shelbyville",    290_000, 3, 1.0, 1100, "62703", 30, "fixer_upper",   "https://mls.example/MLS003", "Great bones, needs TLC."),
    Listing("MLS004", "200 Maple Ln, Springfield", 650_000, 5, 3.0, 2800, "62701", 1,  "move_in_ready", "https://mls.example/MLS004", "Luxury master suite."),
    Listing("MLS005", "44 Birch Blvd, Capital City", 310_000, 3, 2.0, 1350, "62706", 10, "updated",     "https://mls.example/MLS005", "Corner lot, large yard."),
]


class MLSClient:
    def search(self, profile: ClientProfile) -> list[Listing]:
        results = []
        for listing in STUB_LISTINGS:
            price_ok = profile.min_price <= listing.price <= profile.max_price
            beds_ok  = profile.min_beds  <= listing.beds  <= profile.max_beds
            zip_ok   = not profile.zip_codes or listing.zip_code in profile.zip_codes
            if price_ok and beds_ok and zip_ok:
                results.append(listing)
        return results


class Agent1_SearchCuration:
    IDEAL_SQFT = 1_600
    CONDITION_SCORE = {"move_in_ready": 1.0, "updated": 0.7, "fixer_upper": 0.3}
    MAX_DAYS = 60

    def __init__(self, mls_client: MLSClient):
        self.mls = mls_client

    def run(self, profile: ClientProfile, top_n: int = 3) -> list[Listing]:
        candidates = self.mls.search(profile)
        scored = [self._score(listing, profile) for listing in candidates]
        scored.sort(key=lambda l: l.relevance_score, reverse=True)
        print(f"[Agent 1] Found {len(candidates)} candidates -> returning top {min(top_n, len(scored))}.")
        return scored[:top_n]

    def _score(self, listing: Listing, profile: ClientProfile) -> Listing:
        w = profile.preference_weights
        scores = {
            "price_value":  self._price_score(listing.price, profile),
            "neighborhood": 1.0 if listing.zip_code in profile.zip_codes else 0.5,
            "size_match":   self._size_score(listing.sqft),
            "condition":    self.CONDITION_SCORE.get(listing.condition, 0.5),
            "listing_age":  max(0.0, 1.0 - listing.days_on_market / self.MAX_DAYS),
        }
        listing.relevance_score = round(
            sum(scores[dim] * w.get(dim, 0.0) for dim in scores), 4
        )
        return listing

    def _price_score(self, price: float, profile: ClientProfile) -> float:
        mid    = (profile.min_price + profile.max_price) / 2
        spread = (profile.max_price - profile.min_price) / 2
        return max(0.0, 1.0 - abs(price - mid) / spread)

    def _size_score(self, sqft: int) -> float:
        return max(0.0, 1.0 - abs(sqft - self.IDEAL_SQFT) / self.IDEAL_SQFT)


class Agent2_EmailDrafting:
    def __init__(self):
        self.queue: list[EmailDraft] = []

    def run(self, profile: ClientProfile, listings: list[Listing]) -> EmailDraft:
        draft = self._compose(profile, listings)
        self.queue.append(draft)
        print(f"[Agent 2] Draft {draft.draft_id} added to queue (status: {draft.status}).")
        return draft

    def show_queue(self) -> None:
        pending = [d for d in self.queue if d.status == "pending"]
        if not pending:
            print("[Agent 2] No drafts awaiting review.")
            return
        for draft in pending:
            print("=" * 60)
            print(f"Draft ID : {draft.draft_id}  |  Created: {draft.created_at}")
            print(f"Subject  : {draft.subject}")
            print("-" * 60)
            print(draft.body)
            print("=" * 60)

    def approve(self, draft_id: str) -> EmailDraft:
        draft = self._find(draft_id)
        draft.status = "approved"
        print(f"[Agent 2] Draft {draft_id} APPROVED. Ready to send.")
        return draft

    def reject(self, draft_id: str, reason: str = "") -> EmailDraft:
        draft = self._find(draft_id)
        draft.status = "rejected"
        print(f"[Agent 2] Draft {draft_id} REJECTED. Reason: {reason or 'none given'}")
        return draft

    def _compose(self, profile: ClientProfile, listings: list[Listing]) -> EmailDraft:
        listing_blocks = []
        for i, l in enumerate(listings, 1):
            block = textwrap.dedent(f"""
                {i}. {l.address}  |  ${l.price:,.0f}  |  {l.beds}bd / {l.baths}ba  |  {l.sqft:,} sqft
                   Condition   : {l.condition.replace('_', ' ').title()}
                   Days Listed : {l.days_on_market}
                   Description : {l.description}
                   View Listing: {l.listing_url}
                   Relevance   : {l.relevance_score:.2f}
            """).strip()
            listing_blocks.append(block)

        body = textwrap.dedent(f"""
            Hi {profile.name},

            I've curated the following properties based on your search criteria.
            Please review and let me know your thoughts on each one!

            {chr(10).join(listing_blocks)}

            Reply to this email with any feedback -- what you liked, what didn't fit,
            or if you'd like to schedule a showing.

            Best,
            Sarah
        """).strip()

        return EmailDraft(
            client_id=profile.client_id,
            subject=f"Your Personalised Property Picks -- {datetime.now().strftime('%b %d')}",
            body=body,
            listings_included=[l.mls_id for l in listings],
        )

    def _find(self, draft_id: str) -> EmailDraft:
        for d in self.queue:
            if d.draft_id == draft_id:
                return d
        raise KeyError(f"Draft '{draft_id}' not found in queue.")


class Agent3_FeedbackTuning:
    POSITIVE_SIGNALS: dict[str, str] = {
        r"\bprice\b|\baffordable\b|\bbargain\b":    "price_value",
        r"\bneighborhood\b|\blocation\b|\barea\b":  "neighborhood",
        r"\bspacious\b|\broom\b|\bsqft\b":          "size_match",
        r"\bmove.in\b|\bready\b|\bupdated\b":        "condition",
        r"\bnew listing\b|\bjust listed\b":           "listing_age",
    }
    NEGATIVE_SIGNALS: dict[str, str] = {
        r"\btoo expensive\b|\bover budget\b":        "price_value",
        r"\bbad area\b|\bwrong neighborhood\b":       "neighborhood",
        r"\btoo small\b|\btiny\b":                   "size_match",
        r"\bfixer\b|\bneeds work\b|\brun.?down\b":  "condition",
        r"\bold listing\b|\bsitting\b":               "listing_age",
    }
    WEIGHT_NUDGE = 0.05
    WEIGHT_MIN   = 0.05
    WEIGHT_MAX   = 0.60

    def __init__(self):
        self.records: list[FeedbackRecord] = []

    def run(self, profile: ClientProfile, mls_id: str, raw_feedback: str) -> FeedbackRecord:
        record = self._parse(profile, mls_id, raw_feedback)
        self._update_weights(profile, record)
        self._append_note(profile, record)
        self.records.append(record)
        print(f"[Agent 3] Feedback logged ({record.sentiment}) for {mls_id}. "
              f"Weights updated -> {profile.preference_weights}")
        return record

    def print_log(self, client_id: str | None = None) -> None:
        records = [r for r in self.records if client_id is None or r.client_id == client_id]
        if not records:
            print("[Agent 3] No feedback records yet.")
            return
        for r in records:
            print(f"  [{r.timestamp}] {r.client_id} / {r.mls_id} -- "
                  f"{r.sentiment.upper()} | {r.raw_text[:80]}...")

    def _parse(self, profile: ClientProfile, mls_id: str, text: str) -> FeedbackRecord:
        text_lower = text.lower()
        pos_hits, neg_hits = self._extract_preferences(text_lower)
        sentiment = (
            "positive" if len(pos_hits) > len(neg_hits)
            else "negative" if len(neg_hits) > len(pos_hits)
            else "neutral"
        )
        return FeedbackRecord(
            client_id=profile.client_id,
            mls_id=mls_id,
            raw_text=text,
            sentiment=sentiment,
            extracted_preferences={"positive": pos_hits, "negative": neg_hits},
        )

    def _extract_preferences(self, text: str) -> tuple[list[str], list[str]]:
        pos = [dim for pattern, dim in self.POSITIVE_SIGNALS.items() if re.search(pattern, text)]
        neg = [dim for pattern, dim in self.NEGATIVE_SIGNALS.items() if re.search(pattern, text)]
        return pos, neg

    def _update_weights(self, profile: ClientProfile, record: FeedbackRecord) -> None:
        w = profile.preference_weights
        for dim in record.extracted_preferences.get("positive", []):
            w[dim] = min(self.WEIGHT_MAX, round(w.get(dim, 0.1) + self.WEIGHT_NUDGE, 4))
        for dim in record.extracted_preferences.get("negative", []):
            w[dim] = max(self.WEIGHT_MIN, round(w.get(dim, 0.1) - self.WEIGHT_NUDGE, 4))
        total = sum(w.values())
        profile.preference_weights = {k: round(v / total, 4) for k, v in w.items()}

    def _append_note(self, profile: ClientProfile, record: FeedbackRecord) -> None:
        note = (
            f"[{record.timestamp}] MLS {record.mls_id} -- {record.sentiment}. "
            f"Boosted: {record.extracted_preferences.get('positive', [])}. "
            f"Reduced: {record.extracted_preferences.get('negative', [])}.  "
            f"Raw: \"{record.raw_text[:100]}\""
        )
        profile.feedback_notes.append(note)


class RealEstateOrchestrator:
    def __init__(self):
        self.mls    = MLSClient()
        self.agent1 = Agent1_SearchCuration(self.mls)
        self.agent2 = Agent2_EmailDrafting()
        self.agent3 = Agent3_FeedbackTuning()

    def on_new_criteria(self, profile: ClientProfile, top_n: int = 3) -> EmailDraft:
        print(f"\n{'='*60}")
        print(f"TRIGGER: new_criteria  |  Client: {profile.name}")
        print(f"{'='*60}")
        listings = self.agent1.run(profile, top_n=top_n)
        draft    = self.agent2.run(profile, listings)
        return draft

    def on_new_listing(self, profile: ClientProfile, top_n: int = 3) -> EmailDraft:
        print(f"\n{'='*60}")
        print(f"TRIGGER: new_listing  |  Client: {profile.name}")
        print(f"{'='*60}")
        listings = self.agent1.run(profile, top_n=top_n)
        draft    = self.agent2.run(profile, listings)
        return draft

    def on_feedback(self, profile: ClientProfile, mls_id: str, raw_feedback: str) -> FeedbackRecord:
        print(f"\n{'='*60}")
        print(f"TRIGGER: feedback  |  Client: {profile.name}  |  Listing: {mls_id}")
        print(f"{'='*60}")
        return self.agent3.run(profile, mls_id, raw_feedback)

    def show_pending_drafts(self) -> None:
        self.agent2.show_queue()

    def approve_draft(self, draft_id: str) -> EmailDraft:
        return self.agent2.approve(draft_id)

    def reject_draft(self, draft_id: str, reason: str = "") -> EmailDraft:
        return self.agent2.reject(draft_id, reason)


# ── STEP 1: New search criteria trigger ────────────────────────────────────
buyer = ClientProfile(
    client_id="buyer_001",
    name="The Johnson Family",
    min_beds=3,
    max_beds=5,
    min_price=300_000,
    max_price=600_000,
    zip_codes=["62701", "62702"],
)

orchestrator = RealEstateOrchestrator()
draft = orchestrator.on_new_criteria(buyer, top_n=3)

print("\n--- Draft in Review Queue ---")
orchestrator.show_pending_drafts()

# ── STEP 2: Sarah approves the draft ───────────────────────────────────────
orchestrator.approve_draft(draft.draft_id)

# ── STEP 3: Buyer feedback ─────────────────────────────────────────────────
feedback_text = (
    "Hi Sarah! We really liked the price and the neighborhood on MLS001, "
    "but it felt a little too small for us. The updated kitchen on MLS002 "
    "looked great though -- we'd love to see more move-in ready homes!"
)

orchestrator.on_feedback(buyer, mls_id="MLS001", raw_feedback=feedback_text)

print("\n--- Updated Preference Weights ---")
print(json.dumps(buyer.preference_weights, indent=2))

print("\n--- Feedback Notes Logged for Sarah ---")
for note in buyer.feedback_notes:
    print(" *", note)

# ── STEP 4: Re-run with updated weights ────────────────────────────────────
print("\nRe-running search with adjusted preference weights...")
draft2 = orchestrator.on_new_criteria(buyer, top_n=3)
orchestrator.show_pending_drafts()
