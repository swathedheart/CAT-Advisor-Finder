import streamlit as st
import pandas as pd
import glob
import os
import re
from datetime import datetime
from typing import List, Tuple, Dict

# --- Header / Branding ---
st.set_page_config(page_title="CAT Advisor Finder", page_icon="🐱", layout="wide")
st.title("🐱 CAT Advisor Finder")
st.caption("Search your Rolodex CSVs by Entity ID or BR Team Name, optionally filtered by Office City.")

# --- Columns to output (in your requested order) ---
OUTPUT_COLUMNS = [
    "BR Team Name",
    "DST Firm Name",
    "Data Driven Segment",
    "SF Territory",
    "Office State",
    "Office City",
    "Office Address",
    "SFDC Email",
    "DST Phone",
    "SFDC Phone",
    "Team Rank",
    "NB AUM 6'25",
    "2024 CRM Contacts",
    "2025 CRM Contacts",
    "Last Interaction Date",
    "Last BT Interaction Date",
    "SFDC Notes",
]

# --- Required filter columns ---
REQUIRED_FILTER_COLS = ["Entity ID", "BR Team Name", "Office City"]

# --- Date columns to fix (Excel serials -> real dates) ---
DATE_COLUMNS = ["Last BT Interaction Date", "Last Interaction Date"]

# Canonical names we'll reference
NB_AUM_CANON = "NB AUM 6'25"
TEAM_RANK_CANON = "Team Rank"
LAST_INTER_CANON = "Last Interaction Date"
LAST_BT_INTER_CANON = "Last BT Interaction Date"

# ---------------------------------------------
# Header normalization + smart aliasing
# ---------------------------------------------
def normalize_header_token(s: str) -> str:
    """Normalize header: strip, unify quotes, collapse whitespace, lower."""
    if not isinstance(s, str):
        return ""
    # Replace non-breaking and odd spaces
    s = s.replace("\xa0", " ").replace("\u2002", " ").replace("\u2003", " ")
    s = s.strip()
    # Normalize curly quotes
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Collapse whitespace runs
    s = re.sub(r"\s+", " ", s)
    return s.lower()

# Direct aliases (after normalization). Add more if you encounter new ones.
HEADER_ALIASES: Dict[str, str] = {
    # Required
    "entity id": "Entity ID",
    "br team name": "BR Team Name",
    "office city": "Office City",
    # Common outputs
    "sf territory": "SF Territory",
    "dst firm name": "DST Firm Name",
    "contact name": "Contact Name",
    "office address": "Office Address",
    "office state": "Office State",
    "data driven segment": "Data Driven Segment",
    "team rank": TEAM_RANK_CANON,
    "sfdc email": "SFDC Email",
    "dst phone": "DST Phone",
    "sfdc phone": "SFDC Phone",
    "last bt interaction date": LAST_BT_INTER_CANON,
    "last interaction date": LAST_INTER_CANON,
    "last interaction": LAST_INTER_CANON,
    "last interaction dt": LAST_INTER_CANON,
    "last activity date": LAST_INTER_CANON,
    "last contact date": LAST_INTER_CANON,
    "last touch": LAST_INTER_CANON,
    "last touch date": LAST_INTER_CANON,

    # typo variants for safety
    "last interation date": LAST_INTER_CANON,
    "last interation": LAST_INTER_CANON,

    "nb aum 6'25": NB_AUM_CANON,
    "nb aum 6’25": NB_AUM_CANON,  # curly apostrophe
    "nb aum 6/25": NB_AUM_CANON,
    "nb aum jun-25": NB_AUM_CANON,
    "nb aum june 2025": NB_AUM_CANON,
    "2025 crm contacts": "2025 CRM Contacts",
    "2024 crm contacts": "2024 CRM Contacts",
    "sfdc notes": "SFDC Notes",

    # Additional helpful aliases
    "team name": "BR Team Name",
    "br team": "BR Team Name",
    "city": "Office City",
    "office location city": "Office City",
    "office – city": "Office City",
    "office - city": "Office City",
}

def rule_based_alias(norm: str) -> str | None:
    """
    Fuzzy variants:
      - Team Rank: any header containing both 'team' and 'rank'
      - NB AUM 6'25: 'nb aum' plus 6/25-ish or June 2025 hints
      - Last Interaction Date: 'last' + ('interaction'|'interation'|'activity'|'touch'|'contact'), and not 'bt'
    """
    # Team rank
    if "team" in norm and "rank" in norm:
        return TEAM_RANK_CANON

    # NB AUM 6'25 variants
    if norm.startswith("nb aum"):
        slim = re.sub(r"[^a-z0-9 ]", " ", norm)
        slim = re.sub(r"\s+", " ", slim).strip()
        patterns = [
            r"\b6 ?[/-]? ?25\b",          # 6/25, 6-25, 6 25
            r"\bjun(?:e)? ?25\b",         # jun 25, june 25
            r"\b2025\b.*\bjun(?:e)?\b",   # 2025 ... june
            r"\bjun(?:e)?\b.*\b2025\b",   # june ... 2025
        ]
        if any(re.search(p, slim) for p in patterns):
            return NB_AUM_CANON

    # Last Interaction Date variants (non-BT)
    if "last" in norm and not (" bt " in f" {norm} " or "bt " in norm or " bt" in norm):
        if any(tok in norm for tok in ["interaction", "interation", "activity", "touch", "contact"]):
            return LAST_INTER_CANON

    return None

def apply_header_normalization(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Rename columns to canonical names based on normalization, dictionary aliases,
    or regex rules. Returns (renamed_df, rename_map).
    """
    rename_map: Dict[str, str] = {}
    canonical_set = set(REQUIRED_FILTER_COLS + OUTPUT_COLUMNS + [LAST_BT_INTER_CANON, LAST_INTER_CANON])
    canonical_norms = {normalize_header_token(c): c for c in canonical_set}

    for c in df.columns:
        norm = normalize_header_token(c)

        # 1) direct alias
        if norm in HEADER_ALIASES:
            rename_map[c] = HEADER_ALIASES[norm]
            continue
        # 2) exact canonical (ignores case/space)
        if norm in canonical_norms:
            rename_map[c] = canonical_norms[norm]
            continue
        # 3) rule-based
        via_rule = rule_based_alias(norm)
        if via_rule:
            rename_map[c] = via_rule
            continue
        # else: leave as-is

    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map

# ---------------------------------------------
# Other helpers
# ---------------------------------------------
def standardize_team_name(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces—for partial team-name matching."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())

def coalesce_column(df: pd.DataFrame, target: str, variants: list) -> pd.DataFrame:
    """Create/overwrite df[target] by pulling from the first non-empty among variant columns."""
    sources = [c for c in variants if c in df.columns]
    if not sources:
        if target not in df.columns:
            df[target] = ""
        return df

    if target not in df.columns:
        df[target] = ""

    def pick_first(*vals):
        for v in vals:
            if pd.notna(v) and str(v).strip() != "":
                return v
        return ""

    df[target] = pd.DataFrame({i: df[s] for i, s in enumerate(sources)}).apply(
        lambda row: pick_first(*row.values), axis=1
    )
    return df

def fix_date_columns_inplace(df: pd.DataFrame, date_cols: List[str]) -> None:
    """Convert Excel serials and parse strings to YYYY-MM-DD for the given date columns."""
    for col in date_cols:
        if col not in df.columns:
            continue
        s = df[col]

        # Excel serials
        numeric = pd.to_numeric(s, errors="coerce")
        num_mask = numeric.notna()
        if num_mask.any():
            dt_num = pd.to_datetime(numeric[num_mask], unit="D", origin="1899-12-30", errors="coerce")
            df.loc[num_mask, col] = dt_num.dt.strftime("%Y-%m-%d")

        # Parse remaining non-empty strings
        str_mask = s.astype(str).str.strip().ne("")
        if str_mask.any():
            dt_parsed = pd.to_datetime(df.loc[str_mask, col], errors="coerce")
            ok = dt_parsed.notna()
            df.loc[str_mask & ok, col] = dt_parsed[ok].dt.strftime("%Y-%m-%d")

# ---------------------------------------------
# Caching small things only
# ---------------------------------------------
@st.cache_data(ttl=600, show_spinner=False, max_entries=3)
def list_files(glob_pattern: str) -> List[str]:
    return glob.glob(glob_pattern)

# ---------------------------------------------
# Core filtering
# ---------------------------------------------
def filter_results(df: pd.DataFrame, mode: str, key_text: str, city_text: str) -> pd.DataFrame:
    """Apply filters + date normalization + coalescing, then return results in the specified order."""
    df = df.copy()

    # Ensure required columns exist
    for req in REQUIRED_FILTER_COLS:
        if req not in df.columns:
            df[req] = ""

    # City normalization (case-insensitive exact)
    df["_city"] = df["Office City"].astype(str).str.strip().str.lower()
    city_norm = (city_text or "").strip().lower()
    city_has_value = city_norm != ""

    # Key match: Entity ID (exact) or Team Name (partial)
    key_norm = (key_text or "").strip().lower()
    if mode == "Entity ID":
        df["_match"] = df["Entity ID"].astype(str).str.strip().str.lower() == key_norm
    else:
        df["_team_norm"] = df["BR Team Name"].astype(str).apply(standardize_team_name)
        patt = standardize_team_name(key_text)
        df["_match​
