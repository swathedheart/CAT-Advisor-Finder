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
st.caption("Search your Rolodex CSVs by **Entity ID or BR Team Name** and **Office City**.")

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

    # 👇 typo variants for safety
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
        col_series = df[col]
        numeric_mask = pd.to_numeric(col_series, errors="coerce").notna()
        if numeric_mask.any():
            numeric_vals = pd.to_numeric(col_series[numeric_mask], errors="coerce")
            dt_numeric = pd.to_datetime(numeric_vals, unit="D", origin="1899-12-30", errors="coerce")
            df.loc[numeric_mask, col] = dt_numeric.dt.strftime("%Y-%m-%d")
        remaining_mask = df[col].astype(str).str.len() > 0
        if remaining_mask.any():
            dt_parsed = pd.to_datetime(df.loc[remaining_mask, col], errors="coerce", infer_datetime_format=True)
            ok = dt_parsed.notna()
            df.loc[remaining_mask[remaining_mask].index[ok], col] = dt_parsed[ok].dt.strftime("%Y-%m-%d")

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

    # Normalize headers first
    df, _ = apply_header_normalization(df)

    # Ensure required columns exist
    for req in REQUIRED_FILTER_COLS:
        if req not in df.columns:
            df[req] = ""

    # City match (case-insensitive exact)
    df["_city"] = df["Office City"].astype(str).str.strip().str.lower()
    city_norm = (city_text or "").strip().lower()

    # Key match: Entity ID (exact) or Team Name (partial)
    key_norm = (key_text or "").strip().lower()
    if mode == "Entity ID":
        df["_match"] = df["Entity ID"].astype(str).str.strip().str.lower() == key_norm
    else:
        df["_team_norm"] = df["BR Team Name"].astype(str).apply(standardize_team_name)
        df["_match"] = df["_team_norm"].str.contains(standardize_team_name(key_text), na=False)

    result = df.loc[df["_match"] & (df["_city"] == city_norm)].copy()

    # Coalesce tricky columns (just in case variants slipped through)
    result = coalesce_column(
        result,
        target=TEAM_RANK_CANON,
        variants=[
            TEAM_RANK_CANON, " Team Rank ", "Team  Rank", "Team Rank ", " Team Rank", "Rank (Team)"
        ]
    )
    result = coalesce_column(
        result,
        target=NB_AUM_CANON,
        variants=[
            NB_AUM_CANON, "NB AUM 6’25", " NB AUM 6'25 ", "NB AUM 6'25 ", " NB AUM 6'25",
            "NB AUM 6/25", "NB AUM Jun-25", "NB AUM June 2025"
        ]
    )
    result = coalesce_column(
        result,
        target=LAST_INTER_CANON,
        variants=[
            LAST_INTER_CANON, "Last Interaction", "Last Interaction Dt",
            "Last Activity Date", "Last Contact Date", "Last Touch", "Last Touch Date",
            # 👇 typo variants
            "Last Interation Date", "Last Interation",
        ]
    )
    # Last BT Interaction Date is covered by aliases/normalization

    # Convert dates after coalescing
    fix_date_columns_inplace(result, DATE_COLUMNS)

    # Ensure all output columns exist (fill missing with empty) and order columns
    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = ""

    return result[OUTPUT_COLUMNS]

# ---------------------------------------------
# Streamed search across files (no usecols; normalize then filter)
# ---------------------------------------------
def search_across_files(
    files: List[str],
    mode: str,
    key_text: str,
    city_text: str,
    chunk_size: int = 150_000,
) -> Tuple[pd.DataFrame, list, Dict[str, int], Dict[str, int]]:
    """
    Read each CSV in chunks using C/Python engine (pyarrow doesn't support chunksize),
    normalize headers per chunk, filter in-chunk, and accumulate only the matches.

    Returns:
      - combined results DataFrame
      - list of files that failed
      - column_nonempty_counts: non-empty counts per canonical output column (profiling)
      - alias_hit_counts: counts of how often each alias/rule mapped to a canonical name
    """
    matches = []
    failed = []
    column_nonempty_counts: Dict[str, int] = {c: 0 for c in OUTPUT_COLUMNS}
    alias_hit_counts: Dict[str, int] = {}

    engines_that_support_chunks = ["c", "python"]

    for f in files:
        read_ok = False
        try:
            for eng in engines_that_support_chunks:
                try:
                    for chunk in pd.read_csv(
                        f,
                        dtype="string",
                        chunksize=chunk_size,
                        engine=eng,
                        encoding="utf-8",
                        low_memory=True,
                    ):
                        renamed_chunk, rename_map = apply_header_normalization(chunk)
                        for orig, canon in (rename_map or {}).items():
                            key = f"{orig} -> {canon}"
                            alias_hit_counts[key] = alias_hit_counts.get(key, 0) + 1

                        filtered = filter_results(renamed_chunk, mode, key_text, city_text)
                        if not filtered.empty:
                            for col in OUTPUT_COLUMNS:
                                if col in filtered.columns:
                                    non_empty = filtered[col].notna().sum() - (filtered[col] == "").sum()
                                    column_nonempty_counts[col] += int(non_empty)
                            matches.append(filtered)
                    read_ok = True
                    break
                except UnicodeDecodeError:
                    for enc in ["utf-8-sig", "latin-1"]:
                        for chunk in pd.read_csv(
                            f,
                            dtype="string",
                            chunksize=chunk_size,
                            engine=eng,
                            encoding=enc,
                            low_memory=True,
                        ):
                            renamed_chunk, rename_map = apply_header_normalization(chunk)
                            for orig, canon in (rename_map or {}).items():
                                key = f"{orig} -> {canon}"
                                alias_hit_counts[key] = alias_hit_counts.get(key, 0) + 1
                            filtered = filter_results(renamed_chunk, mode, key_text, city_text)
                            if not filtered.empty:
                                for col in OUTPUT_COLUMNS:
                                    if col in filtered.columns:
                                        non_empty = filtered[col].notna().sum() - (filtered[col] == "").sum()
                                        column_nonempty_counts[col] += int(non_empty)
                                matches.append(filtered)
                    read_ok = True
                    break
                except ValueError:
                    continue
            if not read_ok:
                failed.append(f"{os.path.basename(f)} — could not read with C/Python engines")
        except Exception as e:
            failed.append(f"{os.path.basename(f)} — {e}")

    out = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return out, failed, column_nonempty_counts, alias_hit_counts

# ---------------------------------------------
# Sidebar UI
# ---------------------------------------------
with st.sidebar:
    st.header("Data Source")
    glob_pattern = st.text_input(
        "Folder pattern (glob)",
        value="./* Rolodex.csv",
        help="Examples:\n"
             "  ./* Rolodex.csv\n"
             "  C:/path/to/folder/* Rolodex.csv\n"
             "  C:/path/to/folder/**/* Rolodex.csv (includes subfolders)"
    )
    st.divider()
    st.header("Search")
    mode = st.radio("Search by", ["Entity ID", "BR Team Name"], horizontal=True)
    key_input = st.text_input(
        "Entity ID or BR Team Name",
        value="",
        placeholder="e.g., 0038b00003Be6YvAAJ or 'Alpha Partners'"
    )
    city_input = st.text_input("Office City", value="", placeholder="e.g., New York")
    run = st.button("Search", type="primary")

# ---------------------------------------------
# Main flow
# ---------------------------------------------
files = list_files(glob_pattern)
st.write(f"**Found files:** {len(files)}")
if not files:
    st.info("No files matched the pattern. Adjust the glob (e.g., `./**/* Rolodex.csv`).")

st.divider()

if run:
    if not key_input or not city_input:
        st.warning("Enter both the **Entity ID/Team Name** and the **Office City**.")
    elif not files:
        st.warning("No data files found. Check your folder pattern.")
    else:
        with st.spinner("Searching across files…"):
            result, failed, col_counts, alias_hits = search_across_files(
                files, mode, key_input, city_input, chunk_size=150_000
            )

        if failed:
            with st.expander("Files that failed to read", expanded=False):
                st.write("\n".join(failed))

        # Profilers
        with st.expander("Column profile (non-empty counts in results)", expanded=False):
            st.json(col_counts)
        if alias_hits:
            with st.expander("Header mappings observed", expanded=False):
                st.json(alias_hits)

        if result.empty:
            st.info("No matching rows found. Try adjusting the name/ID or city.")
        else:
            st.success(f"Found {len(result)} matching row(s).")
            st.dataframe(result, use_container_width=True, hide_index=True)

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            fname = f"advisor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button("Download CSV", csv_bytes, file_name=fname, mime="text/csv")
