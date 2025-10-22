import streamlit as st
import pandas as pd
import glob
import os
import re
from datetime import datetime
from typing import List, Tuple, Dict
import csv # Import csv module for quoting constant

# --- Header / Branding ---
st.set_page_config(page_title="CAT Advisor Finder", page_icon="🐱", layout="wide")
st.title("🐱 CAT Advisor Finder")
st.caption("Search your Rolodex CSVs by Entity ID or BR Team Name. Office City is an optional filter.")

# --- Columns to output (in your requested order) ---
OUTPUT_COLUMNS = [
    "BR Team Name", "DST Firm Name", "Data Driven Segment", "SF Territory", "Office State", "Office City", "Office Address", "Contact Name", "SFDC Email", "DST Phone", "SFDC Phone", "Team Rank", "NB AUM 6'25", "2024 CRM Contacts", "2025 CRM Contacts", "Last Interaction Date", "Last BT Interaction Date", "SFDC Notes",
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
    "nb aum 6’25": NB_AUM_CANON, # curly apostrophe
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
            r"\b6 ?[/-]? ?25\b", # 6/25, 6-25, 6 25
            r"\bjun(?:e)? ?25\b", # jun 25, june 25
            r"\b2025\b.*\bjun(?:e)?\b", # 2025 ... june
            r"\bjun(?:e)?\b.*\b2025\b", # june ... 2025
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
    Rename columns to canonical names based on normalization, dictionary aliases, or regex rules.
    Returns (renamed_df, rename_map).
    """
    rename_map: Dict[str, str] = {}
    canonical_set = set(REQUIRED_FILTER_COLS + OUTPUT_COLUMNS + [LAST_BT_INTER_CANON, LAST_INTER_CANON])
    canonical_norms = {normalize_header_token(c): c for c in canonical_set}

    for c in df.columns:
        norm = normalize_header_token(c)

        # 1) direct alias
        if norm in HEADER_ALIASES:
            # Check for duplicate target *before* adding
            if HEADER_ALIASES[norm] not in rename_map.values():
                 rename_map[c] = HEADER_ALIASES[norm]
            continue

        # 2) exact canonical (ignores case/space)
        if norm in canonical_norms:
            if canonical_norms[norm] not in rename_map.values():
                rename_map[c] = canonical_norms[norm]
            continue
        
        # 3) rule-based
        via_rule = rule_based_alias(norm)
        if via_rule:
            if via_rule not in rename_map.values():
                rename_map[c] = via_rule
            continue
        
        # else: leave as-is

    if rename_map:
        # Check for duplicate targets one last time
        targets = list(rename_map.values())
        if len(targets) != len(set(targets)):
            pass
        else:
            try:
                df = df.rename(columns=rename_map)
            except Exception as e:
                print(f"Error renaming columns: {e}") # Log for debugging
                pass 
    return df, rename_map

# ---------------------------------------------
# Other helpers
# ---------------------------------------------

def standardize_team_name(s: str) -> str:
    """Lowercase, replace punctuation with space, collapse spaces—for partial team-name matching."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # Replaced punctuation with a space, instead of just removing it.
    s = re.sub(r"[^\w\s]", " ", s) 
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
# Caching 
# ---------------------------------------------
@st.cache_data(ttl=600, show_spinner=False, max_entries=3)
def list_files(glob_pattern: str) -> List[str]:
    return glob.glob(glob_pattern)

# <--- ***** CACHED FUNCTION TO LOAD ALL DATA ***** --->
@st.cache_data(ttl=600, show_spinner="Loading and cleaning all Rolodex files for the first time...")
def load_all_data(files: List[str]) -> Tuple[pd.DataFrame, list, Dict[str, int]]:
    """
    Load all files, normalize, coalesce, and combine into one big DataFrame.
    This is cached to make subsequent searches fast.
    """
    all_chunks = []
    failed = []
    alias_hit_counts: Dict[str, int] = {}
        
    for f in files:
        read_ok = False
        last_exception = None
        try:
            # Use the slow, robust parser
            for chunk in pd.read_csv(
                f,
                dtype="string",
                chunksize=150_000, # Still read in chunks to avoid memory spikes
                engine='python',
                sep='\t',
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
                low_memory=True,
                on_bad_lines='skip',
            ):
                renamed_chunk, rename_map = apply_header_normalization(chunk)
                
                for orig, canon in (rename_map or {}).items():
                    key = f"{orig} -> {canon}"
                    alias_hit_counts[key] = alias_hit_counts.get(key, 0) + 1
                
                all_chunks.append(renamed_chunk)
            read_ok = True 

        except UnicodeDecodeError as e:
            last_exception = e
            for enc in ["utf-8-sig", "latin-1"]:
                try:
                    for chunk in pd.read_csv(
                        f,
                        dtype="string",
                        chunksize=150_000,
                        engine='python',
                        sep='\t',
                        quoting=csv.QUOTE_NONE,
                        encoding=enc,
                        low_memory=True,
                        on_bad_lines='skip',
                    ):
                        renamed_chunk, rename_map = apply_header_normalization(chunk)
                        
                        for orig, canon in (rename_map or {}).items():
                            key = f"{orig} -> {canon}"
                            alias_hit_counts[key] = alias_hit_counts.get(key, 0) + 1
                        
                        all_chunks.append(renamed_chunk)
                    read_ok = True
                    break 
                except Exception as e:
                    last_exception = e
                    continue

        except Exception as e:
            last_exception = e
            if f"{os.path.basename(f)} — {e}" not in failed:
                failed.append(f"{os.path.basename(f)} — {e}")

        if not read_ok and last_exception:
             err_msg = f"{os.path.basename(f)} — could not read. Last error: {last_exception}"
             if err_msg not in failed:
                failed.append(err_msg)

    if not all_chunks:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), failed, alias_hit_counts

    # Combine all chunks into one big DataFrame
    df = pd.concat(all_chunks, ignore_index=True)

    # --- Run pre-filtering cleanup ONCE on the giant DataFrame ---
    
    # Ensure required columns exist
    for req in (REQUIRED_FILTER_COLS + OUTPUT_COLUMNS):
        if req not in df.columns:
            df[req] = ""

    # Coalesce all search columns
    df = coalesce_column(df, "Entity ID", ["Entity ID"])
    df = coalesce_column(df, "BR Team Name", ["BR Team Name", "Broker Team Name", "Team Name", "BR Team"])
    df = coalesce_column(df, "Office City", ["Office City", "City"])

    # Coalesce tricky output columns
    df = coalesce_column(df, TEAM_RANK_CANON, [TEAM_RANK_CANON, " Team Rank ", "Team Rank", "Team Rank ", " Team Rank", "Rank (Team)"])
    df = coalesce_column(df, NB_AUM_CANON, [NB_AUM_CANON, "NB AUM 6’25", " NB AUM 6'25 ", "NB AUM 6'25 ", " NB AUM 6'25", "NB AUM 6/25", "NB AUM Jun-25", "NB AUM June 2025"])
    df = coalesce_column(df, LAST_INTER_CANON, [LAST_INTER_CANON, "Last Interaction", "Last Interaction Dt", "Last Activity Date", "Last Contact Date", "Last Touch", "Last Touch Date", "Last Interation Date", "Last Interation"])
    df = coalesce_column(df, "Contact Name", ["Contact Name"]) # Also clean Contact Name

    # Convert dates
    fix_date_columns_inplace(df, DATE_COLUMNS)

    # <--- ***** CHANGED: Convert AUM to a number ***** --->
    df[NB_AUM_CANON] = pd.to_numeric(df[NB_AUM_CANON], errors='coerce')
    # <--- ***** END OF CHANGE ***** --->

    # Ensure all output columns exist one last time
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df, failed, alias_hit_counts


# ---------------------------------------------
# Core filtering
# ---------------------------------------------
def filter_results(
    df: pd.DataFrame, 
    mode: str, 
    key_text: str, 
    city_text: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply filters to the *already cleaned* master DataFrame."""
    
    df_filtered = df.copy()
    column_nonempty_counts: Dict[str, int] = {c: 0 for c in OUTPUT_COLUMNS}

    # --- City filter ---
    city_norm = (city_text or "").strip().lower()
    if city_norm:
        city_mask = (df_filtered["Office City"].astype(str).str.strip().str.lower() == city_norm)
    else:
        city_mask = True # Don't filter by city if it's empty

    # --- Key match ---
    key_norm = (key_text or "").strip().lower()
    if mode == "Entity ID":
        match_mask = df_filtered["Entity ID"].astype(str).str.strip().str.lower() == key_norm
    else:
        # Standardize team name on the fly for matching
        team_norm = df_filtered["BR Team Name"].astype(str).apply(standardize_team_name)
        match_mask = team_norm.str.contains(standardize_team_name(key_text), na=False)

    result = df_filtered.loc[match_mask & city_mask].copy()
    
    # --- Profiling ---
    if not result.empty:
        for col in OUTPUT_COLUMNS:
            if col in result.columns:
                non_empty = result[col].notna().sum() - (result[col] == "").sum()
                column_nonempty_counts[col] += int(non_empty)

    # Return only the specified output columns
    return result[OUTPUT_COLUMNS], column_nonempty_counts


# ---------------------------------------------
# Sidebar UI
# ---------------------------------------------
with st.sidebar:
    st.header("Data Source")
    glob_pattern = st.text_input(
        "Folder pattern (glob)",
        value="./* Rolodex.csv",
        help="Examples:\n"
             " ./* Rolodex.csv\n"
             " C:/path/to/folder/* Rolodex.csv\n"
             " C:/path/to/folder/**/* Rolodex.csv (includes subfolders)"
    )
    
    st.divider()
    
    st.header("Search")
    mode = st.radio("Search by", ["Entity ID", "BR Team Name"], horizontal=True)
    key_input = st.text_input(
        "Entity ID or BR Team Name",
        value="",
        placeholder="e.g., 0038b00003Be6YvAAJ or 'Alpha Partners'"
    )
    city_input = st.text_input("Office City (Optional)", value="", placeholder="e.g., New York")
    
    run = st.button("Search", type="primary")

# ---------------------------------------------
# Main flow
# ---------------------------------------------
files = list_files(glob_pattern)
st.write(f"Found files: {len(files)}")

if not files:
    st.info("No files matched the pattern. Adjust the glob (e.g., `./**/* Rolodex.csv`).")

st.divider()

if run:
    if not key_input:
        st.warning("Enter an **Entity ID/Team Name** to search.")
    elif not files:
        st.warning("No data files found. Check your folder pattern.")
    else:
        # 1. This will be SLOW the FIRST time, and fast every time after.
        master_df, failed, alias_hits = load_all_data(files)
        
        if failed:
            with st.expander("Files that failed to read or had bad lines", expanded=False):
                st.write("\n".join(failed))
                st.caption("Note: `on_bad_lines='skip'` was used. Some rows in these (or other) files may have been skipped if they were malformed.")

        if master_df.empty and not failed:
             st.warning("Loaded files, but no data was found.")
        
        # Only proceed if we have a master DataFrame
        if not master_df.empty:
            # 2. This is the FAST part. It just filters the in-memory DataFrame.
            with st.spinner("Searching..."):
                result, col_counts = filter_results(
                    master_df, mode, key_input, city_input
                )

            # Profilers
            with st.expander("Column profile (non-empty counts in results)", expanded=False):
                st.json(col_counts)
            
            if alias_hits:
                with st.expander("Header mappings observed (from initial load)", expanded=False):
                    st.json(alias_hits)
            
            # 3. Display results
            if result.empty:
                st.info("No matching rows found. Try adjusting the name/ID or city.")
            else:
                st.success(f"Found {len(result)} matching row(s).")
                
                # <--- ***** CHANGED: Added column_config for formatting ***** --->
                st.dataframe(
                    result, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        NB_AUM_CANON: st.column_config.NumberColumn(
                            label="NB AUM 6'25",
                            format="$%,.0f"  # Format as $1,234,567
                        )
                    }
                )
                # <--- ***** END OF CHANGE ***** --->
                
                csv_bytes = result.to_csv(index=False).encode("utf-8")
                fname = f"advisor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "Download CSV",
                    csv_bytes,
                    file_name=fname,
                    mime="text/csv"
                )
