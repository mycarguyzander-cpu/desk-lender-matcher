import json
import csv
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_TERM_MONTHS = 72
DEFAULT_MAX_DTI = 0.18
DEFAULT_TAX_FEES_RATE = 0.085
DEFAULT_DOC_FEE = 0.0

# -----------------------------
# CORE MATH
# -----------------------------
def monthly_payment(principal: float, apr_percent: float, term_months: int) -> float:
    if principal <= 0:
        return 0.0
    r = (apr_percent / 100.0) / 12.0
    n = term_months
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def pick_apr_for_score(lender: dict, score: int) -> float:
    tiers = sorted(lender.get("apr_by_score", []), key=lambda x: x.get("min_score", 0), reverse=True)
    for t in tiers:
        if score >= int(t.get("min_score", 0)):
            return float(t.get("apr", 0.0))
    return float(tiers[-1]["apr"]) if tiers else 0.0

def calc_amount_financed(vehicle_price: float, down: float, trade: float,
                         tax_fees_rate: float, doc_fee: float) -> float:
    addons = (vehicle_price * tax_fees_rate) + doc_fee
    return max(0.0, vehicle_price + addons - down - trade)

# -----------------------------
# CSV HELPERS (your inventory format)
# -----------------------------
def _norm_header(h: str) -> str:
    return " ".join(str(h).replace("\n", " ").split()).strip().lower()

def _to_float_money(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "nan":
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def _parse_vehicle_field(vehicle_str: str):
    if not vehicle_str:
        return None, "", ""
    parts = str(vehicle_str).strip().split()
    if len(parts) < 2:
        return None, "", str(vehicle_str).strip()

    year = None
    try:
        year = int(parts[0])
    except Exception:
        year = None

    make = parts[1] if len(parts) >= 2 else ""
    model = " ".join(parts[2:]) if len(parts) >= 3 else ""
    return year, make, model

def load_inventory_csv_bytes(file_bytes: bytes):
    text = file_bytes.decode("utf-8-sig", errors="replace").splitlines()
    reader = csv.DictReader(text)
    if not reader.fieldnames:
        raise ValueError("CSV has no headers.")

    hdr_map = {_norm_header(h): h for h in reader.fieldnames}

    req = ["stock #", "vin", "vehicle", "price"]
    missing = [r for r in req if r not in hdr_map]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            " | Required: Stock #, VIN, Vehicle, Price"
        )

    value_candidates = [
        "black book retail clean",
        "j.d. power retail",
        "kbb retail",
        "kbb lending",
        "j.d. power trade in",
        "mmr wholesale",
    ]

    def get(row, key_norm, default=""):
        h = hdr_map.get(key_norm)
        return row.get(h, default) if h else default

    vehicles = []
    for row in reader:
        stock = str(get(row, "stock #")).strip()
        vin = str(get(row, "vin")).strip()
        vehicle_str = str(get(row, "vehicle")).strip()
        price = _to_float_money(get(row, "price"))

        if not stock or price is None or price <= 0:
            continue

        year, make, model = _parse_vehicle_field(vehicle_str)

        value = None
        for c in value_candidates:
            if c in hdr_map:
                value = _to_float_money(get(row, c))
                if value is not None and value > 0:
                    break

        if value is None or value <= 0:
            value = float(price)

        vehicles.append({
            "stock": stock,
            "vin": vin,
            "year": year if year else "",
            "make": make,
            "model": model,
            "price": float(price),
            "value": float(value),
        })

    return vehicles

def load_lenders_json_bytes(file_bytes: bytes):
    data = json.loads(file_bytes.decode("utf-8", errors="replace"))
    if isinstance(data, dict) and "lenders" in data:
        data = data["lenders"]
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of lenders or {\"lenders\": [...]} format.")
    return data

# -----------------------------
# MATCH ENGINE
# -----------------------------
def match(score: int, down: float, trade: float, income_monthly: float,
          term_months: int, store_max_dti: float,
          lenders: list, inventory: list,
          tax_fees_rate: float, doc_fee: float):

    results = []
    for lender in lenders:
        if not (int(lender.get("min_score", 0)) <= score <= int(lender.get("max_score", 900))):
            continue

        apr = pick_apr_for_score(lender, score)
        max_ltv = float(lender.get("max_ltv", 1.0))
        max_payment_dti = float(lender.get("max_payment_dti", store_max_dti))
        max_amount_financed = float(lender.get("max_amount_financed", 10**12))
        min_amount_financed = float(lender.get("min_amount_financed", 0.0))

        lender_hits = []
        for v in inventory:
            price = float(v["price"])
            value = float(v["value"])

            amt_fin = calc_amount_financed(price, down, trade, tax_fees_rate, doc_fee)
            if amt_fin <= 0:
                continue
            if amt_fin > max_amount_financed:
                continue
            if amt_fin < min_amount_financed:
                continue

            ltv = amt_fin / value if value > 0 else 999
            if ltv > max_ltv:
                continue

            pmt = monthly_payment(amt_fin, apr, term_months)
            pmt_dti = pmt / income_monthly if income_monthly > 0 else 999

            if pmt_dti > max_payment_dti:
                continue
            if pmt_dti > store_max_dti:
                continue

            lender_hits.append({
                "stock": v.get("stock", ""),
                "vin": v.get("vin", ""),
                "year": v.get("year", ""),
                "make": v.get("make", ""),
                "model": v.get("model", ""),
                "price": price,
                "value": value,
                "amount_financed": amt_fin,
                "apr": apr,
                "term": term_months,
                "payment": pmt,
                "ltv": ltv,
                "payment_dti": pmt_dti,
            })

        if lender_hits:
            lender_hits.sort(key=lambda x: (x["payment_dti"], x["payment"]))
            results.append({"lender": lender.get("name", "Unknown"), "apr": apr, "matches": lender_hits})

    results.sort(key=lambda r: (-len(r["matches"]), r["apr"]))
    return results

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Desk & Lender Matcher", layout="wide")
st.title("Desk & Lender Matcher")

with st.sidebar:
    st.header("Customer Inputs")
    score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
    down = st.number_input("Down Payment ($)", min_value=0.0, value=2000.0, step=500.0)
    trade = st.number_input("Trade Amount ($)", min_value=0.0, value=0.0, step=500.0)
    income_m = st.number_input("Income (Monthly $)", min_value=1.0, value=5500.0, step=250.0)

    st.header("Deal Settings")
    term = st.number_input("Term (Months)", min_value=12, max_value=96, value=DEFAULT_TERM_MONTHS, step=1)
    store_max_dti = st.number_input("Store Max DTI", min_value=0.01, max_value=1.00, value=DEFAULT_MAX_DTI, step=0.01, format="%.2f")
    taxfees = st.number_input("Tax/Fees Rate", min_value=0.0, max_value=0.30, value=DEFAULT_TAX_FEES_RATE, step=0.005, format="%.3f")
    docfee = st.number_input("Doc Fee ($)", min_value=0.0, value=DEFAULT_DOC_FEE, step=50.0)

    st.header("Uploads")
    inv_file = st.file_uploader("Inventory CSV", type=["csv"])
    lenders_file = st.file_uploader("Lenders JSON", type=["json"])

col1, col2 = st.columns([1, 2])

if not inv_file or not lenders_file:
    st.info("Upload Inventory CSV and Lenders JSON to run matching.")
    st.stop()

try:
    inventory = load_inventory_csv_bytes(inv_file.getvalue())
    lenders = load_lenders_json_bytes(lenders_file.getvalue())
except Exception as e:
    st.error(str(e))
    st.stop()

results = match(
    score=int(score),
    down=float(down),
    trade=float(trade),
    income_monthly=float(income_m),
    term_months=int(term),
    store_max_dti=float(store_max_dti),
    lenders=lenders,
    inventory=inventory,
    tax_fees_rate=float(taxfees),
    doc_fee=float(docfee),
)

with col1:
    st.subheader("Eligible Lenders")
    if not results:
        st.warning("No matches found.")
        st.stop()

    lender_names = [f'{r["lender"]} (APR~{r["apr"]:.2f}%) | {len(r["matches"])} vehicles' for r in results]
    selected_idx = st.radio("Select lender", list(range(len(lender_names))), format_func=lambda i: lender_names[i])

with col2:
    st.subheader("Matching Vehicles")
    sel = results[int(selected_idx)]
    rows = sel["matches"]

    # show top 200 for performance
    rows = rows[:200]

    st.dataframe(rows, use_container_width=True)

    # export
    st.download_button(
        "Download Results (selected lender) as CSV",
        data=("\n".join([
            "stock,vin,year,make,model,price,value,amount_financed,apr,term,payment,ltv,payment_dti"
        ] + [
            f'{m["stock"]},{m["vin"]},{m["year"]},{m["make"]},"{m["model"]}",{m["price"]:.2f},{m["value"]:.2f},{m["amount_financed"]:.2f},{m["apr"]:.2f},{m["term"]},{m["payment"]:.2f},{m["ltv"]:.3f},{m["payment_dti"]:.3f}'
            for m in rows
        ])).encode("utf-8"),
        file_name="matches.csv",
        mime="text/csv"
    )
