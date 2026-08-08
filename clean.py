import os
import re
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

SALTS = [
    "paracetamol", "ibuprofen", "amoxicillin", "metformin", "atorvastatin",
    "omeprazole", "cetirizine", "azithromycin", "amlodipine", "metoprolol",
    "pantoprazole", "ranitidine", "domperidone", "doxycycline", "ciprofloxacin",
    "aspirin", "losartan", "telmisartan", "glimepiride", "montelukast",
    "levothyroxine", "clopidogrel", "rosuvastatin", "esomeprazole", "rabeprazole",
    "ondansetron", "levocetirizine", "fexofenadine", "diclofenac", "tramadol",
    "sertraline", "fluoxetine", "alprazolam", "clonazepam", "gabapentin",
    "pregabalin", "methylcobalamin", "vitamin d3", "calcium carbonate", "zinc sulphate"
]

DOSAGES_RAW = ["250mg", "500mg", "1000mg", "10mg", "20mg", "40mg", "5mg", "400mg", "200mg", "100mg",
               "250 mg", "500 mg", "1000 mg", "10 mg", "20 mg", "40 mg", "5 mg", "400 mg", "200 mg"]

MESSY_PATTERNS = [
    lambda s, d: f"{s.title()} ({d})",
    lambda s, d: f"{s.upper()}-{d.replace(' ', '')}",
    lambda s, d: f"{s} {d}",
    lambda s, d: f"{s.capitalize()} {d.replace('mg', ' mg')}",
    lambda s, d: f"{s}({d})",
]

MANUFACTURERS = [
    "Sun Pharma", "Cipla", "Dr Reddy", "Lupin", "Zydus Cadila",
    "Mankind Pharma", "Alkem Labs", "Abbott India", "Torrent Pharma",
    "Intas Pharma", "Glenmark", "Hetero Drugs", "Aurobindo Pharma",
    "Wockhardt", "Pfizer India", "GSK India", "Sanofi India",
    "Novartis India", "Himalaya", "Dabur Pharma"
]

PACKAGINGS = ["10 tablets", "15 tablets", "30 tablets", "1 bottle", "10 capsules", "6 tablets", "5 ml"]

MEDICINE_PREFIXES = ["Pan", "Ome", "Cet", "Met", "Amox", "Dolo", "Bru", "Aug", "Azee", "Clav"]
MEDICINE_SUFFIXES = ["cin", "tab", "cap", "sol", "plus", "forte", "xr", "sr", "ds", "od"]


def gen_medicine_name(salt):
    prefix = random.choice(MEDICINE_PREFIXES)
    suffix = random.choice(MEDICINE_SUFFIXES)
    return f"{prefix}{suffix} {random.choice(['250', '500', '1000', '10', '20', '40'])}"


def gen_synthetic_dataset(n=5200):
    rows = []
    for _ in range(n):
        salt = random.choice(SALTS)
        dosage = random.choice(DOSAGES_RAW)
        pattern_fn = random.choice(MESSY_PATTERNS)
        composition = pattern_fn(salt, dosage)
        manufacturer = random.choice(MANUFACTURERS)
        packaging = random.choice(PACKAGINGS)
        m_name = gen_medicine_name(salt)
        base_price = round(random.uniform(8, 600), 2)
        if random.random() < 0.06:
            price = 0
        elif random.random() < 0.04:
            price = np.nan
        else:
            price = base_price
        rows.append({
            "medicine_name": m_name,
            "manufacturer": manufacturer,
            "composition": composition,
            "packaging": packaging,
            "price(Rs.)": price
        })
    return pd.DataFrame(rows)


def clean_comp(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'(\d+)\s*(mg|ml|mcg|iu|g)\b', r'\1\2', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data():
    kaggle_path = os.path.join("data", "A_Z_medicines_dataset_of_India.csv")
    fallback_path = os.path.join("data", "medicines_raw.csv")

    if os.path.exists(kaggle_path):
        df = pd.read_csv(kaggle_path)
        df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
        needed = ["medicine_name", "manufacturer", "composition", "price(rs.)"]
        for col in needed:
            if col not in df.columns:
                possible = [c for c in df.columns if any(k in c for k in ["name", "manuf", "comp", "price"])]
                df = df.rename(columns=dict(zip(possible, needed[:len(possible)])))
        df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
    elif os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
    else:
        df = gen_synthetic_dataset(5200)
        df.to_csv(fallback_path, index=False)

    return df


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    df = load_data()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    price_col = [c for c in df.columns if "price" in c]
    if price_col:
        df = df.rename(columns={price_col[0]: "price"})
    else:
        df["price"] = np.nan

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    df["cleaned_composition"] = df["composition"].apply(clean_comp)

    df = df.dropna(subset=["medicine_name", "manufacturer", "cleaned_composition"])
    df = df[df["cleaned_composition"].str.len() > 1]

    df = df.reset_index(drop=True)

    out_path = os.path.join("data", "cleaned_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df[["medicine_name", "manufacturer", "cleaned_composition", "price"]].head())


main()
