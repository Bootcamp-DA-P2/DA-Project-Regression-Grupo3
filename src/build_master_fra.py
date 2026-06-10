from pathlib import Path
import pandas as pd

# ==========================================
# RUTAS
# ==========================================

RAW_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/processed/master_fra.csv")

# ==========================================
# LISTA DONDE GUARDAR TODOS LOS DATAFRAMES
# ==========================================

all_dfs = []

# ==========================================
# RECORRER TODOS LOS CSV
# ==========================================

for csv_file in RAW_PATH.rglob("*.csv"):

    print(f"Leyendo: {csv_file}")

    try:

        df = pd.read_csv(csv_file)

        # ------------------------------
        # AÑO
        # ------------------------------

        year = csv_file.parts[2]
        df["year"] = year

        # ------------------------------
        # CATEGORÍA
        # ------------------------------

        category = csv_file.stem

        category = (
            category
            .replace("_2012", "")
            .replace("_2019", "")
            .replace("LGBT_Survey_", "")
        )

        df["category"] = category

        # ------------------------------
        # LIMPIEZA
        # ------------------------------

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # ------------------------------
        # NORMALIZAR COLUMNAS
        # ------------------------------

        if "target_group" not in df.columns:
            df["target_group"] = "Unknown"

        if "notes" not in df.columns:
            df["notes"] = None

        # ------------------------------
        # GUARDAR
        # ------------------------------

        all_dfs.append(df)

    except Exception as e:

        print(f"ERROR en {csv_file}")
        print(e)

# ==========================================
# UNIR TODO
# ==========================================

master_df = pd.concat(all_dfs, ignore_index=True)

# ==========================================
# REORDENAR COLUMNAS
# ==========================================

desired_order = [
    "year",
    "CountryCode",
    "target_group",
    "subset",
    "category",
    "question_code",
    "question_label",
    "answer",
    "percentage",
    "notes",
]

existing_columns = [
    col for col in desired_order
    if col in master_df.columns
]

master_df = master_df[existing_columns]

# ==========================================
# GUARDAR CSV
# ==========================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

master_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\n====================================")
print("MASTER FRA GENERADO")
print("====================================")
print(f"Filas: {len(master_df):,}")
print(f"Columnas: {len(master_df.columns)}")
print(f"Archivo: {OUTPUT_PATH}")