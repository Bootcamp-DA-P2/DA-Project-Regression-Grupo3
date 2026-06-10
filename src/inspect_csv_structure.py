from pathlib import Path
import pandas as pd

#Con este programa exploramos qué columnas tiene cada archivo 

RAW_PATH = Path("data/raw")

for csv_file in RAW_PATH.rglob("*.csv"):

    print("\n" + "=" * 80)
    print(csv_file)

    try:
        df = pd.read_csv(csv_file)

        print("\nColumnas:")
        print(df.columns.tolist())

        print("\nPrimeras filas:")
        print(df.head(3))

    except Exception as e:
        print(f"ERROR: {e}")