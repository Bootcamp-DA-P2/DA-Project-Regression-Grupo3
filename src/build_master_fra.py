from pathlib import Path

#-----RECORRER CARPETAS DE AÑOS-------------
#Esto nos sirve para poder reproducir y añadir más carpetas con años de cara a futuro

#Busca la carpeta de datos
RAW_PATH = Path("data/raw")

print("Detected survey years:\n")

#Busca por años y recorre las carpetas de años
for year_folder in RAW_PATH.iterdir():

    if year_folder.is_dir():

        print(f"Year: {year_folder.name}")

        for csv_file in year_folder.rglob("*.csv"):
            print(f"   - {csv_file.name}")

        print()