# pip install pandas openpyxl
import pandas as pd
from pathlib import Path
import re

# Lokasi file Excel
excel_path = Path("Study Case DA.xlsx")
if not excel_path.exists():
    excel_path = Path("/mnt/data/Study Case DA.xlsx")  # fallback kalau jalan di lingkungan lain

# Folder output
out_dir = Path("csv_out")
out_dir.mkdir(parents=True, exist_ok=True)

# Baca semua sheet & export ke CSV
xls = pd.ExcelFile(excel_path)
for sheet in xls.sheet_names:
    # dtype=object agar tidak ada konversi tipe yang agresif
    df = pd.read_excel(excel_path, sheet_name=sheet, dtype=object, engine="openpyxl")

    # Nama file yang aman untuk OS (spasi & simbol -> underscore)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", sheet).strip("_")
    out_path = out_dir / f"{safe_name}.csv"

    # Simpan CSV; utf-8-sig supaya mudah dibuka di Excel
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {sheet} -> {out_path}")

print("Done.")
