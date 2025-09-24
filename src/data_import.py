import zipfile

# Angiv stien til ZIP-filen
zip_path = "../data/raw/diabetes-health-indicators-dataset.zip"

# Udpak ZIP-filen til raw-mappen
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("../data/raw/")

print("ZIP file extracted to ../data/raw/")
