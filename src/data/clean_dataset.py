import os

# Ruta de la carpeta
carpeta_train = "data/processed/train/fake"
carpeta_val = "data/processed/validation/fake"
carpeta_test = "data/processed/test/fake"

# Recorrer los archivos de la carpeta
for archivo in os.listdir(carpeta_train):
    if "easy" not in archivo:
        ruta_archivo = os.path.join(carpeta_train, archivo)
        os.remove(ruta_archivo)
        print("Archivo eliminado:", archivo)
# Recorrer los archivos de la carpeta
for archivo in os.listdir(carpeta_val):
    if "easy" not in archivo:
        ruta_archivo = os.path.join(carpeta_val, archivo)
        os.remove(ruta_archivo)
        print("Archivo eliminado:", archivo)
        # Recorrer los archivos de la carpeta
for archivo in os.listdir(carpeta_test):
    if "easy" not in archivo:
        ruta_archivo = os.path.join(carpeta_test, archivo)
        os.remove(ruta_archivo)
        print("Archivo eliminado:", archivo)

print("Eliminación de archivos completada.")
