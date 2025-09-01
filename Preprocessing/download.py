import cdsapi
import os

# Configuration de la clé API CDS
c = cdsapi.Client()

# Définir la zone d'étude et les paramètres
area = [0, -45, -15, -30]  # [lat_max, lon_min, lat_min, lon_max]
year = 2007

# Variables de niveau de pression selon ERA5
variables_pressure = [

    "u_component_of_wind",
    "v_component_of_wind",

]

levels = ['1000', '850', '700', '500', '300']  # Niveaux sélectionnés


# Répertoire de sortie
output_directory = "data/raw/"
os.makedirs(output_directory, exist_ok=True)

months_to_download = [2]

# Télécharger les données pour chaque mois sélectionné
for month in months_to_download:
    # Gérer les années croisées (ex : janvier-juin de l'année suivante)
    download_year = year if month >= 11 else year + 1
    for variable in variables_pressure:
        output_filename = f"{variable}_{year}{month:02d}.nc"
        output_path = os.path.join(output_directory, output_filename)

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'pressure_level': levels,
                'year': str(year),
                'month': f"{month:02d}",
                'day': [str(i).zfill(2) for i in range(1, 32)],
                'time': ['00:00', '06:00','12:00', '18:00'],
                'area': area,
            },
            output_path
        )
        print(f"✅ Downloaded {variable} for {year}-{month:02d}")

print("🎉 All pressure level data downloaded successfully.")
