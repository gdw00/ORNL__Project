import netCDF4 as nc
import numpy as np
import pandas as pd


file_path1 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/surfdata_0.9x1.25_simyr1700_c240826.nc'
file_path2 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/20240903_TRENDY_f09_ICB1850CNRDCTCBC_ad_spinup.elm.h0.0021-01-01-00000.nc'
file_path3 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/20240903_TRENDY_f09_ICB1850CNPRDCTCBC.elm.h0.0401-01-01-00000.nc'
file_path4 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/forcings/crujra.v2.5.5d_FLDS_1901-2023_z01.nc'
file_path5 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/forcings/crujra.v2.5.5d_PSRF_1901-2023_z01.nc'
file_path6 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/forcings/crujra.v2.5.5d_FSDS_1901-2023_z01.nc'
file_path7 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/forcings/crujra.v2.5.5d_QBOT_1901-2023_z01.nc'
file_path8 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/forcings/crujra.v2.5.5d_PRECTmms_1901-2023_z01.nc'
file_path9 = '/mnt/DATA/0_oak_data/0_global_data/ornl_data/forcings/crujra.v2.5.5d_TBOT_1901-2023_z01.nc'


lat1, lon1 = 90, 0  # Top-left corner
lat2, lon2 = -90, 360  # Bottom-right corner

ds1 = nc.Dataset(file_path1)
ds2 = nc.Dataset(file_path2)
ds3 = nc.Dataset(file_path3)
ds4 = nc.Dataset(file_path4)
ds5 = nc.Dataset(file_path5)
ds6 = nc.Dataset(file_path6)
ds7 = nc.Dataset(file_path7)
ds8 = nc.Dataset(file_path8)
ds9 = nc.Dataset(file_path9)

lats = ds2.variables['lat'][:]
lons = ds2.variables['lon'][:]
landmask = ds2.variables['landmask'][:]


lat_indices = np.where((lats >= lat2) & (lats <= lat1))[0]
lon_indices = np.where((lons >= lon1) & (lons <= lon2))[0]

filtered_coordinates = [(i, j) for i in lat_indices for j in lon_indices if landmask[i, j] == 1]
print("Number of filtered coordinates with landmask=1:", len(filtered_coordinates))




batch_size = 1000

save_path = "training_data_landfrac_batched.pkl"

batch_number = 1

for start_idx in range(0, len(filtered_coordinates), batch_size):
    end_idx = min(start_idx + batch_size, len(filtered_coordinates))
    batch_coords = filtered_coordinates[start_idx:end_idx]
    print(f"Processing batch {start_idx} to {end_idx}")


    data_dict = {
        'landfrac':[],   
        'Latitude': [],  
        'Longitude': [],  
        'FLDS': [], 'PSRF': [], 'FSDS': [], 'QBOT': [], 'PRECTmms': [], 'TBOT': [],
        'LANDFRAC_PFT': [], 'PCT_NATVEG': [], 'AREA': [], 'peatf': [], 'abm': [],
        'SOIL_COLOR': [], 'SOIL_ORDER': [], 'PCT_NAT_PFT': [], 'PCT_SAND': [],
        'SOIL3C': [], 'SOIL4C': [], 'DEADSTEMC': [], 'DEADCROOTC': [], 'CWDC': [],
        'TLAI': [], 'GPP': [], 
        'Y_SOIL3C': [], 'Y_SOIL4C': [], 'Y_DEADSTEMC': [], 'Y_DEADCROOTC': [],
        'Y_CWDC': [],  'Y_TLAI': [],  'Y_GPP': [],
        'SCALARAVG_vr': [],

        'HR': [],  'AR': [],  'NPP': [], 'COL_FIRE_CLOSS': [],
        'Y_HR': [],  'Y_AR': [],  'Y_NPP': [], 'Y_COL_FIRE_CLOSS': []
    }
    for i, j in batch_coords:
        data_dict['landfrac'].append(ds2.variables['landfrac'][i, j])  
        data_dict['Latitude'].append(lats[i])
        data_dict['Longitude'].append(lons[j])
        data_dict['LANDFRAC_PFT'].append(ds1.variables['LANDFRAC_PFT'][i, j])
        data_dict['PCT_NATVEG'].append(ds1.variables['PCT_NATVEG'][i, j])
        data_dict['AREA'].append(ds1.variables['AREA'][i, j])
        data_dict['peatf'].append(ds1.variables['peatf'][i, j])
        data_dict['abm'].append(ds1.variables['abm'][i, j])
        data_dict['SOIL_COLOR'].append(ds1.variables['SOIL_COLOR'][i, j])
        data_dict['SOIL_ORDER'].append(ds1.variables['SOIL_ORDER'][i, j])
        data_dict['PCT_SAND'].append(ds1.variables['PCT_SAND'][:, i, j])
        data_dict['PCT_NAT_PFT'].append(ds1.variables['PCT_NAT_PFT'][:, i, j])

        data_dict['SOIL3C'].append(ds2.variables['SOIL3C'][0, i, j])
        data_dict['SOIL4C'].append(ds2.variables['SOIL4C'][0, i, j])
        data_dict['DEADSTEMC'].append(ds2.variables['DEADSTEMC'][0, i, j])
        data_dict['DEADCROOTC'].append(ds2.variables['DEADCROOTC'][0, i, j])
        data_dict['CWDC'].append(ds2.variables['CWDC'][0, i, j])
        data_dict['TLAI'].append(ds2.variables['TLAI'][0, i, j])
    
        data_dict['GPP'].append(ds2.variables['GPP'][0, i, j])
        data_dict['SCALARAVG_vr'].append(ds2.variables['SCALARAVG_vr'][0, :, i, j])
        data_dict['HR'].append(ds2.variables['HR'][0, i, j])
        data_dict['AR'].append(ds2.variables['AR'][0, i, j])
        data_dict['NPP'].append(ds2.variables['NPP'][0, i, j])
        data_dict['COL_FIRE_CLOSS'].append(ds2.variables['COL_FIRE_CLOSS'][0, i, j])

        data_dict['Y_SOIL3C'].append(ds3.variables['SOIL3C'][0, i, j])
        data_dict['Y_SOIL4C'].append(ds3.variables['SOIL4C'][0, i, j])
        data_dict['Y_DEADSTEMC'].append(ds3.variables['DEADSTEMC'][0, i, j])
        data_dict['Y_DEADCROOTC'].append(ds3.variables['DEADCROOTC'][0, i, j])
        data_dict['Y_CWDC'].append(ds3.variables['CWDC'][0, i, j])
        data_dict['Y_TLAI'].append(ds3.variables['TLAI'][0, i, j])

        data_dict['Y_GPP'].append(ds3.variables['GPP'][0, i, j])
        data_dict['Y_HR'].append(ds2.variables['HR'][0, i, j])
        data_dict['Y_AR'].append(ds2.variables['AR'][0, i, j])
        data_dict['Y_NPP'].append(ds2.variables['NPP'][0, i, j])
        data_dict['Y_COL_FIRE_CLOSS'].append(ds2.variables['COL_FIRE_CLOSS'][0, i, j])
    lats4, lons4 = ds4.variables['LATIXY'][:], ds4.variables['LONGXY'][:]
    for i, j in batch_coords:
        lat, lon = lats[i], lons[j]
        closest_lat_idx = np.abs(lats4 - lat).argmin()
        closest_lon_idx = np.abs(lons4 - lon + 360).argmin()

        flds_values = ds4.variables['FLDS'][closest_lat_idx, :] * 0.03356934 + 500.0
        data_dict['FLDS'].append(flds_values)
        
        flds_values = ds5.variables['PSRF'][closest_lat_idx, :] * 3.356934 + 70000.0
        data_dict['PSRF'].append(flds_values)
        
        flds_values = ds6.variables['FSDS'][closest_lat_idx, :] * 0.06781006 + 990.0
        data_dict['FSDS'].append(flds_values)
        
        flds_values = ds7.variables['QBOT'][closest_lat_idx, :] * 3.356934e-06 + 0.05
        data_dict['QBOT'].append(flds_values)
        
        flds_values = ds8.variables['PRECTmms'][closest_lat_idx, :] * 2.685547e-06
        data_dict['PRECTmms'].append(flds_values)
        
        flds_values = ds9.variables['TBOT'][closest_lat_idx, :] * 0.005874634 + 262.5
        data_dict['TBOT'].append(flds_values)

    df_batch = pd.DataFrame(data_dict)


    batch_save_path = f"/mnt/DATA/0_oak_data/1_global_monthly_newinput_data/training_data_batch_{batch_number:02d}.pkl"
    df_batch.to_pickle(batch_save_path)
    print(f"Saved batch {start_idx} to {end_idx} to {batch_save_path}")

    batch_number += 1

print(f"Data successfully saved.")

# Close the datasets
ds1.close()
ds2.close()
ds3.close()
ds4.close()
ds5.close()
ds6.close()
ds7.close()
ds8.close()
ds9.close()



import pandas as pd
import glob


input_files = sorted(glob.glob('/mnt/DATA/0_oak_data/1_global_monthly_newinput_data/training_data_batch_*.pkl'))

output_dir = '/mnt/DATA/0_oak_data/1_global_monthly_newinput_data/1_global_data/'  

# Define columns for processing
time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
single_value_columns = [
    'LANDFRAC_PFT', 'PCT_NATVEG', 'AREA', 'peatf', 'abm', 'SOIL_COLOR', 'SOIL_ORDER',
    'SOIL3C', 'SOIL4C', 'DEADSTEMC', 'DEADCROOTC', 'CWDC', 'TLAI', 'GPP',
    'Y_SOIL3C', 'Y_SOIL4C', 'Y_DEADSTEMC', 'Y_DEADCROOTC', 'Y_CWDC', 'Y_TLAI', 'Y_GPP',
    'HR',  'AR',  'NPP', 'COL_FIRE_CLOSS',
    'Y_HR',  'Y_AR',  'Y_NPP', 'Y_COL_FIRE_CLOSS'
]
list_like_columns = ['PCT_NAT_PFT', 'PCT_SAND', 'SCALARAVG_vr']


time_series_length = 179580
steps_per_day = 4
days_per_year = 365
years_in_data = 123
months_per_year = 12
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def calculate_monthly_avg(time_series):
    if not isinstance(time_series, (list, np.ndarray)):
        print(f"Invalid type: {type(time_series)}")
        return []
    if len(time_series) != time_series_length:
        print(f"Invalid length: {len(time_series)}, expected {time_series_length}")
        return []
    
    monthly_averages = []
    start_idx = 0
    for year in range(years_in_data):
        for month_days in days_per_month:
            end_idx = start_idx + month_days * steps_per_day
            monthly_avg = np.mean(time_series[start_idx:end_idx])
            monthly_averages.append(monthly_avg)
            start_idx = end_idx
    return monthly_averages





for file_path in input_files:
    print(f"Processing {file_path}...")
    
    
    df = pd.read_pickle(file_path)
    print(df.info())

    for col in time_series_columns:
        if col in df.columns:
            print(f"Processing {col}...")
            df[col] = df[col].apply(calculate_monthly_avg)

            sample_data = df[col].apply(lambda x: x if isinstance(x, list) else [])
            # print(f"Variable: {col}")
            # print(f"Length of each sample: {sample_data.apply(len).unique()}")
            # print("Top 5 Samples:")
            # print("-" * 50)

    for col in single_value_columns:
        df[col] = df[col].astype(str).str.strip()  # Remove any whitespace
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in list_like_columns:
        expanded_cols = df[col].apply(pd.Series).fillna(0)  # Replace NaN values with 0
        expanded_cols = expanded_cols.add_prefix(f"{col}_")
        df = df.drop(col, axis=1).join(expanded_cols)

    y_columns = [col for col in df.columns if col.startswith('Y_')]
    other_columns = [col for col in df.columns if not col.startswith('Y_')]
    df = df[other_columns + y_columns]

    output_file = f"{output_dir}1_{file_path.split('/')[-1]}"
    df.to_pickle(output_file)
    print(f"Processed file saved as {output_file}")
    print(df.info())
    print(df[['Latitude', 'Longitude']].head())

print("Processing complete.")



