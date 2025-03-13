#%%
import pandas as pd

#%%
# Load the data from the uploaded files
usace_bmp_path = 'USACE_BMP_database/USACE_BMP_database.csv'
wam_luid_path = 'USACE_BMP_database/WAM_unique_LUID_optim_TP.csv'

usace_bmp_df = pd.read_csv(usace_bmp_path)
wam_luid_df = pd.read_csv(wam_luid_path)

#%%
# Filter BMPs based on the LUIDs provided in the WAM_unique_LUID_optim_TN file
selected_luids = wam_luid_df['LUID']
filtered_bmps = usace_bmp_df[usace_bmp_df['LU_CODE'].isin(selected_luids)]

#%%
# Save the filtered BMPs to a new CSV file (optional)

filtered_bmps.to_csv('Filtered_BMPs.csv', index=False)

# Display the filtered BMPs
#print(filtered_bmps)

# %%
