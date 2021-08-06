: '
    Pipeline to donwload and preprocess entelai-predoc data.
'

# Download all data (there is less than 2500 entries, thus, we let the total_entry parameter as it default 2500)
# Output is saved in the default location ./responses.json
python get_data.py

# Filter data from ./responses.json to retain only entries from FLENI (organization_id == 2) into ./filtered_responses.json
python filter_data_by_organization.py --check_integrity

# Convert ./filtered_responses.json data from JSON to CSV in ./filtered_responses.csv 
python json_data_to_csv.py

# generate an excel file with only the DNI and date columns to do data curation where doctors will fill for each patient and data, its diagnose.
python remove_info_cols_for_data_curations.py