import argparse
import pandas as pd
from datetime import datetime
import numpy as np

def main(input, output):

    usecols = [
        'patient_identifier', 
        'questionnarie_date',
        'medical_consultation_id',
        'patient_id',
        'practitioner_id'
    ]

    renamecols = {
        'patient_id': 'ID paciente',
        'patient_identifier': 'DNI paciente',
        'medical_consultation_id': 'ID consulta',
        'practitioner_id': 'ID médico',
        'questionnarie_date': 'Fecha consulta'
    }

    data = pd.read_csv(
        input,
        sep=';',
        usecols=usecols,
        dtype='int',
        parse_dates=['questionnarie_date']
    )

    data['questionnarie_date'] = data['questionnarie_date'].dt.date

    data.rename(columns=renamecols, inplace=True)
    
    writer = pd.ExcelWriter(output,  date_format='dd-mm-YYYY')

    data.to_excel(
        writer,
        index=False,
        encoding='utf-8'
    )

    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remove all the columns information to performe a data curation using the patient identifier and the questioner creation time')

    parser.add_argument(
        '--input',
        type=str,
        default='../data/preprocessed_predoc_fleni/filtered_responses.csv',
        help='The path to the CSV file containing the raw data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/preprocessed_predoc_fleni/FLENI_predoc_curation.xlsx',
        help='The output path where to store the resulting excel data'
    )

    args = parser.parse_args()

    main(args.input, args.output)
