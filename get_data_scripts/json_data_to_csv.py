import argparse
import pandas as pd
import json


def json_to_csv(json_data):
    """
    Converts json responses to dataframe format
    """
    df = pd.DataFrame()
    for elem in json_data:
        df = pd.concat([df, pd.DataFrame(elem)], axis=1, sort=False)
    df = df.transpose()
    return df

def main(input, output):

    all_data = json.load(open(input, 'r'))
    df = json_to_csv(all_data)
    df.to_csv(output, sep=';')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert JSON formatted raw data into a CSV')

    parser.add_argument(
        '--input',
        type=str,
        default='../data/preprocessed_predoc_fleni/filtered_responses.json',
        help='The path to the JSON file containing the raw data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/preprocessed_predoc_fleni/filtered_responses.csv',
        help='The output path where to store the resulting CSV data'
    )

    args = parser.parse_args()

    main(args.input, args.output)
