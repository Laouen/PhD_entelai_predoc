from tqdm import tqdm
import argparse
import json

def main(input, organization_id, output, check_integrity):
    
    all_data = json.load(open(input, 'r'))

    if check_integrity:
        # This verifies that all entries are dictionaries with exactly one key that is a number and a dictionaty as value.
        # It also verifies that all ditionaries have an organization_id field

        print('check data integrity')
        for data in tqdm(all_data):
            keys = list(data.keys())
            assert len(keys) == 1
            assert keys[0].isdecimal()
            assert 'organization_id' in data[keys[0]]
    
    print(f'Filtering data to obtain entries from organization ID {organization_id}')
    filtered_data = [
        d for d in tqdm(all_data)
        if list(d.values())[0]['organization_id'] == organization_id
    ]

    json.dump(filtered_data, open(output, 'a'));

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Filter entelai predoc raw data by organization')

    parser.add_argument(
        '--input',
        type=str,
        default='../data/preprocessed_predoc_fleni/responses.json',
        help='the input file where all the unfiltered data is stored. Default as responses.json which is the default output of get_data.py'
    )
    
    parser.add_argument(
        '--organization_id',
        type=int,
        default=2,
        help='the id of the organization that indicate the organization we want to retain data. Default is 2 which is FLENI organization id.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/preprocessed_predoc_fleni/filtered_responses.json',
        help='The output file where to store the filtered data. Default as ./filtered_responses.json'
    )

    parser.add_argument(
        '--check_integrity',
        default=False,
        action='store_true',
        help='If true, checks the data integrity with basic asserts to check the data is complete and has the expected shape'
    )

    args = parser.parse_args()

    main(args.input, args.organization_id, args.output, args.check_integrity)
