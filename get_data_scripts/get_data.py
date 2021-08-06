from tqdm import tqdm
import requests
import argparse
import json

def main(total_entries, limit, output):
    with requests.Session() as s:

        # Retrieve the CSRF token first
        s.get('https://predoc.entelai.com')  # sets cookie
        if 'csrftoken' in s.cookies:
            # Django 1.6 and up
            csrftoken = s.cookies['csrftoken']
        else:
            # older versions
            csrftoken = s.cookies['csrf']

        # login
        login_data = dict(
            username=input('Username: '),
            password=input('Password: '),
            csrfmiddlewaretoken=csrftoken,
            next='/'
        )
        login_url = 'https://predoc.entelai.com/admin/login/?next=/admin/'
        s.post(
            login_url,
            data=login_data,
            headers=dict(Referer='https://predoc.entelai.com')
        )

        # Collect data and write it in a file
        res = []
        with open(output, 'a') as writer:
            
            pbar = tqdm(range(0, total_entries, limit))
            for offset in pbar:
                response = s.get(f'https://predoc.entelai.com/rna/{offset}/{limit}/data')

                if response.status_code == 200:
                    data = response.json()['data']
                    pbar.set_description(f'total data obtained for range [{offset}, {offset+limit}] is: {len(data)}')
                    res += data
                
                else:
                    pbar.set_description(f'Get fail with status code {response.status_code} for range [{offset},{offset+limit}]')
                    break

            writer.write(json.dumps(res, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Donwload the entelai-predoc raw data')

    parser.add_argument(
        '--total_entries',
        type=int,
        default=2500,
        help='The total number of entries to download. -1 means download the entire dataset'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=40,
        help='the number of entries to download at the same time'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/preprocessed_predoc_fleni/responses.json',
        help='The output file where to store the downloaded data in JSON format'
    )

    args = parser.parse_args()

    main(
        args.total_entries,
        args.limit,
        args.output,
        args.username,
        args.password
    )
