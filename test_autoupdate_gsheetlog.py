import os
import re
import glob
import argparse
import requests
import gspread
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from oauth2client.service_account import ServiceAccountCredentials
from gspread.exceptions import APIError


def count_files_in_folder(folder_path):
    """Counts the number of files in a given folder."""
    return sum(os.path.isfile(os.path.join(folder_path, item)) for item in os.listdir(folder_path))

def count_files_with_strings(folder_path, strings_to_count):
    """Counts the number of files that contain specific strings in their names."""
    counts = {string: 0 for string in strings_to_count}
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            for string in strings_to_count:
                if string in item:
                    counts[string] += 1
    return counts

def authenticate_google_sheets(credentials_file, scope):
    """Authenticates and returns a Google Sheets client."""
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    return gspread.authorize(credentials)

def fetch_log_links(tree_url):
    """Fetches and returns log links from a given URL."""
    response = requests.get(tree_url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the page. Status code: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.find_all('a')

def update_with_retry(sheet, updates, max_retries=5):
    """Updates multiple cells with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            # Batch update all cells at once
            sheet.batch_update(updates)
            return True
        except APIError as e:
            if e.response.status_code == 429:  # Rate limit error
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 5, 9, 17, 33 seconds
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise
    print(f"Failed to update after {max_retries} retries")
    return False

def main():
    incomingpath = '/data/tierras/incoming'
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
    credentials_file = os.path.join(os.getcwd(), 'tierras-obs-log-b5e277783823.json')
    gc = authenticate_google_sheets(credentials_file, scope)
    sheet = gc.open('Tierras_Observing_Log').worksheet('2025')
    gs_dates = sheet.col_values(1)
    col2, col3, col4 = 2, 3, 4
    
    tree_url = 'http://linmax.sao.arizona.edu/60logs/'
    log_links = fetch_log_links(tree_url)
    
<<<<<<< Updated upstream
    start_date = datetime(2024, 10, 7) #YYYY,MM,DD
=======
    start_date = datetime(2025, 11, 25)
>>>>>>> Stashed changes
    end_date = datetime.now().date()
    
    # Collect all updates to batch them
    batch_updates = []
    
    current_date = start_date
    while current_date.date() <= end_date:
        formatted_date = current_date.strftime("%Y%m%d")
        datepath = os.path.join(incomingpath, formatted_date)
        row = [ind for ind, date in enumerate(gs_dates) if date == current_date.strftime("%m/%d/%Y")]
        
        if not row:
            print(f'{current_date} not found on Google Sheets')
            current_date += timedelta(days=1)
            continue
        row = row[0] + 1

        try:
            flist = os.listdir(datepath)
            target_list = set(f.split('.')[2] for f in flist)
            if 'FLAT001' in target_list:
                target_list = {t for t in target_list if not t.startswith('FLAT')}
                target_list.add('FLAT')
            
            nfiles_per_targets = count_files_with_strings(datepath, target_list)
            targets_str = '/'.join(nfiles_per_targets.keys())
            nexps_str = '/'.join(map(str, nfiles_per_targets.values()))
            
            batch_updates.append({'range': f'B{row}', 'values': [[targets_str]]})
            batch_updates.append({'range': f'C{row}', 'values': [[nexps_str]]})
        except FileNotFoundError:
            batch_updates.append({'range': f'B{row}', 'values': [['No Observations Gathered']]})
            batch_updates.append({'range': f'C{row}', 'values': [['0']]})
        
        formatted_date2 = current_date.strftime("%Y.%m.%d")
        pattern = re.compile(r'^60log\.{}.[A-Za-z0-9]+\.shtml$'.format(formatted_date2))
        log_names = [link.get('href') for link in log_links if link.get('href') and pattern.match(link.get('href'))]

        for log_name in log_names:
            log_url = f"http://linmax.sao.arizona.edu/60logs/{log_name}"
            log_response = requests.get(log_url)
            print(log_name, log_response)

            if log_response.status_code == 200:
                log_content = log_response.text
                found_hours = [float(match) for match in re.findall(r'HOURS OBSERVED -- (\d{1,2}(?:\.\d{1,2})?)', log_content) if 0 <= float(match) <= 24]
                if found_hours:
                    print(f'{sheet}, {row}, {col4}, {found_hours[0]}')
                    batch_updates.append({'range': f'D{row}', 'values': [[found_hours[0]]]})
        
        # Batch update every 50 updates to stay well under the 60/min limit
        if len(batch_updates) >= 50:
            print(f"Performing batch update of {len(batch_updates)} cells...")
            update_with_retry(sheet, batch_updates)
            batch_updates = []
            time.sleep(1)  # Small delay between batches
        
        current_date += timedelta(days=1)
    
    # Update any remaining cells
    if batch_updates:
        print(f"Performing final batch update of {len(batch_updates)} cells...")
        update_with_retry(sheet, batch_updates)

if __name__ == "__main__":
    main()
