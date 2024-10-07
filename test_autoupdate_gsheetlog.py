import os
import re
import glob
import argparse
import requests
import gspread
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from oauth2client.service_account import ServiceAccountCredentials


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

def update_google_sheet(sheet, row, col, value):
    """Updates a specific cell in a Google Sheet."""
    sheet.update_cell(row, col, value)

def main():
    incomingpath = '/data/tierras/incoming'
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
    credentials_file = os.path.join(os.getcwd(), 'tierras-obs-log-b5e277783823.json')
    gc = authenticate_google_sheets(credentials_file, scope)
    sheet = gc.open('Tierras_Observing_Log').worksheet('2024')  # TODO: automate creation of new year log sheet!
    gs_dates = sheet.col_values(1)
    col2, col3, col4 = 2, 3, 4
    
    tree_url = 'http://linmax.sao.arizona.edu/60logs/'
    log_links = fetch_log_links(tree_url)
    
    start_date = datetime(2024, 10, 7) #YYYY,MM,DD
    end_date = datetime.now().date()
    
    current_date = start_date
    while current_date.date() <= end_date:
        formatted_date = current_date.strftime("%Y%m%d")
        datepath = os.path.join(incomingpath, formatted_date)
        row = [ind for ind, date in enumerate(gs_dates) if date == current_date.strftime("%m/%d/%Y")]
        
        if not row:
            print ('{current_date} not found on Google Sheets'.format(current_date))
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
            
            update_google_sheet(sheet, row, col2, targets_str)
            update_google_sheet(sheet, row, col3, nexps_str)
        except FileNotFoundError:
            update_google_sheet(sheet, row, col2, 'No Observations Gathered')
            update_google_sheet(sheet, row, col3, '0')
        
        formatted_date2 = current_date.strftime("%Y.%m.%d")
        pattern = re.compile(r'^60log\.{}.[A-Za-z0-9]+\.shtml$'.format(formatted_date2))
        #log_name = next((link.get('href') for link in log_links if link.get('href') and pattern.match(link.get('href'))), None)
        log_names = [link.get('href') for link in log_links if link.get('href') and pattern.match(link.get('href'))]

        for log_name in log_names:
            #if log_name: 
            log_url = f"http://linmax.sao.arizona.edu/60logs/{log_name}"
            log_response = requests.get(log_url)
            print (log_name,log_response)

            if log_response.status_code == 200:
                log_content = log_response.text
                found_hours = [float(match) for match in re.findall(r'HOURS OBSERVED -- (\d{1,2}(?:\.\d{1,2})?)', log_content) if 0 <= float(match) <= 24]
                if found_hours:
                    print ('{},{},{},{}'.format(sheet, row, col4, found_hours[0]))
                    update_google_sheet(sheet, row, col4, found_hours[0])
            
        
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
