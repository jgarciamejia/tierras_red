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
    count = 0
    items = os.listdir(folder_path)
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            count += 1
    return count

def count_files_with_strings(folder_path, strings_to_count):
    # Initialize a dictionary to store counts for each string
    counts = {string: 0 for string in strings_to_count}

    # Use os.listdir to get a list of all items (files and directories) in the folder
    items = os.listdir(folder_path)

    # Iterate through the items and count files that contain each string
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            for string in strings_to_count:
                if string in item:
                    counts[string] += 1

    return counts

# Define path of Tierras data 
incomingpath = '/data/tierras/incoming'

# Authenticate and open the Tierras Obs Log Google Sheet
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/spreadsheets',
         'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
credentials_folder = os.getcwd()
credentials_file = os.path.join(credentials_folder, 'tierras-obs-log-b5e277783823.json')
credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
gc = gspread.authorize(credentials)
sheet = gc.open('Tierras_Observing_Log').worksheet(str('2024'))  # TODO: automate creation of new year log sheet!
gs_dates = sheet.col_values(1)

# Fetch content of 60-in log tree link containing the names of all logs since 2016
tree_url = 'http://linmax.sao.arizona.edu/60logs/'
tree_response = requests.get(tree_url)

if tree_response.status_code != 200:
    print(f"Failed to retrieve the page. Status code: {tree_response.status_code}")
else:
    content = tree_response.text

    # Parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')

    # Find all links in the page (assuming logs are listed as <a> elements)
    log_links = soup.find_all('a')

    # Catch up from a given date to today
    # Define the start date and the end date (today's date)
    start_date = datetime(2024, 5, 25)
    end_date = datetime.now().date()

    # Initialize a list to store the dates
    dates_list = []

    # Loop through each day between start_date and end_date
    current_date = start_date

    while current_date.date() <= end_date:
        
        formatted_date = current_date.strftime("%Y%m%d")

        # Populate Columns 2 & 3 : ID targets observed on date and nexp per target
        col2, col3 = 2, 3
        datepath = os.path.join(incomingpath,formatted_date)
        
        flist = os.listdir(datepath) 

        # If no observations gathered on a given date
        if len(flist) < 1:
            sheet.update_cell(row, col2, 'No Observations Gathered')
            sheet.update_cell(row, col3, '0')
        
        else:
            target_list = set([flist[ind].split('.')[2] for ind in range(len(flist))])
            if 'FLAT001' in target_list:
                target_list = [targetname for targetname in target_list if not targetname.startswith('FLAT')]
                target_list.append('FLAT')
            
            nfiles_per_targets = count_files_with_strings(datepath,target_list)
            nfilesum = 0
            for target,nfiles in nfiles_per_targets.items():
                nfilesum += nfiles

            targets = list(nfiles_per_targets.keys())
            nexps = list(nfiles_per_targets.values())

            targets_str = '/'.join(targets)
            nexps_str = '/'.join(map(str, nexps))

            sheet.update_cell(row, col2, targets_str)
            sheet.update_cell(row, col3, nexps_str)


        # Now, populate Column 4 with the No. of hours observed by the 60-inch. 

        # Re-ormat the date as needed (YYYY.MM.DD)
        formatted_date2 = current_date.strftime("%Y.%m.%d")

        # Regular expression pattern to match the current date log filename
        pattern = re.compile(r'^60log\.{}.[A-Za-z0-9]+\.shtml$'.format(formatted_date2))

        # Extract dates from the link texts or URLs
        log_name = None
        for link in log_links:
            href = link.get('href')
            # Check if the link href matches the log pattern
            if href and pattern.match(href):
                log_name = href
                break

        if log_name:
            print(f"Log for {formatted_date2}: {log_name}")

            # Generate the URL with the formatted date
            log_url = "http://linmax.sao.arizona.edu/60logs/" + log_name

            # Define regular expression pattern to search in the date log url
            pattern = re.compile(r'HOURS OBSERVED -- (\d{1,2}(?:\.\d{1,2})?)')

            # Fetch the content of the date log URL
            log_response = requests.get(log_url)
            found_hours = []

            if log_response.status_code == 200:
                log_content = log_response.text

                # Fetch the number of hours observed by the 60 in on that date
                for match in pattern.findall(log_content):
                    print(match)
                    hour = float(match)
                    if 0 <= hour <= 24:
                        found_hours.append(hour)

            # Ensure found_hours is not empty
            if found_hours:
                # Fetch the row of the date on the Tierras Observing log
                formatted_date3 = current_date.strftime("%m/%d/%Y")
                try:
                    row = [ind for ind, date in enumerate(gs_dates) if date == formatted_date3][0] + 1
                    # Column 4 = write number of hours observed by 60 in
                    col4 = 4
                    sheet.update_cell(row, col4, found_hours[0])
                except IndexError:
                    print(f"No matching date found in the Google Sheet for {formatted_date3}")

        # Move to the next day
        current_date += timedelta(days=1)


    #     else:
    #         print(f"No log found for {formatted_date}")
    # else:
    #     print(f"Failed to retrieve the page. Status code: {response.status_code}")





# # Script generated with the aid of CHATGPT: https://chat.openai.com/share/1d740f94-32b9-47d4-a62f-9112e1494544,
# https://chatgpt.com/share/93eafecb-c0a0-42a9-bbc5-2b9ba7853c4f