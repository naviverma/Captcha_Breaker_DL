import os
import requests
import concurrent.futures
import time

shortname = 'nasingh'
parallel_jobs = 10
max_retries = 100

# Creating a captcha directory
if not os.path.exists('captchas'):
    os.makedirs('captchas')

# Removing previous logs
if os.path.exists('successes.txt'):
    os.remove('successes.txt')
if os.path.exists('failed_downloads.txt'):
    os.remove('failed_downloads.txt')

# Read the file list
with open('file_list.txt', 'r') as f:
    filenames = [line.strip() for line in f.readlines()]

# Function to download a single file with retries
def process_file(filename):
    retry_count = 0
    success = False

    while retry_count < max_retries:
        try:
            response = requests.get('https://cs7ns1.scss.tcd.ie', params={'shortname': shortname, 'myfilename': filename})
            if response.status_code == 200:
                # Save the file
                with open(f'captchas/{filename}', 'wb') as f:
                    f.write(response.content)
                print(f"Download succeeded for {filename}.")
                with open('successes.txt', 'a') as success_log:
                    success_log.write(f"{filename}\n")
                success = True
                break
            else:
                print(f"Failed to download {filename}. Status code: {response.status_code}")
                retry_count += 1
                time.sleep(0.1)  # Small delay before retrying
        except Exception as e:
            print(f"An error occurred while downloading {filename}: {e}")
            retry_count += 1
            time.sleep(0.1)  # Small delay before retrying

    if not success:
        with open('failed_downloads.txt', 'a') as failed_log:
            failed_log.write(f"{filename}\n")

# Start timing the download process
start_time = time.time()

# Use ThreadPoolExecutor to download files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
    executor.map(process_file, filenames)

# End timing the download process
end_time = time.time()
total_time = end_time - start_time

# Summarize results
success_count = len(open('successes.txt').readlines()) if os.path.exists('successes.txt') else 0
failed_count = len(open('failed_downloads.txt').readlines()) if os.path.exists('failed_downloads.txt') else 0
print(f"Download completed: {success_count} successful, {failed_count} failed.")
print(f"Total time taken for downloading: {total_time:.2f} seconds")
