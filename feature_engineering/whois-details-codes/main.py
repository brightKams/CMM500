import os
import whois
import csv
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Format the dates properly
def format_date(date):
    if isinstance(date, list):
        date = date[0]  # Use the first date if it's a list
    return date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date

# Process a single URL
def process_url(url):
    try:
        w = whois.whois(url)

        # Handle domain name
        domain_name = w.domain_name
        if isinstance(domain_name, list):
            domain_name = domain_name[0].lower()

        # Format dates
        creation_date = format_date(w.creation_date)
        expiration_date = format_date(w.expiration_date)

        # Format name servers
        name_servers = ', '.join(w.name_servers) if w.name_servers else ''

        return {
            'Domain Name': domain_name if domain_name else url,  # Use domain_name if available; otherwise, fall back to the input URL
            'Registrar': w.registrar if w.registrar else 'N/A',
            'Creation Date': creation_date if creation_date else 'N/A',
            'Expiration Date': expiration_date if expiration_date else 'N/A',
            'Name Servers': name_servers if name_servers else 'N/A',
            'Error': ''  # No error
        }
    except Exception as e:
        # If an error occurs, log the error and retain the input URL as Domain Name
        return {
            'Domain Name': url,  # Retain the input domain name
            'Registrar': 'N/A',
            'Creation Date': 'N/A',
            'Expiration Date': 'N/A',
            'Name Servers': 'N/A',
            'Error': str(e)  # Log the error message
        }

# Batch processing
def process_batch(batch, start_count):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_url, url) for url in batch]
        for idx, future in enumerate(futures):
            result = future.result()
            if result:
                # Add the count to each row
                result['Count'] = start_count + idx
                results.append(result)
    return results

# Main processing logic for a single file
def process_file(input_file, output_file, batch_size=50000):
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        urls = [row['Domain'] for row in reader if 'Domain' in row and row['Domain'].strip()]
    
    # Check if URLs were read successfully
    if not urls:
        print(f"No URLs found in {input_file}.")
        return

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Count', 'Domain Name', 'Registrar', 'Creation Date', 'Expiration Date', 'Name Servers', 'Error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        global_count = 1  # Initialize the global counter
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {len(urls) // batch_size + 1} for {input_file}")
            results = process_batch(batch, global_count)
            if results:  # Check if there are valid results before writing
                writer.writerows(results)
            else:
                print(f"No valid results for batch {i // batch_size + 1} in {input_file}.")
            global_count += len(batch)
            time.sleep(2)  # Add delay between batches to prevent rate-limiting

# Process specified files
def process_files(input_folder, output_folder, file_names):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for file_name in file_names:
        input_path = os.path.join(input_folder, file_name)
        result_file_name = file_name.replace("chunk_", "result_")  # Rename chunk_1.csv to result_1.csv
        output_path = os.path.join(output_folder, result_file_name)
        print(f"Processing: {input_path}")
        process_file(input_path, output_path)  # Process the file
        print(f"Completed: {output_path}")

# Example usage
if __name__ == "__main__":
    input_folder = 'chunks'  # Folder containing the CSV files
    output_folder = 'output'  # Folder to store the results
    files_to_process = [
        'chunk_19.csv',
        'chunk_20.csv'
        ]  # List of CSV files to process
    process_files(input_folder, output_folder, files_to_process)
