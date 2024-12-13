import pandas as pd
import re
from urllib.parse import urlparse

# Function to validate a URL
def is_valid_url(url):
    regex = re.compile(
        r'^(https?://)?'  # Optional scheme
        r'(www\.)?'       # Optional www
        r'[a-zA-Z0-9.-]+' # Domain name
        r'\.[a-zA-Z]{2,}' # Top-level domain
        r'(/.*)?$',       # Optional path
        re.IGNORECASE
    )
    return re.match(regex, url) is not None

# Function to extract the main domain from a URL
def extract_main_domain(url):
    try:
        # Parse the URL
        parsed_url = urlparse(url if url.startswith("http") else f"http://{url}")
        # Split domain into parts
        domain_parts = parsed_url.netloc.split(".")
        # Return main domain (e.g., "godaddysites.com" from "subdomain.godaddysites.com")
        return ".".join(domain_parts[-2:]) if len(domain_parts) > 1 else parsed_url.netloc
    except Exception as e:
        return None

# Function to clean and validate the data
def clean_and_validate(input_file, output_file):
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)

        # Ensure the 'Domain' column exists
        if 'Domain' not in df.columns:
            print(f"'Domain' column not found in {input_file}.")
            return

        # Drop rows with missing or empty values in the 'Domain' column
        df = df.dropna(subset=['Domain'])
        df = df[df['Domain'].str.strip() != '']

        # Normalize the URLs: strip whitespaces and convert to lowercase
        df['Domain'] = df['Domain'].str.strip().str.lower()

        # Validate the URLs and keep only valid ones
        df = df[df['Domain'].apply(is_valid_url)]

        # Extract the main domain
        df['Main_Domain'] = df['Domain'].apply(extract_main_domain)

        # Remove duplicates based on the main domain
        df = df.drop_duplicates(subset=['Main_Domain'])

        # Drop the helper 'Main_Domain' column before saving
        df = df.drop(columns=['Main_Domain'])

        # Save the cleaned data to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Main function
if __name__ == "__main__":
    # File paths
    input_file = "urls.csv"  # Replace with the path to your input CSV
    output_file = "cleaned_urls.csv"  # Replace with the path to your output CSV

    # Clean and validate the data
    clean_and_validate(input_file, output_file)
