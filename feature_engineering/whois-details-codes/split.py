import pandas as pd
import os

# Function to split the cleaned data into chunks
def split_into_chunks(input_file, output_dir, chunk_size=10000):
    try:
        # Load the cleaned CSV file
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Calculate the number of chunks
        total_rows = len(df)
        num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size != 0 else 0)

        # Split and save each chunk
        for i in range(num_chunks):
            start_row = i * chunk_size
            end_row = min(start_row + chunk_size, total_rows)

            chunk = df.iloc[start_row:end_row]
            chunk_file = os.path.join(output_dir, f"chunk_{i+1}.csv")

            chunk.to_csv(chunk_file, index=False)
            print(f"Saved {chunk_file} with {len(chunk)} rows.")

    except Exception as e:
        print(f"Error splitting file: {e}")

# Main function
if __name__ == "__main__":
    # File paths
    input_file = "cleaned_urls.csv"  # Replace with your cleaned CSV file
    output_dir = "chunks"  # Directory to save the chunks

    # Split the cleaned data into chunks
    split_into_chunks(input_file, output_dir)
