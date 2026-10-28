import os
import zipfile
import io
import requests
import pandas as pd

def download_uci_online_retail():
    data_raw_dir = os.path.join(os.getcwd(), 'data', 'raw')
    os.makedirs(data_raw_dir, exist_ok=True)
    
    zip_url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    csv_target = os.path.join(data_raw_dir, "online_retail.csv")
    xlsx_target = os.path.join(data_raw_dir, "Online Retail.xlsx")
    
    print(f"Connecting to official UCI repository: {zip_url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    response = requests.get(zip_url, headers=headers, stream=True)
    if response.status_code != 200:
        # Fallback to direct xlsx URL if zip fails
        fallback_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        print(f"Zip returned status {response.status_code}. Trying direct URL: {fallback_url}...")
        response = requests.get(fallback_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(xlsx_target, "wb") as f:
            f.write(response.content)
    else:
        print("Extracting zip archive...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(data_raw_dir)
            print(f"Extracted archive files to: {data_raw_dir}")
            
    # Check if xlsx exists
    if not os.path.exists(xlsx_target):
        # Find any .xlsx extracted
        for file in os.listdir(data_raw_dir):
            if file.endswith('.xlsx'):
                xlsx_target = os.path.join(data_raw_dir, file)
                break
                
    print(f"Reading Excel dataset from {xlsx_target} (this may take a few moments)...")
    df = pd.read_excel(xlsx_target)
    
    print(f"Dataset successfully loaded. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"Saving to CSV at {csv_target}...")
    df.to_csv(csv_target, index=False)
    print("CSV conversion complete!")
    
    # Create a fast curated demo dataset (15,000 clean transactions)
    # Filter out cancelled invoices (InvoiceNo starting with 'C'), missing descriptions, missing CustomerIDs
    clean_df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)].dropna(subset=['Description', 'CustomerID'])
    clean_df = clean_df[(clean_df['Quantity'] > 0) & (clean_df['UnitPrice'] > 0)]
    
    # Compute TotalAmount
    clean_df['TotalAmount'] = (clean_df['Quantity'] * clean_df['UnitPrice']).round(2)
    
    data_processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    os.makedirs(data_processed_dir, exist_ok=True)
    sample_csv = os.path.join(data_processed_dir, "online_retail_demo.csv")
    
    # Take a balanced sample of 25,000 rows for high-speed app interaction
    if len(clean_df) > 25000:
        demo_df = clean_df.sample(n=25000, random_state=42)
    else:
        demo_df = clean_df
        
    demo_df.to_csv(sample_csv, index=False)
    print(f"Curated clean demo dataset saved to {sample_csv} with {len(demo_df)} rows and {demo_df['Description'].nunique()} unique products!")

if __name__ == '__main__':
    download_uci_online_retail()
