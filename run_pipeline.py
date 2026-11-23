import os
import re
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions

output_dir = r"C:\Users\engsu\.gemini\antigravity\scratch\ecommerce_scraping"

def get_webdriver():
    options = EdgeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Edg/124.0.0.0')
    try:
        return webdriver.Edge(options=options)
    except Exception as e:
        print(f"Edge driver error: {e}. Trying Chrome...")
        c_opts = ChromeOptions()
        c_opts.add_argument('--headless=new')
        c_opts.add_argument('--disable-gpu')
        c_opts.add_argument('--no-sandbox')
        c_opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')
        return webdriver.Chrome(options=c_opts)

driver = get_webdriver()

# ==============================================================================
# 1. Scrape Amazon Egypt (Verified Extraction)
# ==============================================================================
print("--- 1. Scraping Amazon Egypt ---")
driver.get('https://www.amazon.eg/s?k=smartphones&language=en_AE')
time.sleep(5)
soup_amz = BeautifulSoup(driver.page_source, 'html.parser')

amz_items = soup_amz.select('div[data-component-type="s-search-result"]')
print(f"Found {len(amz_items)} product listings on Amazon Egypt.")

amazon_data = []
for item in amz_items:
    # 1. Product Name
    title_el = item.select_one('h2 span') or item.select_one('h2 a')
    title = title_el.get_text(strip=True) if title_el else None
    if not title:
        continue
        
    # 2. Price
    price_offscreen = item.select_one('.a-price .a-offscreen')
    if price_offscreen:
        price = price_offscreen.get_text(strip=True)
    else:
        price_whole = item.select_one('.a-price-whole')
        price = f"EGP {price_whole.get_text(strip=True).replace(',', '')}" if price_whole else None
            
    # 3. Rating
    rating_el = item.select_one('i.a-icon-star-small span') or item.select_one('.a-icon-alt')
    rating = rating_el.get_text(strip=True) if rating_el else None
    
    # 4. Availability (Verified from page, not assumed from price)
    card_text = item.get_text(separator=' ', strip=True).lower()
    stock_match = re.search(r'only\s+(\d+)\s+left in stock', card_text)
    if stock_match:
        avail = f"Only {stock_match.group(1)} left in stock"
    elif 'currently unavailable' in card_text:
        avail = 'Currently unavailable'
    elif 'out of stock' in card_text:
        avail = 'Out of Stock'
    else:
        # Cannot be verified from search result card alone
        avail = np.nan
        
    amazon_data.append({
        'Product Name': title,
        'Price': price,
        'Rating': rating,
        'Availability': avail,
        'Source': 'Amazon'
    })

df_amazon = pd.DataFrame(amazon_data)
print(f"Scraped {len(df_amazon)} records from Amazon Egypt.")

# ==============================================================================
# 2. Scrape Jumia Egypt (Verified Extraction)
# ==============================================================================
print("\n--- 2. Scraping Jumia Egypt ---")
driver.get('https://www.jumia.com.eg/smartphones/')
time.sleep(5)
soup_jumia = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

jumia_cards = soup_jumia.select('article.prd')
print(f"Found {len(jumia_cards)} product listings on Jumia Egypt.")

jumia_data = []
for c in jumia_cards:
    name_el = c.select_one('h3.name')
    title = name_el.get_text(strip=True) if name_el else None
    if not title:
        continue
        
    prc_el = c.select_one('.prc')
    price = prc_el.get_text(strip=True) if prc_el else None
    
    rev_el = c.select_one('.rev')
    rating = rev_el.get_text(strip=True) if rev_el else None
    
    # Availability (Verified from badges/tags, not assumed from price)
    tag_el = c.select_one('.tag')
    tag_text = tag_el.get_text(strip=True).lower() if tag_el else ''
    card_text = c.get_text(separator=' ', strip=True).lower()
    
    if 'out of stock' in tag_text or 'out of stock' in card_text:
        avail = 'Out of Stock'
    elif 'low stock' in tag_text or 'low stock' in card_text:
        avail = 'Low Stock'
    else:
        # Cannot be verified from search result card alone
        avail = np.nan
        
    jumia_data.append({
        'Product Name': title,
        'Price': price,
        'Rating': rating,
        'Availability': avail,
        'Source': 'Jumia'
    })

df_jumia = pd.DataFrame(jumia_data)
print(f"Scraped {len(df_jumia)} records from Jumia Egypt.")

# ==============================================================================
# 3. Combine Datasets (pd.concat)
# ==============================================================================
print("\n--- 3. Combining Raw Datasets ---")
df_combined = pd.concat([df_amazon, df_jumia], ignore_index=True)
print(f"Combined Raw Dataset Shape: {df_combined.shape}")

# Save raw dataset
raw_csv_path = os.path.join(output_dir, "smartphones_raw.csv")
df_combined.to_csv(raw_csv_path, index=False, encoding='utf-8-sig')
print(f"Saved raw dataset to: {raw_csv_path}")

# ==============================================================================
# 4. Overview Before Preprocessing
# ==============================================================================
print("\n--- 4. Dataset Overview [Before Preprocessing] ---")
print("\nhead():")
print(df_combined.head())

print("\ninfo():")
df_combined.info()

print("\ndescribe(include='all'):")
print(df_combined.describe(include='all'))

print("\nisnull().sum():")
print(df_combined.isnull().sum())

# ==============================================================================
# 5. Data Preprocessing & Validation
# ==============================================================================
print("\n--- 5. Data Preprocessing & Validation ---")
df_cleaned = df_combined.copy()

# Step 5.1: Text Cleaning (Remove extra whitespaces & unify casing)
for col in ['Product Name', 'Availability', 'Source']:
    # Standardize casing to lowercase while preserving NaN values
    df_cleaned[col] = df_cleaned[col].apply(
        lambda x: re.sub(r'\s+', ' ', str(x)).strip().lower() if pd.notna(x) else np.nan
    )

# Standardize Source naming
df_cleaned['Source'] = df_cleaned['Source'].replace({'amazon': 'Amazon', 'jumia': 'Jumia'})

# Step 5.2: Convert Price to Numerical Values (float)
def clean_price(val):
    if pd.isna(val) or str(val).strip().lower() in ['none', 'nan', '']:
        return np.nan
    val_clean = str(val).replace('\xa0', ' ').replace(',', '')
    match = re.search(r'(\d+(?:\.\d+)?)', val_clean)
    return float(match.group(1)) if match else np.nan

df_cleaned['Price'] = df_cleaned['Price'].apply(clean_price)

# Step 5.3: Convert Rating to Numerical Values (float)
def clean_rating(val):
    if pd.isna(val) or str(val).strip().lower() in ['none', 'nan', '']:
        return np.nan
    match = re.search(r'(\d+(?:\.\d+)?)', str(val))
    return float(match.group(1)) if match else np.nan

df_cleaned['Rating'] = df_cleaned['Rating'].apply(clean_rating)

# Step 5.4: Product & Price Validation (Identify and remove accessories / non-smartphones)
initial_records = len(df_cleaned)
# Investigate items with suspicious prices (< 1000 EGP) or non-smartphone feature phones
is_accessory = (df_cleaned['Price'].notna()) & (df_cleaned['Price'] < 1000)
is_feature_phone = df_cleaned['Product Name'].str.contains('nokia 106', case=False, na=False)

non_smartphones = df_cleaned[is_accessory | is_feature_phone]
print(f"\nIdentified {len(non_smartphones)} non-smartphone / accessory records:")
for _, r in non_smartphones.iterrows():
    print(f"  - [{r['Source']}] Price: {r['Price']} EGP | {r['Product Name']}")

df_cleaned = df_cleaned[~(is_accessory | is_feature_phone)].copy().reset_index(drop=True)
print(f"Filtered out non-smartphones. Valid smartphones count: {len(df_cleaned)}")

# Step 5.5: Deduplication
# Remove duplicate listings within the same source platform, preserving cross-platform overlap
before_dedup = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates(subset=['Product Name', 'Source'], keep='first').reset_index(drop=True)
print(f"Deduplication: Removed {before_dedup - len(df_cleaned)} duplicate rows within platforms.")
print(f"Total unique valid smartphone records: {len(df_cleaned)}")

# Step 5.6: Handling Missing Values (Source-Specific Median Imputation)
print("\nMissing values before imputation:")
print(df_cleaned.isnull().sum())

# Calculate medians separately for Amazon and Jumia
median_ratings = df_cleaned.groupby('Source')['Rating'].median().round(1)
median_prices = df_cleaned.groupby('Source')['Price'].median().round(2)
print(f"\nSource-Specific Median Ratings:\n{median_ratings}")
print(f"Source-Specific Median Prices:\n{median_prices}")

# Impute ratings using source-specific median
df_cleaned['Rating'] = df_cleaned.groupby('Source')['Rating'].transform(
    lambda s: s.fillna(round(s.median(), 1))
)

# Impute prices using source-specific median if any missing
if df_cleaned['Price'].isnull().sum() > 0:
    df_cleaned['Price'] = df_cleaned.groupby('Source')['Price'].transform(
        lambda s: s.fillna(round(s.median(), 2))
    )

# Note on Availability:
# When stock status is unverified on the search card, it is retained as NaN
# to avoid fabricating inventory claims.

# ==============================================================================
# 6. Overview After Preprocessing
# ==============================================================================
print("\n--- 6. Dataset Overview [After Preprocessing] ---")
print("\nhead():")
print(df_cleaned.head())

print("\ninfo():")
df_cleaned.info()

print("\ndescribe():")
print(df_cleaned.describe())

print("\nisnull().sum():")
print(df_cleaned.isnull().sum())

# Save cleaned dataset
cleaned_csv_path = os.path.join(output_dir, "smartphones_cleaned.csv")
df_cleaned.to_csv(cleaned_csv_path, index=False, encoding='utf-8-sig')
print(f"\nSaved cleaned dataset to: {cleaned_csv_path}")

print(f"\nFinal Valid Smartphone Records: {len(df_cleaned)}")
print(f"Remaining Missing Values:\n{df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0]}")
