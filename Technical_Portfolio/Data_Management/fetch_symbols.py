import requests
import time
import os

def fetch_symbols(categories, processed_file='processed_categories.txt'):
    all_symbols = []
    processed_categories = load_processed_categories(processed_file)

    for category in categories:
        if category in processed_categories:
            print(f"Skipping category '{category}', already processed.")
            continue

        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category={category}" 
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    symbols = [coin['symbol'].upper() + 'USDT' for coin in data]
                    all_symbols.extend(symbols)
                    print(f"Fetching symbols for category '{category}' successful.")
                    
                    # Mark this category as processed
                    save_processed_category(category, processed_file)
                    break
                
                elif response.status_code == 429:
                    print(f"Rate limit hit. Waiting before retrying... (Category: {category})")
                    time.sleep(60)  # Adjust this wait time based on rate limits
                else:
                    print(f"Failed to fetch symbols for category '{category}', status code: {response.status_code}")
                    break

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for category '{category}': {e}")
                time.sleep(2)

    return all_symbols

def send_to_server(symbols):
    url = 'http://localhost:5000/receive_symbols'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json={'symbols': symbols}, headers=headers)
    if response.status_code == 200:
        print("Symbols sent successfully")
    else:
        print(f"Failed to send symbols, status code: {response.status_code}")

# Functions to load and save processed categories
def load_processed_categories(filename):
    # Get the absolute path of the file
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, filename)
    
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def save_processed_category(category, filename):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'a') as f:
        f.write(category + '\n')

def upload_symbols(symbols, filename='symbols.txt'):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path
    file_path = os.path.join(script_dir, filename)
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print("The file contains content.")
    else:
        # Write to a .txt file, 5 elements per line
         with open(file_path, "w", encoding = 'utf-8') as file:
            for i in range(0, len(symbols), 5):  # Iterate in steps of 5
                line = " ".join(symbols[i:i+5])  # Get a slice of 5 elements
                file.write(line + "\n")  # Write the line to the file with a newline character
        


# List of all categories
all_categories = ['layer-1'
    # 'layer-1', 'depin', 'proof-of-work-pow', 'proof-of-stake-pos', 'meme-token', 'dog-themed-coins', 
    # 'eth-2-0-staking', 'non-fungible-tokens-nft', 'governance', 'artificial-intelligence', 
    # 'infrastructure', 'layer-2', 'zero-knowledge-zk', 'storage', 'oracle', 'bitcoin-fork', 
    # 'restaking', 'rollup', 'metaverse', 'privacy-coins', 'layer-0-l0', 'solana-meme-coins', 
    # 'data-availability', 'internet-of-things-iot', 'frog-themed-coins', 'ai-agents', 
    # 'superchain-ecosystem', 'bitcoin-layer-2', 'bridge-governance-tokens', 'modular-blockchain', 
    # 'cat-themed-coins', 'cross-chain-communication', 'analytics', 'identity', 'wallets', 'masternodes'
]


# import chardet

def get_symbols():
    symbols = []
     # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    
    # Construct the full path to symbols.txt
    file_path = os.path.join(current_dir, 'symbols.txt')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            for word in line.split():
                symbols.append(word)
    return symbols



# Fetch symbols and send to server
fetched_symbols = fetch_symbols(all_categories)
symbols = get_symbols()
print(symbols)
upload_symbols(fetched_symbols)
# send_to_server(symbols)


