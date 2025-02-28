{
# Clore Hyra Script README

This Python script interacts with [Clore AI Marketplace](https://clore.ai/) to:
1. Retrieve a list of GPU servers available for rent.
2. Compare prices and performance (bk/s) with your active orders.
3. Generate a detailed HTML report including tables, charts, and (optionally) a hashpower graph.

## Customization
All customizable settings are located at the top of the script. These include:
- **User Identity:**  
  `USER_IDENTITY = "Ollebolle22 fa35e84c8d9fe32ed99c46b76a2c3b0568480491c2223ed6f8321165fe95486e"`  
  *(Example – change to your own from btchunters.com)*
- **API Key:**  
  `API_KEY = "YOUR_CLORE_API_KEY_HERE"`
- **Jupyter Token and SSH Password:**  
  `JUPYTER_TOKEN = "example_token"`  
  `SSH_PASSWORD = "example_password"`
- **File Paths:**  
  `CACHE_FILE = "marketplace_cache.json"`  
  `HISTORY_FILE = "clore_history.log"`  
  `RESULT_HTML_FILE = "clore_rent_results.html"`  
  `MEASUREMENTS_FILE = "measurements.json"`
- **Other Settings:**  
  `COOLDOWN = 600` *(cache cooldown in seconds)*
- **Local Currency Conversion:**  
  The function `get_sek_rate()` fetches the USD→SEK exchange rate and can be modified for other local currencies.

## Requirements
- **Python 3.7 or higher**
- **Required packages:**

    ```bash
    pip install requests pandas plotly
    ```

## Installation and Usage
1. **Clone or download** this repository.
2. **Edit** the customization section at the top of the script to insert your API key, user identity, and other settings as needed.
3. **Run** the script:
    ```bash
    python clore_rent_results.py
    ```
4. When executed, the script will:
   - Print a **JSON** summary in the console.
   - Generate an **HTML** report (e.g., `clore_rent_results.html`) with detailed information.
   - Log history to a file (e.g., `clore_history.log`).

## How It Works
- **Data Fetching and Processing:**
  - `get_marketplace_servers_data()`: Retrieves and caches marketplace data from Clore.
  - `process_server_data()` & `process_order_data()`: Calculate metrics such as cost per bk/s and filter servers based on GPU model/watt usage.
- **HTML Report Generation:**
  - `generate_html_summary()`: Creates a table summarizing top marketplace deals.
  - `generate_html_comparison()`: Compares your active orders with cheaper marketplace alternatives.
  - `generate_statistic_section()`: Summarizes daily/monthly costs (optionally converting USD→SEK).
  - `generate_hashpower_graph()`: Produces a line chart using Plotly based on `measurements.json` data (if available).
  - `generate_html_page()`: Combines all sections into a single HTML report.
- **Rent Functionality:**
  - The report includes a **Rent** button that sends a POST request to the Clore API.
  - **Note**: A response with `{"code":0}` indicates the server has been successfully rented.
- **Additional Notes:**
  - **"Actual Hash"** and **"Deviation (%)"** fields are not yet functional and will be fixed soon if there is interest.
  - You can modify `get_sek_rate()` to convert to any local currency.
  - `measurements.json` is optional. If present, the script displays a hashpower graph.
  - History logs are stored in `clore_history.log`.
}
