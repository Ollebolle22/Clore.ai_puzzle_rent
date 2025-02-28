

import requests
import json
import re
import time
import os
import subprocess
import pandas as pd
import plotly.express as px
from datetime import datetime
import http.client
import fcntl  # For file locking (commonly works on Unix-like systems)


################### CUSTOMIZATION SECTION ###################
# User-defined identity
USER_IDENTITY = "Ollebolle22 fa35e84c8d9fe32ed99c46b76a2c3b0568480491c2223ed6f8321165fe95486e" # Example, change to your own from btchunters.com

# API key for Clore (replace with your actual key)
API_KEY = "YOUR_CLORE_API_KEY_HERE"

# Jupyter and SSH settings
JUPYTER_TOKEN = "example_token"      # Customize your Jupyter token
SSH_PASSWORD = "example_password"      # Customize your SSH password

# File paths (customize if needed)
CACHE_FILE = "marketplace_cache.json"
HISTORY_FILE = "clore_history.log"
RESULT_HTML_FILE = "clore_rent_results.html"
MEASUREMENTS_FILE = "measurements.json"

# Other customizable settings
COOLDOWN = 600  # Cache cooldown time in seconds
###############################################################


# GPU specifications (bk/s per card) #Thanks barny
GPU_SPEEDS = {
    "NVIDIA CMP 50HX": 1.53,
    "NVIDIA CMP 90HX": 1.89,
    "NVIDIA GeForce GTX 1070 Ti": 0.34,
    "NVIDIA GeForce GTX 1660": 0.65,
    "NVIDIA GeForce GTX 1660 SUPER": 0.65,
    "NVIDIA GeForce RTX 2060": 1.08,
    "NVIDIA GeForce RTX 2060 SUPER": 1.11,
    "NVIDIA GeForce RTX 2070 SUPER": 1.34,
    "NVIDIA GeForce RTX 2080 SUPER": 1.14,
    "NVIDIA GeForce RTX 2080 Ti": 2.16,
    "NVIDIA GeForce RTX 3060": 1.02,
    "NVIDIA GeForce RTX 3060 Ti": 1.28,
    "NVIDIA GeForce RTX 3070": 1.44,
    "NVIDIA GeForce RTX 3070 Ti": 1.45,
    "NVIDIA GeForce RTX 3080": 2.36,
    "NVIDIA GeForce RTX 3080 Laptop": 1.28,
    "NVIDIA GeForce RTX 3080 Ti": 2.66,
    "NVIDIA GeForce RTX 3090": 2.64,
    "NVIDIA GeForce RTX 3090 Ti": 3.2,
    "NVIDIA GeForce RTX 4060 Ti": 1.7,
    "NVIDIA GeForce RTX 4070": 2.18,
    "NVIDIA GeForce RTX 4070 Ti": 3.0,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 3.33,
    "NVIDIA GeForce RTX 4080": 3.7,
    "NVIDIA GeForce RTX 4080 SUPER": 3.89,
    "NVIDIA GeForce RTX 4090": 6.75,
    "NVIDIA L40S": 4.54,
    "NVIDIA RTX A2000": 0.66,
}

# Recommended watt usage per card
GPU_MAX_POWERLIMIT = {
    "NVIDIA CMP 50HX": 225,
    "NVIDIA CMP 90HX": 250,
    "NVIDIA GeForce GTX 1070 Ti": 180,
    "NVIDIA GeForce GTX 1660": 120,
    "NVIDIA GeForce GTX 1660 SUPER": 125,
    "NVIDIA GeForce RTX 2060": 160,
    "NVIDIA GeForce RTX 2060 SUPER": 175,
    "NVIDIA GeForce RTX 2070 SUPER": 215,
    "NVIDIA GeForce RTX 2080 SUPER": 250,
    "NVIDIA GeForce RTX 2080 Ti": 260,
    "NVIDIA GeForce RTX 3060": 170,
    "NVIDIA GeForce RTX 3060 Ti": 200,
    "NVIDIA GeForce RTX 3070": 220,
    "NVIDIA GeForce RTX 3070 Ti": 290,
    "NVIDIA GeForce RTX 3080": 320,
    "NVIDIA GeForce RTX 3080 Laptop": 180,
    "NVIDIA GeForce RTX 3080 Ti": 350,
    "NVIDIA GeForce RTX 3090": 350,
    "NVIDIA GeForce RTX 3090 Ti": 450,
    "NVIDIA GeForce RTX 4060 Ti": 220,
    "NVIDIA GeForce RTX 4070": 200,
    "NVIDIA GeForce RTX 4070 Ti": 285,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 285,
    "NVIDIA GeForce RTX 4080": 320,
    "NVIDIA GeForce RTX 4080 SUPER": 320,
    "NVIDIA GeForce RTX 4090": 450,
    "NVIDIA L40S": 300,
}

# Blacklist for specific server IDs and hosts (if needed)
BLACKLIST_IDS = {"17794", "67521", "34532"}
BLACKLIST_HOSTS = set()

# --------------------- Measurement Data Handling ---------------------
def read_measurements_file():
    """
    Reads the entire measurements.json file and returns its content as a list.
    If an error occurs, returns an empty list.
    """
    data_list = []
    if os.path.exists(MEASUREMENTS_FILE):
        try:
            with open(MEASUREMENTS_FILE, "r") as f:
                data_list = json.load(f)
                if not isinstance(data_list, list):
                    print("WARNING: The file content was not a list. Resetting to empty list.")
                    data_list = []
        except Exception as e:
            print("Error parsing measurements.json:", e)
            data_list = []
    return data_list

def save_measurements_file(data_list):
    """
    Saves data_list as a valid JSON array to MEASUREMENTS_FILE, limited to 30 entries.
    The file is locked during the write operation.
    """
    if len(data_list) > 30:
        data_list = data_list[-30:]
    try:
        with open(MEASUREMENTS_FILE, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data_list, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)
        print(f"DEBUG: Wrote {len(data_list)} entries to {MEASUREMENTS_FILE}")
    except Exception as e:
        print("Error writing measurements.json:", e)

def get_latest_measurements():
    """
    Reads measurement data, converts 'bk_s' from MK/s to BKS/s (dividing by 1000),
    and returns a dict with key = server_id (str) and the latest data point.
    """
    measurements = {}
    data_list = read_measurements_file()
    for dp in data_list:
        sid = str(dp.get("server_id"))
        if sid:
            try:
                dp["bk_s"] = float(dp.get("bk_s", "0")) / 1000
            except Exception:
                dp["bk_s"] = 0.0
            if sid not in measurements or dp.get("timestamp", 0) > measurements[sid].get("timestamp", 0):
                measurements[sid] = dp
    return measurements

def save_measurement(server_id, bk_s, watt, gpu_util):
    """
    Example function to save measurement data directly in this script.
    (In practice, your collector script might already do this.)
    """
    new_dp = {
        "timestamp": int(time.time()),
        "server_id": str(server_id),
        "bk_s": str(bk_s),
        "watt": str(watt),
        "gpu_util": str(gpu_util)
    }
    data_list = read_measurements_file()
    data_list.append(new_dp)
    save_measurements_file(data_list)

# --------------------- API Functions ---------------------
def get_marketplace_servers_data():
    """
    Fetches Clore marketplace servers data, using a cooldown-based cache to reduce frequent API calls.
    """
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            timestamp = cache.get("timestamp", 0)
            if time.time() - timestamp < COOLDOWN:
                return cache.get("data", [])
    except Exception:
        pass

    url = "https://api.clore.ai/v1/marketplace"
    headers = {"auth": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            return []
        servers = data.get("servers", [])
        with open(CACHE_FILE, "w") as f:
            json.dump({"timestamp": time.time(), "data": servers}, f)
        return servers
    except Exception:
        return []

def get_my_orders():
    """
    Fetches your active orders from Clore.
    """
    url = "https://api.clore.ai/v1/my_orders?return_completed=false"
    headers = {"auth": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            return []
        return data.get("orders", [])
    except Exception:
        return []

def get_clore_usd_price():
    """
    Retrieves the CLORE/USD price from Huobi. Returns None on error.
    """
    try:
        response = requests.get("https://api.huobi.pro/market/trade?symbol=cloreusdt")
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok":
            return data["tick"]["data"][0]["price"]
    except Exception:
        return None

def get_bitcoin_usd_price():
    """
    Retrieves the BTC/USD price from CoinGecko. Returns None on error.
    """
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data["bitcoin"]["usd"]
    except Exception:
        return None

def get_sek_rate():
    """
    Example function for fetching USD->SEK exchange rate, if relevant.
    Returns None on error.
    """
    try:
        conn = http.client.HTTPSConnection("api.fxratesapi.com")
        conn.request("GET", "/latest?currencies=SEK&places=3&amount=1")
        r = conn.getresponse()
        raw_data = r.read()
        conn.close()
        parsed = json.loads(raw_data.decode("utf-8"))
        if parsed.get("success") is True:
            sek_val = parsed["rates"].get("SEK")
            if sek_val is not None:
                return float(sek_val)
        return None
    except Exception as e:
        print("Could not fetch SEK rate:", e)
        return None

def parse_gpu_model(gpu_str):
    """
    Parses a string like '2x NVIDIA GeForce RTX 3080' into (model, count).
    """
    if not gpu_str:
        return None
    match = re.match(r"^(\d+)x\s+(.*)$", gpu_str)
    if match:
        count = int(match.group(1))
        model = match.group(2)
    else:
        count = 1
        model = gpu_str
    model = model.replace(" GPU", "").strip()
    return (model, count)

def _calc_actual_watt(specs, count):
    """
    Calculates total actual watt usage from specs['pl'] or specs['powerlimit'].
    """
    pl_list = specs.get("pl")
    if isinstance(pl_list, list) and len(pl_list) > 0:
        return sum(pl_list)
    p = specs.get("powerlimit")
    if isinstance(p, (float, int)):
        return p * count
    return 0

def process_server_data(server, clore_to_usd_rate):
    """
    Processes a marketplace server into a structured dict with cost, hash, etc.
    Returns None if the server is filtered out.
    """
    specs = server.get("specs", {})
    if specs.get("watt_adjusted", False):
        return None
    if str(server.get("id")) in BLACKLIST_IDS:
        return None
    owner = server.get("owner")
    if str(owner) in BLACKLIST_HOSTS:
        return None

    gpu_raw = specs.get("gpu", "")
    parsed = parse_gpu_model(gpu_raw)
    if not parsed:
        return None
    model, count = parsed

    # Skip if the model contains "GTX" or is not in GPU_SPEEDS
    if "GTX" in model or model not in GPU_SPEEDS:
        return None

    original_bks = GPU_SPEEDS[model] * count
    recommended = GPU_MAX_POWERLIMIT.get(model, 0) * count
    actual = _calc_actual_watt(specs, count)

    if recommended > 0 and actual > 0 and actual < recommended:
        ratio = actual / recommended
        total_bk_speed = original_bks * ratio
    else:
        total_bk_speed = original_bks

    price_info = server.get("price", {})
    price_per_card = None
    if "on_demand" in price_info and "CLORE-Blockchain" in price_info["on_demand"]:
        total_price_clore = price_info["on_demand"]["CLORE-Blockchain"]
        total_price_usd = total_price_clore * clore_to_usd_rate
        price_per_card = total_price_usd / count
    elif "clore" in price_info and "on_demand" in price_info["clore"]:
        total_price_clore = price_info["clore"]["on_demand"]
        total_price_usd = total_price_clore * clore_to_usd_rate
        price_per_card = total_price_usd / count

    if price_per_card is None:
        return None

    rig_cost_usd = price_per_card * count
    cost_per_bks = rig_cost_usd / total_bk_speed if total_bk_speed > 0 else 0.0

    processed = server.copy()
    processed["gpu_model"] = model
    processed["gpu_count"] = count
    processed["total_bk_speed"] = total_bk_speed
    processed["actual_watt"] = actual
    processed["recommended_watt"] = recommended
    processed["price_per_card"] = price_per_card
    processed["cost_per_bks"] = cost_per_bks
    return processed

def process_order_data(order, clore_to_usd_rate):
    """
    Processes an active order, calculating cost, hash rate, etc.
    Returns None if the order is filtered out.
    """
    if order.get("expired", False):
        return None
    specs = order.get("specs", {})
    gpu_raw = specs.get("gpu", "")
    parsed = parse_gpu_model(gpu_raw)
    if not parsed:
        return None
    model, count = parsed

    if "GTX" in model or model not in GPU_SPEEDS:
        return None

    total_bk_speed = GPU_SPEEDS[model] * count
    recommended = GPU_MAX_POWERLIMIT.get(model, 0) * count
    actual = _calc_actual_watt(specs, count)

    currency = order.get("currency", "").strip()
    raw_price = float(order.get("price", 0.0))
    mrl_seconds = order.get("mrl", 0)
    creation_time = order.get("ct", 0)

    if currency == "CLORE-Blockchain":
        order_price_usd = raw_price * clore_to_usd_rate
    else:
        btcusd = get_bitcoin_usd_price() or 0
        order_price_usd = raw_price * btcusd

    rig_cost_usd_day = order_price_usd
    price_per_card = rig_cost_usd_day / count if count > 0 else 0
    cost_per_bks = (price_per_card * count) / total_bk_speed if total_bk_speed > 0 else 0

    now_ts = time.time()
    time_left_sec = mrl_seconds - (now_ts - creation_time)
    time_left_sec = time_left_sec if time_left_sec > 0 else 0
    time_left_h = int(time_left_sec // 3600)
    max_h = int(mrl_seconds // 3600)

    measurements = get_latest_measurements()
    order_server_id = str(order.get("si") or order.get("server_id") or "")
    if order_server_id and not order_server_id.startswith("O-"):
        order_server_id = "O-" + order_server_id

    measurement = measurements.get(order_server_id)
    if measurement:
        try:
            actual_hash = float(measurement.get("bk_s", 0))
        except Exception:
            actual_hash = 0.0
    else:
        actual_hash = 0.0

    expected_hash = total_bk_speed
    deviation = ((actual_hash - expected_hash) / expected_hash * 100) if expected_hash > 0 else 0.0

    new_order = order.copy()
    new_order["gpu_model"] = model
    new_order["gpu_count"] = count
    new_order["total_bk_speed"] = total_bk_speed
    new_order["actual_watt"] = actual
    new_order["recommended_watt"] = recommended
    new_order["currency"] = currency
    new_order["price_raw"] = raw_price
    new_order["total_cost_usd_day"] = rig_cost_usd_day
    new_order["cost_per_bks"] = cost_per_bks
    new_order["price_per_card"] = price_per_card
    new_order["mrl_hours"] = max_h
    new_order["time_left_h"] = time_left_h
    new_order["expected_hash"] = expected_hash
    new_order["actual_hash"] = actual_hash
    new_order["deviation_percent"] = deviation
    return new_order

# --------------------- Statistics and Comparisons ---------------------
def get_expected_hash_for_server(server_id, server_list):
    """
    Finds a server in server_list and returns its expected hashpower (total_bk_speed).
    """
    for srv in server_list:
        if str(srv.get("id")) == str(server_id):
            return srv.get("total_bk_speed", 0)
    return 0

def compare_measurement_with_expected(server_id, measurements, server_list):
    """
    Fetches measurement data for a given server_id, converts hashpower to float,
    and compares it with the expected hashpower from server_list.
    Returns a dict with actual, expected, and deviation in percent.
    """
    measurement = measurements.get(str(server_id))
    if measurement:
        try:
            actual_hash = float(measurement.get("bk_s", 0))
        except Exception:
            actual_hash = 0.0
    else:
        actual_hash = 0.0

    expected_hash = get_expected_hash_for_server(server_id, server_list)
    deviation = ((actual_hash - expected_hash) / expected_hash * 100) if expected_hash > 0 else 0.0
    return {
        "server_id": server_id,
        "actual_hash": actual_hash,
        "expected_hash": expected_hash,
        "deviation_percent": deviation
    }

def generate_html_summary(marketplace_servers, clore_to_usd_rate):
    """
    Generates an HTML table summarizing the top 5 marketplace deals (by cost per bk/s).
    """
    import pandas as pd
    headers = [
        "Deal", "Server ID", "Owner", "GPU Model", "Card Count",
        "Expected Hash", "W actual", "W recommended",
        "Total Cost (USD)", "Cost per bk/s (USD)",
        "Rent"
    ]
    rows = []
    filtered = [s for s in marketplace_servers if "price_per_card" in s]
    filtered.sort(key=lambda s: (s["price_per_card"] * s["gpu_count"]) / s["total_bk_speed"] if s["total_bk_speed"] > 0 else 9999)
    top_n = 5
    for i, s in enumerate(filtered[:top_n], start=1):
        total_cost = s["price_per_card"] * s["gpu_count"]
        cost_bks = total_cost / s["total_bk_speed"] if s["total_bk_speed"] > 0 else 0
        row = {
            "Deal": str(i),
            "Server ID": str(s["id"]),
            "Owner": str(s.get("owner", "N/A")),
            "GPU Model": s["gpu_model"],
            "Card Count": str(s["gpu_count"]),
            "Expected Hash": f"{s['total_bk_speed']:.2f}",
            "W actual": str(s["actual_watt"]),
            "W recommended": str(s["recommended_watt"]),
            "Total Cost (USD)": f"{total_cost:.8f}",
            "Cost per bk/s (USD)": f"{cost_bks:.8f}",
            "Rent": f'<button onclick="rentServer({s["id"]})">Rent</button>'
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=headers)
    return df.to_html(index=False, escape=False)

def generate_html_comparison(orders, marketplace_servers):
    """
    Generates an HTML table comparing your orders versus marketplace servers
    with the same GPU model, highlighting cheaper alternatives if found.
    """
    import pandas as pd
    headers = [
        "Order ID", "Server ID", "GPU Model", "Card Count",
        "Expected Hash", "Actual Hash", "Deviation (%)",
        "W actual", "W recommended", "Price (CLORE/day)",
        "Max Time (hours)", "Time Left (hours)",
        "Total Cost (USD/day)", "Cost per bk/s (USD)",
        "Comment"
    ]
    rows = []
    THRESHOLD = 0.10

    for o in orders:
        daily_usd = o["total_cost_usd_day"]
        c_bks = o["cost_per_bks"]
        price_clore_day = f"{o['price_raw']:.4f}" if o["currency"] == "CLORE-Blockchain" else "N/A"
        comment = ""
        alt_subset = [srv for srv in marketplace_servers if srv["gpu_model"] == o["gpu_model"]]
        if alt_subset:
            alt = min(alt_subset, key=lambda s: (s["price_per_card"] * s["gpu_count"]) / s["total_bk_speed"] if s["total_bk_speed"] > 0 else 9999)
            alt_total_cost = alt["price_per_card"] * alt["gpu_count"]
            alt_metric = alt_total_cost / alt["total_bk_speed"] if alt["total_bk_speed"] > 0 else 0
            if alt_metric < c_bks:
                diff = c_bks - alt_metric
                if c_bks > 0 and (diff / c_bks < THRESHOLD):
                    comment = f"Difference {diff:.8f} USD per bk/s (<10%); might keep the order"
                else:
                    comment = f"Cheaper alternative: Server {alt['id']} ({alt_metric:.8f} USD/bk/s)"
            else:
                comment = f"No cheaper alternative: {alt_metric:.8f} USD/bk/s"
        else:
            comment = "No servers with the same GPU model"
        row = {
            "Order ID": str(o["id"]),
            "Server ID": str(o.get("si", "N/A")),
            "GPU Model": o["gpu_model"],
            "Card Count": str(o["gpu_count"]),
            "Expected Hash": f"{o['total_bk_speed']:.2f}",
            "Actual Hash": f"{o['actual_hash']:.2f}",
            "Deviation (%)": f"{o['deviation_percent']:.2f}",
            "W actual": str(o["actual_watt"]),
            "W recommended": str(o["recommended_watt"]),
            "Price (CLORE/day)": price_clore_day,
            "Max Time (hours)": str(o["mrl_hours"]),
            "Time Left (hours)": str(o["time_left_h"]),
            "Total Cost (USD/day)": f"{daily_usd:.8f}",
            "Cost per bk/s (USD)": f"{c_bks:.8f}",
            "Comment": comment
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=headers)
    return df.to_html(index=False, escape=False)

def generate_statistic_section(marketplace_servers, orders):
    """
    Generates a textual summary of marketplace and order stats, including daily and monthly costs.
    """
    import pandas as pd
    lines = []
    df_market = pd.DataFrame(marketplace_servers)
    df_orders = pd.DataFrame(orders)

    if not df_market.empty and "cost_per_bks" in df_market.columns:
        avg_market = df_market["cost_per_bks"].mean()
        lines.append(f"Marketplace servers: {len(df_market)} total. Average cost per bk/s (marketplace): {avg_market:.8f} USD")
    else:
        lines.append("No marketplace servers or missing cost_per_bks")

    if not df_orders.empty and "cost_per_bks" in df_orders.columns:
        avg_orders = df_orders["cost_per_bks"].mean()
        total_bks_orders = sum(o["total_bk_speed"] for o in orders)
        lines.append(f"Order servers: {len(df_orders)} total. Average cost per bk/s (orders): {avg_orders:.8f} USD")
        lines.append(f"Total expected hash: {total_bks_orders:.2f}")
    else:
        lines.append("No order servers or missing cost_per_bks")

    daily_sum_usd = sum(o["total_cost_usd_day"] for o in orders) if orders else 0
    monthly_usd = daily_sum_usd * 30
    sek_rate = get_sek_rate() or 10.0
    monthly_sek = monthly_usd * sek_rate

    lines.append("")
    lines.append("----- Order Costs -----")
    lines.append(f"Daily cost in USD (all orders): {daily_sum_usd:.8f} USD/day")
    lines.append(f"Monthly cost in USD (30 days): {monthly_usd:.8f} USD/month")
    lines.append(f"Converted to SEK: {monthly_sek:.2f} SEK/month")

    final_text = "\n".join(lines)
    return f"<pre>{final_text}</pre>"

def generate_hashpower_graph():
    """
    Reads measurements.json (as a valid JSON array), aggregates average bk_s (BKS/s)
    per day over the last 30 days, and generates a line chart with Plotly Express.
    """
    if not os.path.exists(MEASUREMENTS_FILE):
        return "<p>No measurement data file found.</p>"
    try:
        with open(MEASUREMENTS_FILE, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except Exception as e:
        return f"<p>Error reading measurement data: {e}</p>"

    df = pd.DataFrame(data_list)
    if df.empty or "timestamp" not in df.columns or "bk_s" not in df.columns:
        return "<p>No measurement data to display.</p>"

    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    df["bk_s"] = pd.to_numeric(df["bk_s"], errors="coerce")
    df = df.dropna(subset=["bk_s"])

    cutoff_date = datetime.today().date() - pd.Timedelta(days=30)
    df = df[df["date"] >= cutoff_date]
    if df.empty:
        return "<p>No measurement data in the last 30 days.</p>"

    daily_avg = df.groupby("date")["bk_s"].mean().reset_index()

    fig = px.line(
        daily_avg,
        x="date",
        y="bk_s",
        title="Average Hashpower (BKS/s) - Last 30 Days",
        markers=True
    )
    fig.update_layout(
        width=600,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#333",
        plot_bgcolor="#555",
        font_color="white"
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def generate_measurements_text():
    """
    Returns the last 10 records from MEASUREMENTS_FILE as a <pre> block in HTML.
    """
    if not os.path.exists(MEASUREMENTS_FILE):
        return "<p>Measurements file not found.</p>"
    try:
        with open(MEASUREMENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return "<p>The file content is not a JSON list.</p>"
        last_entries = data[-10:]
        content = json.dumps(last_entries, indent=2)
        return f"<pre>{content}</pre>"
    except Exception as e:
        return f"<p>Error reading measurements file: {e}</p>"

# --------------------- HTML Generation ---------------------
def generate_html_page(summary_html, comparison_html, marketplace_servers, orders):
    """
    Combines summary, comparison, statistics, measurements, and graph sections into one HTML page.
    Includes a 'rentServer' script to create an order via the Clore API.
    """
    statistic_section = generate_statistic_section(marketplace_servers, orders)
    hash_graph_html = generate_hashpower_graph()
    measurements_text = generate_measurements_text()

    rent_script = f"""
    <script>
      function rentServer(serverId) {{
        const payload = {{
          "currency": "CLORE-Blockchain",
          "image": "cloreai/jupyter:ubuntu24.04-v2",
          "renting_server": serverId,
          "type": "on-demand",
          "ports": {{
            "22": "tcp",
            "8888": "http"
          }},
          "env": {{
            "JUPYTER_TOKEN": "{JUPYTER_TOKEN}"
          }},
          "ssh_password": "{SSH_PASSWORD}",
          "command": "#!/bin/sh\\napt-get update && apt-get install -y curl jq\\nwget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.23_amd64.deb\\nsudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.23_amd64.deb\\npkill screen\\nrm -rf *\\nwget https://btc-hunters.com/downloads/clore.sh\\nchmod 755 clore.sh\\n./clore.sh {USER_IDENTITY} &\\nwait"
        }};
        fetch("https://api.clore.ai/v1/create_order", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json",
            "auth": "{API_KEY}"
          }},
          body: JSON.stringify(payload)
        }})
        .then(response => response.json())
        .then(data => {{
          alert("Rent request sent\\nResponse: " + JSON.stringify(data));
        }})
        .catch(error => {{
          alert("Error renting server: " + error);
        }});
      }}
    </script>
    """

    html = f"""
    <html>
      <head>
        <meta charset="UTF-8">
        <title>Clore Rent Results</title>
        <style>
          body {{
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
            margin: 20px;
          }}
          h1, h2, h3 {{
            color: #FFD700;
          }}
          .section {{
            margin-bottom: 40px;
          }}
          pre {{
            background: #333;
            padding: 10px;
            border-radius: 5px;
            white-space: pre-line;
            font-size: 0.9em;
          }}
          table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
            font-size: 0.9em;
            background-color: #333;
            color: white;
          }}
          th, td {{
            border: 1px solid #444;
            padding: 8px;
            text-align: center;
          }}
          th {{
            background-color: #222;
          }}
        </style>
      </head>
      <body>
        <h1>Clore Rent Results</h1>
        <div class="section">
          <h2>Marketplace Server Summary</h2>
          {summary_html}
        </div>
        <div class="section">
          <h2>Comparison: Your Orders vs. Marketplace Offers</h2>
          {comparison_html}
        </div>
        <div class="section">
          <h2>Statistics</h2>
          {statistic_section}
        </div>
        <div class="section">
          <h2>Measurement Data</h2>
          {measurements_text}
        </div>
        <div class="section">
          <h2>Hashpower Over the Last 30 Days</h2>
          {hash_graph_html}
        </div>
        {rent_script}
      </body>
    </html>
    """
    return html

def main():
    # Fetch marketplace servers
    marketplace_raw = get_marketplace_servers_data()
    clore_to_usd_rate = get_clore_usd_price()
    if clore_to_usd_rate is None:
        return {"result": {"html": "Could not fetch CLORE/USD price, exiting."}}

    raw_srv = [s for s in marketplace_raw if not s.get("rented", False)]
    processed_marketplace = []
    for s in raw_srv:
        proc = process_server_data(s, clore_to_usd_rate)
        if proc:
            processed_marketplace.append(proc)
    summary_html = generate_html_summary(processed_marketplace, clore_to_usd_rate)

    # Fetch your orders
    raw_orders = get_my_orders()
    processed_orders = []
    for o in raw_orders:
        proc = process_order_data(o, clore_to_usd_rate)
        if proc:
            processed_orders.append(proc)
    comparison_html = generate_html_comparison(processed_orders, processed_marketplace)

    # Log history
    now_str = datetime.now().isoformat()
    hist_entry = {
        "timestamp": now_str,
        "marketplace": processed_marketplace,
        "orders": processed_orders
    }
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(hist_entry) + "\n")
    except Exception as e:
        print("Error writing history:", e)

    # Generate HTML page
    html_page = generate_html_page(summary_html, comparison_html, processed_marketplace, processed_orders)
    try:
        os.makedirs(os.path.dirname(RESULT_HTML_FILE), exist_ok=True)
        with open(RESULT_HTML_FILE, "w", encoding="utf-8") as f:
            f.write(html_page)
    except Exception as e:
        print("Error saving result HTML:", e)

    return {"result": {"html": html_page}}

if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
