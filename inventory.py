import pandas as pd
from datetime import datetime, timedelta

# 1. Load the pricing data (sales records)
pricing_data = pd.read_csv('pricingdata.csv')
pricing_data['Date'] = pd.to_datetime(pricing_data['Date'])

# 2. Load the inventory data
# Adjusted to the format you gave (ItemID,InitialInventory)
inventory_data = pd.read_csv('store_inventory.csv')
inventory_data.rename(columns={'InitialInventory': 'Inventory'}, inplace=True)

# 3. Calculate average daily sales per item from all data
avg_sales = pricing_data.groupby('ItemID').agg(
    total_sold=('QuantitySold', 'sum'),
    days_sold=('Date', lambda x: (x.max() - x.min()).days + 1)  # inclusive days
).reset_index()
avg_sales['avg_daily_sales'] = avg_sales['total_sold'] / avg_sales['days_sold']

# 4. Function to restock an item
def restock_item(item_id, restock_amount):
    current_inv = inventory_data.loc[inventory_data['ItemID'] == item_id, 'Inventory'].values[0]
    new_inv = current_inv + restock_amount
    inventory_data.loc[inventory_data['ItemID'] == item_id, 'Inventory'] = new_inv
    print(f"Restocked {restock_amount} units of {item_id}. New inventory: {new_inv}")

# 5. Function to check inventory and schedule restock
def check_and_restock(buffer_days=7, restock_amount=0):
    last_date = pricing_data['Date'].max()

    for _, row in avg_sales.iterrows():
        item = row['ItemID']
        avg_daily_sale = row['avg_daily_sales']
        current_inventory = inventory_data.loc[inventory_data['ItemID'] == item, 'Inventory'].values[0]

        if avg_daily_sale == 0:
            continue  # skip if no sales data

        days_until_runout = current_inventory / avg_daily_sale

        # If we will run out within the buffer period
        if days_until_runout <= buffer_days:
            # Calculate restock day: buffer_days before running out
            restock_day = last_date + timedelta(days=int(days_until_runout) - buffer_days)
            restock_day_str = restock_day.date()

            # Decide restock amount: if not specified, default to 14 days of average sales rounded up
            amount_to_restock = restock_amount if restock_amount > 0 else int(avg_daily_sale * 14 + 0.5)

            print(f"Schedule to restock {item} on {restock_day_str} (one week before stock runs out).")

            # Perform the restock now (or you could store this schedule for later)
            restock_item(item, amount_to_restock)

# Example usage:
check_and_restock(buffer_days=7, restock_amount=100)  # restock 100 units for each item needing it

# To check current inventory after restock
print("\nUpdated Inventory Data:")
print(inventory_data)
