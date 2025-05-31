import pandas as pd

# Load the data
df = pd.read_csv('pricingdata.csv', parse_dates=['Date'])

# Compute profit per row
df['Profit'] = df['QuantitySold'] * (df['CostPerItem'] - df['OriginalCostToManufacture'])

# Define date of price change
price_change_date = pd.to_datetime('2025-05-25')

# Split before and after the change
before = df[df['Date'] < price_change_date]
after = df[df['Date'] >= price_change_date]

# Aggregate data
profit_before = before.groupby('ItemID')['Profit'].sum()
profit_after = after.groupby('ItemID')['Profit'].sum()
avg_cost_before = before.groupby('ItemID')['CostPerItem'].mean()
avg_cost_after = after.groupby('ItemID')['CostPerItem'].mean()

# Merge into one summary table
summary = pd.DataFrame({
    'Profit_Before': profit_before,
    'Profit_After': profit_after,
    'AvgCost_Before': avg_cost_before,
    'AvgCost_After': avg_cost_after
}).fillna(0)

# Compute relative derivative
summary['Relative_Derivative'] = (summary['Profit_After'] - summary['Profit_Before']) / summary['AvgCost_Before']

# Function to suggest new price
def adjust_price(row):
    base_price = row['AvgCost_After']
    rel_deriv = row['Relative_Derivative']

    # If derivative is tiny, just average prices (means unclear effect)
    if abs(rel_deriv) < 5:  # small signal
        return round((row['AvgCost_After'] + row['AvgCost_Before']) / 2, 2)

    # Adjust price proportionally — capped within ±10%
    scale = 0.02 * rel_deriv  # scale the derivative to get price change
    new_price = base_price * (1 + scale)
    min_price = 0.90 * base_price
    max_price = 1.10 * base_price
    return round(max(min_price, min(max_price, new_price)), 2)

# Apply the function
summary['Suggested_New_Price'] = summary.apply(adjust_price, axis=1)

# Final columns to inspect (remove Relative_Derivative)
final = summary[['AvgCost_Before', 'AvgCost_After', 'Profit_Before', 'Profit_After', 'Suggested_New_Price']]

# Round all values to two decimal places
final = final.round(2)

print(final)
final.to_csv('suggested_price_changes_relative.csv')
