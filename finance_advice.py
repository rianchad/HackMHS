from datetime import date, timedelta
from typing import List
from collections import defaultdict
import pandas as pd
import os
import json

# -------------------- Constants --------------------
profit_margin = float(os.environ.get("PROFIT_MARGIN", 0.30))
money_starting = float(os.environ.get("MONEY_STARTING", 100000))
interest_rate = float(os.environ.get("INTEREST_RATE", 0.03))
loan_term_days = int(os.environ.get("LOAN_TERM_DAYS", 180))
min_cash_reserve_ratio = float(os.environ.get("MIN_CASH_RESERVE_RATIO", 0.20))

# -------------------- Expense Class --------------------
class Expence:
    def __init__(self, name: str, cost: float, due_date: date, other_possible_dates: List[date], urgency: int):
        self.name = name
        self.cost = cost
        self.due_date = due_date
        self.other_possible_dates = other_possible_dates
        self.urgency = urgency

    def __repr__(self):
        return f"Expence(name={self.name},cost={self.cost}, due_date={self.due_date}, urgency={self.urgency})"

# -------------------- Loan Class --------------------
class Loan:
    def __init__(self, amount: float, taken_on: date, interest_rate: float, term_days: int):
        self.amount = amount
        self.taken_on = taken_on
        self.interest_rate = interest_rate
        self.term_days = term_days
        self.due_date = taken_on + timedelta(days=term_days)
        self.total_due = amount * (1 + interest_rate)

    def __repr__(self):
        return (f"Loan(amount={self.amount:.2f}, taken_on={self.taken_on}, due_date={self.due_date}, "
                f"total_due={self.total_due:.2f})")

# -------------------- Expenses --------------------
expenses = []
expenses_json = os.environ.get("EXPENSES_JSON")
if expenses_json:
    try:
        loaded = json.loads(expenses_json)
        for exp in loaded:
            # exp: {name, cost, due_date, other_possible_dates, urgency}
            expenses.append(
                Expence(
                    name=exp.get("name", ""),
                    cost=float(exp.get("cost", 0)),
                    due_date=date.fromisoformat(exp.get("due_date")),
                    other_possible_dates=[date.fromisoformat(d) for d in exp.get("other_possible_dates", [])],
                    urgency=int(exp.get("urgency", 0))
                )
            )
    except Exception as e:
        print("Failed to parse EXPENSES_JSON:", e)
if not expenses:
    expenses = [
    ]

# -------------------- Load and Process Revenue Data --------------------
df_original = pd.read_csv("predicted_revenue.csv")

# Ensure date column is datetime.date type
df_original["date"] = pd.to_datetime(df_original["date"]).dt.date

# Calculate profit estimate
df_original["profit_estimate"] = df_original["predicted_revenue"] * profit_margin

# Make a copy for revision
df_revised = df_original.copy()

# -------------------- Cashflow Simulation --------------------
current_money = money_starting
expected_totals = []
revised_expenses = []
loans = []
paid_expenses = set()

# Expand expenses with delay options
adjusted_expenses = []
for exp in expenses:
    for delay in range(exp.urgency + 1):
        possible_date = exp.due_date + timedelta(days=delay)
        adjusted_expenses.append((possible_date, exp.cost, exp.urgency, exp.due_date))
adjusted_expenses.sort()

# Group expenses by date
daily_expenses = defaultdict(list)
for due_date, cost, urgency, original_due in adjusted_expenses:
    daily_expenses[due_date].append((cost, urgency, original_due))

# Track loans due by date
loans_due = defaultdict(list)

for i, row in df_revised.iterrows():
    current_day = row["date"]
    profit_today = row["profit_estimate"]
    current_money += profit_today

    # Pay back loans due today
    if current_day in loans_due:
        for loan in loans_due[current_day]:
            if current_money >= loan.total_due:
                current_money -= loan.total_due
                revised_expenses.append((current_day, 0, loan.total_due, "Loan Repaid"))
            else:
                # Pay as much as possible if insufficient funds
                revised_expenses.append((current_day, 0, current_money, "Partial Loan Repaid"))
                current_money = 0
        del loans_due[current_day]

    # Pay expenses due today
    if current_day in daily_expenses:
        for cost, urgency, original_due in daily_expenses[current_day]:
            if (original_due, cost) in paid_expenses:
                continue

            # Calculate minimum cash reserve required after paying expense
            min_cash_after_payment = current_money * min_cash_reserve_ratio
            total_needed = cost + min_cash_after_payment

            if current_money >= total_needed:
                # Pay expense while keeping reserve
                current_money -= cost
                paid_expenses.add((original_due, cost))
                revised_expenses.append((original_due, current_day, cost, "Paid"))
            else:
                # Take loan to cover shortfall + keep reserve
                shortfall = total_needed - current_money
                loan = Loan(amount=shortfall, taken_on=current_day, interest_rate=interest_rate, term_days=loan_term_days)
                current_money += loan.amount  # receive loan money
                current_money -= cost         # pay expense
                paid_expenses.add((original_due, cost))
                loans.append(loan)
                loans_due[loan.due_date].append(loan)
                revised_expenses.append((original_due, current_day, cost, "Loan"))

    expected_totals.append(current_money)

df_revised["expected_total_money"] = expected_totals

# -------------------- Output --------------------
print("ORIGINAL DATAFRAME:")
print(df_original.head(10))

print("\nREVISED DATAFRAME WITH CASHFLOW:")
print(df_revised.head(10))

print("\nEXPENSE PAYMENTS & LOANS:")
for event in revised_expenses:
    # Round floats in event tuple
    event_rounded = tuple(round(x, 2) if isinstance(x, float) else x for x in event)
    print(event_rounded)

print("\nLOANS TAKEN:")
for loan in loans:
    print(
        f"Loan(amount=${loan.amount:,.2f}, taken_on={loan.taken_on}, due_date={loan.due_date}, total_due=${loan.total_due:,.2f})"
    )

# Write a summary advice file for the frontend (cleaner output, rounded values, formatted for HTML)
summary_lines = []
summary_lines.append("<b>Cashflow Simulation Summary</b><br>")
summary_lines.append(f"<b>Starting Money:</b> ${money_starting:,.2f}<br>")
summary_lines.append(f"<b>Profit Margin:</b> {round(profit_margin*100, 2):.2f}%<br>")
summary_lines.append(f"<b>Interest Rate:</b> {round(interest_rate*100, 2):.2f}%<br>")
summary_lines.append(f"<b>Loan Term:</b> {loan_term_days} days<br>")
summary_lines.append(f"<b>Min Cash Reserve Ratio:</b> {round(min_cash_reserve_ratio*100, 2):.2f}%<br><br>")

summary_lines.append("<b>Recent Expense Payments & Loans:</b><ul>")
for event in revised_expenses[-10:]:
    event_rounded = tuple(round(x, 2) if isinstance(x, float) else x for x in event)
    # Format: (original_due, current_day, cost, status)
    summary_lines.append(
        f"<li><b>{event_rounded[3]}</b> | Due: {event_rounded[0]} | Paid: {event_rounded[1]} | Amount: ${event_rounded[2]:,.2f}</li>"
        if len(event_rounded) == 4 else f"<li>{event_rounded}</li>"
    )
summary_lines.append("</ul>")

summary_lines.append("<b>Recent Loans Taken:</b>")
summary_lines.append("""
<table style="border-collapse:collapse;">
<tr>
  <th style="border:1px solid #ccc;padding:4px;">Amount</th>
  <th style="border:1px solid #ccc;padding:4px;">Taken On</th>
  <th style="border:1px solid #ccc;padding:4px;">Due Date</th>
  <th style="border:1px solid #ccc;padding:4px;">Total Due</th>
</tr>
""")
for loan in loans[-5:]:
    summary_lines.append(
        f"<tr>"
        f"<td style='border:1px solid #ccc;padding:4px;'>${loan.amount:,.2f}</td>"
        f"<td style='border:1px solid #ccc;padding:4px;'>{loan.taken_on}</td>"
        f"<td style='border:1px solid #ccc;padding:4px;'>{loan.due_date}</td>"
        f"<td style='border:1px solid #ccc;padding:4px;'>${loan.total_due:,.2f}</td>"
        f"</tr>"
    )
summary_lines.append("</table>")

summary_lines.append(f"<b>Final Expected Money:</b> ${df_revised['expected_total_money'].iloc[-1]:,.2f}")

with open("advice_summary.txt", "w") as f:
    f.write('\n'.join(summary_lines))

# Save df_revised for download
df_revised = df_revised.round(2)
df_revised.to_csv("df_revised.csv", index=False)
