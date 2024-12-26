import pandas as pd
import datetime as dt

# Example data
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'TransactionDate': ['2024-06-01', '2024-06-25', '2024-06-15', '2024-05-01', '2024-06-02'],
    'PurchaseAmount': [100, 200, 150, 50, 300]
}
df = pd.DataFrame(data)
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# Reference date (today)
today = dt.datetime(2024, 6, 30)

# RFM Calculation
rfm = df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (today - x.max()).days,  # Recency
    'CustomerID': 'count',                               # Frequency
    'PurchaseAmount': 'sum'                              # Monetary
})

rfm.rename(columns={
    'TransactionDate': 'Recency',
    'CustomerID': 'Frequency',
    'PurchaseAmount': 'Monetary'
}, inplace=True)

# Assign scores (1–5 based on quantiles)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine scores
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print(rfm)
