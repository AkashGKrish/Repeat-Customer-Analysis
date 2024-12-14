import pandas as pd
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    """
    Load and prepare dataset for analysis.
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Clean Amount US$ column
    data['Amount US$'] = data['Amount US$'].str.replace(',', '').astype(float)
    
    # Parse dates and sort by transaction time
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data = data.sort_values(by=['customer_id', 'Date'])
    
    # Calculate time gap between consecutive transactions for each customer
    data['Time_Difference'] = data.groupby('customer_id')['Date'].diff().dt.days
    
    # Aggregate customer-level features
    customer_features = data.groupby('customer_id').agg(
        avg_transaction_amt=('Amount US$', 'mean'),
        transaction_frequency=('Time_Difference', 'mean'),
        total_transactions=('Transaction_id', 'count'),
        unique_products=('Product', 'nunique'),
        gender=('Gender', 'first'),
        device=('Device_Type', 'first')
    ).reset_index()
    
    return data, customer_features

def analyze_repeat_customers(customer_features, transaction_data):
    """
    Analyze repeat customers from the prepared dataset.
    """
    # Identify repeat customers
    repeat_customers = customer_features[customer_features['total_transactions'] > 1]
    
    # Demographics Analysis
    demographics = repeat_customers.groupby(['gender', 'device']).size().reset_index(name='count')
    
    # Transaction Behavior
    transaction_behavior = repeat_customers[['avg_transaction_amt', 'transaction_frequency']].describe()
    
    # Product Preferences
    top_products = (
        transaction_data[transaction_data['customer_id'].isin(repeat_customers['customer_id'])]
        .groupby('Category').size().reset_index(name='count').sort_values(by='count', ascending=False)
    )
    
    return repeat_customers, demographics, transaction_behavior, top_products

def plot_gender_distribution(repeat_customers):
    """
    Plot the gender distribution of repeat customers.
    """
    plt.figure(figsize=(8, 6))
    repeat_customers['gender'].value_counts().plot(kind='bar')
    plt.title('Gender Distribution of Repeat Customers')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

def main():
    # Path to the dataset
    file_path = 'test_task_data.csv'  # Replace with your dataset path
    
    # Load and prepare data
    transaction_data, customer_features = load_and_prepare_data(file_path)
    
    # Analyze repeat customers
    repeat_customers, demographics, transaction_behavior, top_products = analyze_repeat_customers(customer_features, transaction_data)
    
    # Report Results
    print("Number of Repeat Customers:", len(repeat_customers))
    print("\nDemographics Summary:")
    print(demographics)
    print("\nTransaction Behavior Summary:")
    print(transaction_behavior)
    print("\nTop Products for Repeat Customers:")
    print(top_products)
    
    # Visualize Insights
    plot_gender_distribution(repeat_customers)

if __name__ == "__main__":
    main()
