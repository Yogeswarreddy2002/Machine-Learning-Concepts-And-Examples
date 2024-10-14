import pandas as pd
from collections import defaultdict

# Sample dataset
transactions = [
    ['A', 'C', 'D'],
    ['B', 'C', 'E'],
    ['A', 'B', 'C', 'E'],
    ['B', 'E']
]

# Step 1: Convert transactions into a dictionary of items and their transaction IDs
def get_item_transaction_dict(transactions):
    item_transaction = defaultdict(set)
    for tid, transaction in enumerate(transactions):
        for item in transaction:
            item_transaction[item].add(tid)
    return item_transaction

# Step 2: Eclat algorithm
def eclat(prefix, items, item_transaction_dict, min_support):
    frequent_itemsets = []
    while items:
        item = items.pop(0)
        new_prefix = prefix + [item]
        support = len(item_transaction_dict[item])

        # Check if support meets the minimum threshold
        if support >= min_support:
            frequent_itemsets.append((new_prefix, support))

            # Get new combinations by intersecting transaction IDs
            new_items = []
            for other_item in items:
                new_transaction_ids = item_transaction_dict[item] & item_transaction_dict[other_item]
                if len(new_transaction_ids) >= min_support:
                    new_items.append(other_item)
                    item_transaction_dict[other_item] = new_transaction_ids
            
            # Recursively find frequent itemsets with new prefix
            frequent_itemsets.extend(eclat(new_prefix, new_items, item_transaction_dict, min_support))
    
    return frequent_itemsets

# Step 3: Execute the Eclat algorithm
min_support = 2  # Set minimum support
item_transaction_dict = get_item_transaction_dict(transactions)
items = list(item_transaction_dict.keys())
frequent_itemsets = eclat([], items, item_transaction_dict, min_support)

# Convert result to DataFrame for better visualization
df_results = pd.DataFrame(frequent_itemsets, columns=['Itemset', 'Support'])
print(df_results)
