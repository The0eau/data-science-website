import numpy as np

class Eclat:
    """
    ECLAT (Equivalence Class Transformation) algorithm for frequent itemset mining.
    
    Parameters
    ----------
    min_support : float, default=0.5
        The minimum threshold for itemset frequency (normalized 0 to 1).
    """
    def __init__(self, min_support=0.5):
        self.min_support = min_support
        self.frequent_itemsets = {}

    def fit(self, transactions):
        """
        Transforms data to vertical format and starts the recursive search.
        """
        n_trans = len(transactions)
        self.min_abs_support = self.min_support * n_trans
        
        # Step 1: Create Tidlists (Vertical Format)
        # {Item: {set of transaction IDs}}
        tidlists = {}
        for idx, trans in enumerate(transactions):
            for item in trans:
                if item not in tidlists:
                    tidlists[item] = set()
                tidlists[item].add(idx)
        
        # Step 2: Filter items that don't meet min_support
        initial_tidlists = {
            frozenset([k]): v for k, v in tidlists.items() 
            if len(v) >= self.min_abs_support
        }
        
        # Step 3: Recursive Depth-First Search
        self._eclat_recursive(initial_tidlists, n_trans)
        return self

    def _eclat_recursive(self, current_tidlists, n_trans):
        """
        Recursive function to find frequent itemsets by intersecting Tidlists.
        """
        items = list(current_tidlists.keys())
        
        for i in range(len(items)):
            item_i = items[i]
            tid_i = current_tidlists[item_i]
            
            # Record frequent itemset
            self.frequent_itemsets[item_i] = len(tid_i) / n_trans
            
            # Try to combine with subsequent items
            next_tier = {}
            for j in range(i + 1, len(items)):
                item_j = items[j]
                tid_j = current_tidlists[item_j]
                
                # The "Magic" of ECLAT: Set Intersection
                intersection = tid_i & tid_j
                
                if len(intersection) >= self.min_abs_support:
                    new_itemset = item_i | item_j
                    next_tier[new_itemset] = intersection
            
            # Recursive call into the next level of the tree
            if next_tier:
                self._eclat_recursive(next_tier, n_trans)

# --- TEST BLOCK ---
if __name__ == "__main__":
    dataset = [
        ['Milk', 'Bread', 'Eggs'],
        ['Milk', 'Bread'],
        ['Milk', 'Eggs'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread', 'Butter', 'Eggs']
    ]

    model = Eclat(min_support=0.4)
    model.fit(dataset)

    print("--- ECLAT Frequent Itemsets ---")
    # Sort by support for readability
    sorted_items = sorted(model.frequent_itemsets.items(), key=lambda x: x[1], reverse=True)
    for itemset, support in sorted_items:
        print(f"{set(itemset)}: Support = {support:.2f}")