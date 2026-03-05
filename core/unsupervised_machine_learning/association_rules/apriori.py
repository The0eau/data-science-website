from itertools import combinations, chain

class Apriori:
    """
    Apriori algorithm for Association Rule Mining from scratch.
    
    Parameters
    ----------
    min_support : float, default=0.5
        The minimum threshold for itemset frequency (0 to 1).
    min_confidence : float, default=0.7
        The minimum threshold for rule reliability (0 to 1).
    """
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {} # Stores {itemset: support}
        self.rules = []

    def _get_support(self, transactions, itemset):
        """Calculates the frequency of an itemset in the dataset."""
        count = 0
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        return count / len(transactions)

    def fit(self, transactions):
        """
        Executes the Apriori algorithm on a list of transactions.
        """
        n_trans = len(transactions)
        # Step 1: Find frequent 1-itemsets
        all_items = set(chain(*transactions))
        current_candidates = [frozenset([item]) for item in all_items]
        
        k = 1
        while current_candidates:
            # Filter candidates by support
            frequent_k = {}
            for itemset in current_candidates:
                support = self._get_support(transactions, itemset)
                if support >= self.min_support:
                    frequent_k[itemset] = support
            
            if not frequent_k:
                break
                
            self.frequent_itemsets.update(frequent_k)
            
            # Step 2: Generate k+1 candidates from frequent k-itemsets
            current_candidates = self._generate_next_candidates(list(frequent_k.keys()), k)
            k += 1
        
        self._generate_rules()
        return self

    def _generate_next_candidates(self, prev_frequent, k):
        """Joins frequent k-itemsets to create (k+1)-candidates."""
        new_candidates = set()
        for i in range(len(prev_frequent)):
            for j in range(i + 1, len(prev_frequent)):
                # Join if the first k-1 items are identical
                l1, l2 = list(prev_frequent[i]), list(prev_frequent[j])
                l1.sort(); l2.sort()
                if l1[:-1] == l2[:-1]:
                    new_candidates.add(prev_frequent[i] | prev_frequent[j])
        return new_candidates

    def _generate_rules(self):
        """Extracts association rules (A -> B) based on confidence."""
        for itemset, support_abc in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            # For an itemset like {A, B}, test A -> B and B -> A
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    support_a = self.frequent_itemsets.get(antecedent)
                    confidence = support_abc / support_a
                    
                    if confidence >= self.min_confidence:
                        lift = confidence / self.frequent_itemsets.get(consequent)
                        self.rules.append({
                            'rule': f"{set(antecedent)} -> {set(consequent)}",
                            'confidence': confidence,
                            'lift': lift
                        })

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Sample Market Basket Data
    dataset = [
        ['Milk', 'Bread', 'Eggs'],
        ['Milk', 'Bread'],
        ['Milk', 'Eggs'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread', 'Butter', 'Eggs']
    ]

    model = Apriori(min_support=0.4, min_confidence=0.6)
    model.fit(dataset)

    print("--- Frequent Itemsets ---")
    for itemset, support in model.frequent_itemsets.items():
        print(f"{set(itemset)}: {support:.2f}")

    print("\n--- Association Rules ---")
    for r in model.rules:
        print(f"Rule: {r['rule']} | Conf: {r['confidence']:.2f} | Lift: {r['lift']:.2f}")