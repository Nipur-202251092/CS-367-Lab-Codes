import random

def generate_k_sat(n, m, k):
    clauses = []

    for _ in range(m):
        clause = set()
        while len(clause) < k:
            var = random.randint(1, n)
            sign = random.choice([True, False])
            literal = var if sign else -var

            if var not in [abs(lit) for lit in clause]:
                clause.add(literal)

        clauses.append(sorted(clause, key=abs))

    return clauses

n = int(input("Enter the number of variables (n): "))
m = int(input("Enter the number of clauses (m): "))
k = int(input("Enter the number of literals per clause (k): "))

k_sat_formula = generate_k_sat(n, m, k)
for clause in k_sat_formula:
    print(f"({' ∨ '.join([f'¬x{abs(lit)}' if lit < 0 else f'x{lit}' for lit in clause])})")
