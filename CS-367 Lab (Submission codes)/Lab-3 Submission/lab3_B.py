import random

def generate_3_sat(n, m):
    clauses = []
    for _ in range(m):
        clause = set()
        while len(clause) < 3:
            var = random.randint(1, n)
            sign = random.choice([True, False])
            literal = var if sign else -var
            if var not in [abs(lit) for lit in clause]:
                clause.add(literal)
        clauses.append(list(clause))
    return clauses

def hill_climbing(clauses, n, max_steps, heuristic):
    def evaluate(solution):
        return heuristic(clauses, solution)

    solution = [random.choice([True, False]) for _ in range(n)]
    best_score = evaluate(solution)

    for step in range(max_steps):
        neighbor = solution[:]
        var_to_flip = random.randint(0, n-1)
        neighbor[var_to_flip] = not neighbor[var_to_flip]

        neighbor_score = evaluate(neighbor)
        if neighbor_score > best_score:
            solution = neighbor
            best_score = neighbor_score

    return best_score, solution


def beam_search(clauses, n, max_steps, beam_width, heuristic):
    def evaluate(solution):
        return heuristic(clauses, solution)

    beam = [[random.choice([True, False]) for _ in range(n)] for _ in range(beam_width)]
    best_solution = None
    best_score = -1

    for step in range(max_steps):
        neighbors = []
        for solution in beam:
            for i in range(n):
                neighbor = solution[:]
                neighbor[i] = not neighbor[i]
                neighbors.append(neighbor)

        # Sort neighbors by heuristic
        neighbors.sort(key=lambda sol: evaluate(sol), reverse=True)
        beam = neighbors[:beam_width]

        top_score = evaluate(beam[0])
        if top_score > best_score:
            best_score = top_score
            best_solution = beam[0]

    return best_score, best_solution

def vnd(clauses, n, max_steps, heuristic):
    def evaluate(solution):
        return heuristic(clauses, solution)

    def neighborhood1(solution):           
        neighbors = []
        for i in range(n):
            neighbor = solution[:]
            neighbor[i] = not neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def neighborhood2(solution):        
        neighbors = []
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = solution[:]
                neighbor[i] = not neighbor[i]
                neighbor[j] = not neighbor[j]
                neighbors.append(neighbor)
        return neighbors

    def neighborhood3(solution):       
        neighbors = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    neighbor = solution[:]
                    neighbor[i] = not neighbor[i]
                    neighbor[j] = not neighbor[j]
                    neighbor[k] = not neighbor[k]
                    neighbors.append(neighbor)
        return neighbors

    neighborhoods = [neighborhood1, neighborhood2, neighborhood3]

    solution = [random.choice([True, False]) for _ in range(n)]
    best_score = evaluate(solution)

    for step in range(max_steps):
        for neighborhood in neighborhoods:
            neighbors = neighborhood(solution)
            neighbors.sort(key=lambda sol: evaluate(sol), reverse=True)
            top_neighbor = neighbors[0]
            neighbor_score = evaluate(top_neighbor)

            if neighbor_score > best_score:
                solution = top_neighbor
                best_score = neighbor_score
                break

    return best_score, solution


# Heuristic 1: Number of satisfied clauses
def heuristic_1(clauses, solution):
    satisfied = 0
    for clause in clauses:
        if any((lit > 0 and solution[abs(lit) - 1]) or (lit < 0 and not solution[abs(lit) - 1]) for lit in clause):
            satisfied += 1
    return satisfied

# Heuristic 2: Weighted clause satisfaction
def heuristic_2(clauses, solution):
    satisfied = 0
    for clause in clauses:
        if any((lit > 0 and solution[abs(lit) - 1]) or (lit < 0 and not solution[abs(lit) - 1]) for lit in clause):
            satisfied += 1
        else:
            satisfied -= 0.5
    return satisfied
