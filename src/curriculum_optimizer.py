def recommend_clos_for_skill_row(row):
    clos = [c for c in row.index if c.startswith('CLO_')]
    if row.get('gap', 0) <= 0:
        return []
    missing = [c for c in clos if row.get(c, 0) == 0]
    return missing
