def term_count(term):
    return len(set(term)), len(term)


def drop_term(term, x2, x3):
    unique_terms, total_terms = term_count(term)
    extra = False
    if not x2:
        extra = unique_terms == 1 and total_terms > 1
    elif not x3:
        extra = unique_terms == 1 and total_terms > 2
    return (total_terms > unique_terms > 1 and total_terms > 1) or extra
