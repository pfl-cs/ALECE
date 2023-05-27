import re
def remove_duplicate_blanks(s):
    l = len(s)
    s = s.replace("  ", " ")
    while l != len(s):
        l = len(s)
        s = s.replace("  ", " ")

    return s

def load_str_str_map_from_lines(lines, delim="||", key_pos=0, val_pos=1):
    m = {}
    for line in lines:
        terms = line.strip().split(delim)
        key = terms[key_pos]
        if val_pos < 0:
            value = line
        else:
            value = terms[val_pos]
        m[key] = value
    return m

def is_number(_s):
    # patt = '(?!\d+)'
    # patt = re.compile(patt)
    s = _s.strip()
    if s == '.':
        return False
    terms = s.split('.')
    if len(terms) > 2:
        return False
    for term in terms:
        if len(term) > 0:
            if term.isdigit() == False:
                return False
    return True
