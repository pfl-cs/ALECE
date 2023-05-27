def load_str_str_map(filepath, delim="||", key_pos=0, val_pos=1):
    m = {}
    with open(filepath, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            terms = line.strip().split(delim)
            key = terms[key_pos]
            if val_pos < 0:
                value = line
            else:
                value = terms[val_pos]
            m[key] = value
    return m


def read_all_lines(path):
    with open(path, 'r') as reader:
        lines = reader.readlines()
    return lines

def read_first_line(path):
    with open(path, 'r') as reader:
        line = reader.readline()
    return line

def read_all(path):
    with open(path, 'r') as reader:
        s = reader.read()
    return s


def write_all(path, s):
    with open(path, 'w') as writer:
        writer.write(s)

def write_all_lines(path, lines):
    with open(path, 'w') as writer:
        writer.writelines(lines)
