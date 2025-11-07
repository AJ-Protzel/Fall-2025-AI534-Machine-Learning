__author__ = "Liang Huang"

import sys
import os

ERROR_LOG_PATH = "dev_error_log.txt"

def error(msg):
    print(f"ERROR: {msg}")
    exit(1)

def truncate(s):
    return s if len(s) < 150 else f"{s[:100]} ...(truncated)... {s[-20:]}"

def symbol(log_path=ERROR_LOG_PATH):
    errors = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    errors.append(float(stripped))

    if len(errors) < 2:
        return "=="

    current_error = errors[-1]
    last_error = errors[-2]
    best_error = min(errors)

    if current_error == last_error:
        symbol1 = '='
    elif current_error > last_error:
        symbol1 = '-'
    else:
        symbol1 = '+'

    if current_error == best_error:
        symbol2 = '='
    elif current_error > best_error:
        symbol2 = '-'
    else:
        symbol2 = '+'

    return symbol1 + symbol2

if __name__ == "__main__":
    infile = sys.stdin if len(sys.argv) == 1 else open(sys.argv[1])

    pos = 0
    for i, line in enumerate(infile, 5999):
        fields = line.strip().split(",")
        if i == 5999:
            if fields != ['id', 'age', 'sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'hours', 'country', 'target']:
                error("Header format incorrect.")
            continue
        if len(fields) != 11:
            error(f"Line {i}: Expected 11 fields, got {len(fields)}.\n{truncate(line)}")
        if fields[-1].upper() not in [">50K", "<=50K"]:
            error(f"Line {i}: Invalid target value '{fields[-1]}'.")
        if int(fields[0]) != i:
            error(f"Line {i}: ID mismatch (got {fields[0]}).")
        pos += fields[-1].upper() == ">50K"

    if i != 6999:
        error(f"Expected 1000 data lines, got {i + 1 - 6000}.")

    pos_rate = pos / 10.0
    current_error = 100 - pos_rate  # Assuming error = 100 - positive rate

    print(f"Success {symbol()}")