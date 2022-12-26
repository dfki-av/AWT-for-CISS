#!/usr/bin/python

import sys

path = sys.argv[1]
task = sys.argv[2]

avg_first = 0.
avg_last = 0.
initial = task.split('-')[0]
col_index = int(initial)+1

with open(path, 'r') as f:
    for line_index, line in enumerate(f):
        split = line.split(',')
        a = split[-1]
        a = float(a)
        step = line.split(',')[0]

        if col_index > -1:
            if len(split[1:col_index+1]) == 0:
                avg_first = 0.
            else:
                avg_first = sum([float(i) for i in split[1:col_index+1] if i not in ('x', 'X')]) / len(split[1:col_index+1])
            if len(split[col_index+1:-1]) == 0:
                avg_last = 0.
            else:
                avg_last = sum([float(i) for i in split[col_index+1:-1] if i not in ('x', 'X')]) / len(split[col_index+1:-1])

print(f"Last Step: {step}")
print(f"Final Mean IoU {round(100 * a, 2)}")
print(f'Mean IoU first {round(100 * avg_first, 2)}')
print(f'Mean IoU last {round(100 * avg_last, 2)}')

