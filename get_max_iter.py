import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', metavar='F', type=str)
args = parser.parse_args()
file_name = args.file

max_iter = 0.0
max_num = 100.0
with open(file_name, 'r') as f:
    lines = f.readlines()
    for l in lines:
        ll = l.strip()
        distance = float(ll)
        if distance < 100.0:
            if max_iter < distance:
                max_iter = distance

print("max_iter = {}".format(int(max_iter / 0.1)))