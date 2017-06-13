'''Crop sequence from 2000 nuc to defined promoter range
'''
import csv

center = 2001//2
with open('UTR_chlamy.2k.txt') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    out = []
    for i, line in enumerate(reader):
        seq = line[1][center - 450:center + 50]
        out.append([line[0], seq])

csv.writer(open('UTR_chlamy.500bp.txt', 'w'), delimiter='\t').writerows(out)
