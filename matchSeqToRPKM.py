'''Get gene name, rpkm and sequence from input file
'''
import csv
import sys

seq = sys.argv[1]

# Open input files with genename and rpkm
with open(seq) as f:
    out = []
    reader = csv.reader(f, delimiter=',')
    rpkms = {k[0]: k[1] for k in reader}

    # Match sequence to gene names
    with open('UTR_chlamy.500bp.txt') as f:
        newseq = csv.reader(f, delimiter='\t')
        for line in newseq:
            name = '.'.join(line[0].split('.')[:4])
            try:
                rpkm = rpkms[name]
            except:
                continue
            out.append([name, rpkm, line[-1]])

# Svae data
csv.writer(open('rpkmWSequences.500bp.txt', 'w'), delimiter='\t').writerows(out)
