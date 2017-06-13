'''Change sequence to one-hot encoding and save as pickle file
'''
import numpy as np
import csv
import pickle

with open('rpkmWSequences.500bp.txt', 'r') as f:
    out = []
    reader = csv.reader(f, delimiter='\t')

    # Loop though input file
    for line in reader:
        name = line[0]
        rpkm = float(line[1])
        seq = line[2]
        one_hot = np.zeros(shape=(4, 500))

        # Generate one-hot encoding
        for i, nuc in enumerate(seq):
            if nuc == 'A':
                one_hot[0, i] = 1
            elif nuc == 'C':
                one_hot[1, i] = 1
            elif nuc == 'G':
                one_hot[2, i] = 1
            elif nuc == 'T':
                one_hot[3, i] = 1
            else:  # nuc == 'N'
                if nuc != 'N':
                    print(nuc, 'hwat')
                    exit(1)
        out.append([name, rpkm, one_hot])

# Save file
with open('pickle/onehot500bp.-450+50.txt', 'wb') as f:
    pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

