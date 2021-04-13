# import statements
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo import draw, read
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Seq import Seq
from matplotlib import pyplot as plt
import numpy as np 
from skbio.stats.distance import DistanceMatrix
from skbio import tree
from skbio import io
import os
import random
import sys, getopt

'''
options:
    d = display
    m = msa comparison
    b X = X bootstrap iterations
args: 
    input_fn
'''

# global variables
aa_dict = {}
aa_dict['A'] = np.array([1, 0, 0, 0])
aa_dict['G'] = np.array([0, 0, 0, 0])
aa_dict['I'] = np.array([4, 0, 0, 0])
aa_dict['L'] = np.array([4, 0, 0, 0])
aa_dict['P'] = np.asarray([3, 0, 0, 0])
aa_dict['V'] = np.asarray([3, 0, 0, 0])
aa_dict['F'] = np.asarray([7, 0, 0, 0])
aa_dict['W'] = np.asarray([9, 1, 0, 0])
aa_dict['Y'] = np.asarray([7, 0, 1, 1])
aa_dict['D'] = np.asarray([2, 0, 2, 0])
aa_dict['E'] = np.asarray([3, 0, 2, 0])
aa_dict['R'] = np.asarray([4, 3, 0, 0])
aa_dict['H'] = np.asarray([4, 2, 0, 0])
aa_dict['K'] = np.asarray([4, 1, 0, 0])
aa_dict['S'] = np.asarray([1, 0, 1, 0])
aa_dict['T'] = np.asarray([2, 0, 1, 0])
aa_dict['C'] = np.asarray([1, 0, 0, 1])
aa_dict['M'] = np.asarray([3, 0, 0, 1])
aa_dict['N'] = np.asarray([2, 1, 1, 0])
aa_dict['Q'] = np.asarray([3, 1, 1, 0])
aas = list(aa_dict.keys())
aas.sort()

# functions
def get_sequences(input_fn, n):
    if(n>0):
        initial_sequences = SeqIO.to_dict(SeqIO.parse(input_fn, "fasta"))
        sequences = {}
        for iter, iseq in zip(range(0, len(initial_sequences)), initial_sequences.values()):
            for i in range(0, n):
                old_seq = str(iseq.seq)
                new_seq = random.choices(old_seq, k=len(old_seq))
                id = 'BSeq' + str(iter) + '_' + str(i)
                sequences[id] = new_seq
    else:
        sequences = SeqIO.to_dict(SeqIO.parse(input_fn, "fasta"))
    return sequences

def get_nc_dm(sequences, b, a):
    counts = {}
    for key, sequence in sequences.items():
        if(a):
            seqcomp = [0.0 for i in range(0, 20)]
            for aa in sequence:
                indx = aas.index(aa)
                seqcomp[indx] += 1.0
        else:
            seqcomp = [0.0, 0.0, 0.0, 0.0]
            for aa in sequence:
                seqcomp = [sum(x) for x in zip(seqcomp, aa_dict[aa])]
        total_comp = sum(seqcomp) # number of elements in side chains
        seqcomp = [sc/total_comp for sc in seqcomp]
        if(b):
            counts[key] = list(seqcomp)
        else:
            counts[sequence.id] = list(seqcomp)
    normalized_counts = np.array(list(counts.items()), dtype=object)
    matrix_names = list(normalized_counts[:, 0])
    distance_matrix = [[]] * len(normalized_counts)

    return counts, normalized_counts, matrix_names, distance_matrix

def get_centroid(normalized_counts):
    centroid = [0.0, 0.0, 0.0, 0.0]
    for count in normalized_counts[:, 1]:
        centroid = [(centroid[indx] + count[indx]) for indx in range(0, len(centroid))]
        centroid = [c/len(normalized_counts) for c in centroid]
    return centroid

def graph_results(normalized_counts, d):
    nc = list(normalized_counts[:, 1])

    c = [nc[i][0] for i in range(len(nc))]
    n = [nc[i][1] for i in range(len(nc))]
    o = [nc[i][2] for i in range(len(nc))]
    s = [nc[i][3] for i in range(len(nc))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(c, n, o, c=s, cmap=plt.hot())
    fig.colorbar(img)
    plt.savefig('graph.png', format='png')
    if(d):
        plt.show()

def get_scores(normalized_counts, centroid, distance_matrix):
    pairwise_scores = {}
    centroid_scores = {}
    for i in range(0, len(normalized_counts)):
        id = normalized_counts[i][0]
        comp = normalized_counts[i][1]
        score = [0.0, 0.0, 0.0, 0.0]
        row_distances = []
        centroid_scores[id] = [comp[indx] - centroid[indx] for indx in range(0, len(centroid))]
        for j in range(0, len(normalized_counts)):
            distance = [0.0, 0.0, 0.0, 0.0]
            if i==j:
                row_distances.append(0.0)
            else:
                comparison_comp = np.array(normalized_counts[j][1])
                for indx in range(0, len(score)):
                    distance[indx] = ((comp[indx] - comparison_comp[indx])**2)**0.5
                    score[indx] = score[indx] + comp[indx]-comparison_comp[indx]
                row_distances.append(sum(distance))
        distance_matrix[i] = row_distances
        pairwise_scores[id] = score
    return pairwise_scores, centroid_scores, distance_matrix

def get_tree(distance_matrix, matrix_names, b, d):
    if(not b):
        matrix_names = [s[s.index('|')+1:] for s in matrix_names]
        matrix_names = [s[s.index('|')+1:] for s in matrix_names]
    dm = DistanceMatrix(data=distance_matrix, ids=matrix_names)
    f = open("distance_matrix", "w")
    f.write('\t' + str(len(matrix_names)) + '\n')
    io.write(dm, format='lsmat', into=f)
    f.close()
    with open("distance_matrix", "r") as f:
        lines = f.readlines()
    with open("distance_matrix", "w") as f:
        for i, line in zip(range(0, len(lines)), lines):
            if not i==1:
                f.write(line)
    bias_tree = tree.nj(dm)
    f = open("bias_tree.dnd", "w")
    io.write(bias_tree, format='newick', into=f)
    f.close()
    if(d):
        printable_tree = read("./bias_tree.dnd", "newick")
        draw(printable_tree)

def save_results(counts, centroid, pairwise_scores, centroid_scores, distance_matrix):
    f = open("results.txt", "w+")
    f.write('Normalized Counts: % composition of side chains, formatted [C, N, O, S]\n')
    f.write(str(counts))
    f.write('\n\nCentroid: from counts\n')
    f.write(str(centroid))
    f.write('\n\nPairwise Scores: cumulative distance from all other sequences\n')
    f.write(str(pairwise_scores))
    f.write('\n\nCentroid Scores: distance from centroid\n')
    f.write(str(centroid_scores))
    f.write('\n\nDistance Matrix: euclidean distance between sequences\n')
    f.write(str(distance_matrix))
    f.close()

def msa(fname):
    fastas = SeqIO.parse(fname, "fasta")
    sequences = []
    maxlen = 0
    for record in fastas:
        id_str = record.id
        id_str = id_str[id_str.index('|')+1:]
        id_str = id_str[id_str.index('|')+1:]
        record.id = id_str
        sequences.append(record)
        length = len(record.seq)
        if maxlen < length:
            maxlen = length

    for record in sequences:
        record_str = str(record.seq)
        for i in range(len(record_str), maxlen):
            record_str = record_str + '-'
        record.seq = Seq(record_str)
    alignment = MultipleSeqAlignment(sequences)
    calc = DistanceCalculator('identity')
    msa_dm = calc.get_distance(alignment)
    f = open("msa_dm", "w")
    f.write("\t" + str(len(sequences)) + "\n")
    f.write(str(msa_dm))
    f.close()

def nb_biases(f, d, m, a):
    sequences = get_sequences(f, 0)
    counts, normalized_counts, matrix_names, distance_matrix = get_nc_dm(sequences, False, a)
    centroid = get_centroid(normalized_counts)
    pairwise_scores, centroid_scores, distance_matrix = get_scores(normalized_counts, centroid, distance_matrix)
    get_tree(distance_matrix, matrix_names, False, d)
    graph_results(normalized_counts, d)
    if(m):
        msa(f)
    save_results(counts, centroid, pairwise_scores, centroid_scores, distance_matrix)

def bootstrapped_biases(f, i, d, m, a):
    sequences = get_sequences(f, i)
    counts, normalized_counts, matrix_names, distance_matrix = get_nc_dm(sequences, True, a)
    centroid = get_centroid(normalized_counts)
    pairwise_scores, centroid_scores, distance_matrix = get_scores(normalized_counts, centroid, distance_matrix)
    get_tree(distance_matrix, matrix_names, True, d)
    graph_results(normalized_counts, d)
    if(m):
        msa(f)
    save_results(counts, centroid, pairwise_scores, centroid_scores, distance_matrix)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "admcb:")
    except getopt.GetoptError:
        print('Error: Invalid options')
        sys.exit(2)

    input_fn = './seq'
    if(len(args) > 0):
        input_fn = args[0]

    iters = 0
    aa = False
    msa = False
    display = False
    for opt, arg in opts:
        if opt=='-a':
            aa = True
        if opt=='-b':
            iters = int(arg)
        elif opt=='-d':
            display = True
        elif opt=='-m':
            msa = True
    
    if(iters>0):
        bootstrapped_biases(input_fn, iters, display, msa, aa)
    else:
        nb_biases(input_fn, display, msa, aa)

if __name__ == "__main__":
    main(sys.argv[1:])