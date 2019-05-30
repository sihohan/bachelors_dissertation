from itertools import product

import numpy as np
import random


def cartesian_product_generation(k, iterable):

    # Generate the k-fold Cartesian product for an interable, i.e. 'ACGT'
    return [''.join(kmer) for kmer in product('{}'.format(iterable), repeat=k)]


def sequence2positional_feature(k, sequence):

    '''
    :param k: k-mer
    :param sequence: nucleotide sequence
    :return: positional feature set
    '''

    # Preallocate a numpy array filled with zeros
    positional_feature_set = np.zeros((len(sequence)-k+1)*(4**k), dtype=int)

    # Generate the k-fold Cartesian product for 'ACGT'
    cartesian_product = cartesian_product_generation(k=k, iterable='ACGT')

    # For each position of the sequence
    for s in range(len(sequence)-k+1):

        # Take a k-mer, e.g., a mononucleotide if k = 1, a dinucleotide if k = 2, ...
        k_mer = sequence[s:s+k]

        # If the k_mer contains an 'N', i.e. an unknown nucleotide, ignore it
        if 'N' in k_mer:
            continue

        # Assign the value 1 to a feature if the k-mer at the defined position is a mono-, a di- or a trinucleotide
        # and add each feature to the positional feature set
        if sequence[s:s+k] in cartesian_product:
            positional_feature_set[s*(4**k)+(cartesian_product.index(k_mer))] = 1

    # Return the positional feature set with length = (len(sequence)-k+1)*(4**k)
    return positional_feature_set


def sequence2compositional_feature(k, p, q, sequence):

    '''
    :param k: k-mer
    :param p: context upstream
    :param q: context downstream
    :param sequence: nucleotide sequence
    :return: compositional feature set
    '''

    # Preallocate a numpy array filled with zeros
    compositional_feature_set = np.zeros(2*(4**k), dtype=int)

    # Genrerate the k-fold Cartesian product for 'ACGT'
    cartesian_product = cartesian_product_generation(k=k, iterable='ACGT')

    # For each position of the sequence
    for s in range(len(sequence)-k+1):

        # Take a k-mer, e.g., a mononucleotide if k=1, a dinucleotide if k=2, ...
        k_mer = sequence[s:s+k]
        # If the k_mer contains an 'N', i.e. an unknown nucleotide, ignore it
        if 'N' in k_mer:
            continue

        # Assign the value 1 to a feature if the k-mer is in the local upstream context
        # and add each feature to the compositional feature set
        if k_mer in sequence[:p] and compositional_feature_set[cartesian_product.index(k_mer)] == 0:
            compositional_feature_set[cartesian_product.index(k_mer)] = 1

        # Assign the value 1 to a feature if the k-mer is in the local downstream context
        # and add each feature to the compositional feature set
        if k_mer in sequence[len(sequence)-q:] and \
                compositional_feature_set[cartesian_product.index(k_mer)+(4**k)] == 0:
            compositional_feature_set[cartesian_product.index(k_mer)+(4**k)] = 1

    # Return the compositional feature set with length = 2*(4**k)
    return compositional_feature_set


def sequence2coding_potential(p, q, sequence):

    # k = 3, because a codon is a nucleotide triplet
    k = 3

    # Preallocate a numpy array filled with zeros
    coding_potential_feature_set = np.zeros(3*2*(4**k), dtype=int)

    # Generate the 3-fold Cartesian product for 'ACGT'
    all_codons = cartesian_product_generation(k, 'ACGT')

    # For each reading frame R = 1, 2, 3
    for r in range(k):

        # For each position of the sequence
        for s in range(r, len(sequence)-k+1, k):

            # Take a codon triplet
            codon = sequence[s:s+k]
            # If the k_mer contains an 'N', i.e. an unknown nucleotide, ignore it
            if 'N' in codon:
                continue

            # Assign the value 1 to a feature if the k-mer is in the local upstream context
            # and add each feature to the compositional feature set
            if codon in sequence[:p] and coding_potential_feature_set[all_codons.index(codon)+r] == 0:
                coding_potential_feature_set[all_codons.index(codon)+r] = 1

            # Assign the value 1 to a feature if the k-mer is in the local downstream context
            # and add each feature to the compositional feature set
            if codon in sequence[len(sequence)-q:] and \
                    coding_potential_feature_set[all_codons.index(codon)+(r-1)*(4**k)] == 0:
                coding_potential_feature_set[all_codons.index(codon)+(r-1)*(4**k)] = 1
                    
    return coding_potential_feature_set


def train_test_extraction(data_type, p, q, feature_type, k, subsample):

    '''
    :param data_type: 'donors' or 'acceptors'
    :param p: context upstream
    :param q: context downstream
    :param feature_type: 'positional', 'compositional' or 'potential'
    :param k: k-mer
    :param subsample: True or False
    :return: manually extracted feature sets
    '''

    def sequence_extraction(data_file, p, q, subsample):

        '''
        :param data_file: 'donors.pos', 'donors.neg', 'acceptors.pos' or 'acceptors.neg'
        :param p: context upstream
        :param q: context downstream
        :return: sequence with p context upstream and q context downstream
        '''

        # Read sequences
        seq_list = open('{}'.format(data_file)).read().splitlines()[1::2]

        # Randomly subsample if needed, i.e. the data set is too big
        if subsample is True:
            if data_file.split('.')[1] == 'pos':
                # Specify subsample size in the second argument
                seq_list = random.sample(seq_list, 1000)
            if data_file.split('.')[1] == 'neg':
                # Specify subsample size in the second argument
                seq_list = random.sample(seq_list, 10000)

        # Define sequence length, and local contexts p and q
        seq_length = len(seq_list[0])
        start_index = seq_length//2-1-p
        end_index = seq_length//2+1+q

        # Preallocate a numpy array filled with 'None'
        extracted_seq_list = np.full(len(seq_list), fill_value=None)

        # For all samples in the sequence list
        for seq_i in range(len(seq_list)):

            # Take a sequence
            sequence = seq_list[seq_i]

            # Take the subset with p context upstream and q context downstream
            extracted_seq_list[seq_i] = sequence[start_index:end_index]

        # Return the array containing sequences with p context upstream and q context downstream
        return extracted_seq_list

    # Extract subsets from donor samples
    if data_type == 'donors':
        pos_seq_list = sequence_extraction(data_file='donors.pos', p=p, q=q, subsample=subsample)
        neg_seq_list = sequence_extraction(data_file='donors.neg', p=p, q=q, subsample=subsample)

    # Extract subsets from acceptors samples
    if data_type == 'acceptors':
        pos_seq_list = sequence_extraction(data_file='acceptors.pos', p=p, q=q, subsample=subsample)
        neg_seq_list = sequence_extraction(data_file='acceptors.neg', p=p, q=q, subsample=subsample)

    # Save the positive sequence list in a .npy format
    # This sequence will later be used for nucleotide pattern detection
    np.save('{}_pos_seq_list_feat_imp.npy'.format(data_type), pos_seq_list)

    # Preallocate a numpy array filled with 'None'
    y = np.zeros(len(pos_seq_list)+len(neg_seq_list), dtype=int)

    # Positional feature sets
    if feature_type == 'positional':

        # Preallocate a numpy array filled with 'None'
        X = np.zeros([len(pos_seq_list)+len(neg_seq_list), (len(pos_seq_list[0])-k+1)*(4**k)], dtype=int)

        # For each position in the preallocated feature set
        for i in range(len(pos_seq_list)+len(neg_seq_list)):

            # Construct the positional feature set for all positive sample sequences
            if i < len(pos_seq_list):
                pos_seq = pos_seq_list[i]
                X[i] = sequence2positional_feature(k=k, sequence=pos_seq)
                y[i] = 1

            # Construct the positional feature set for all negative sample sequences
            else:
                neg_seq = neg_seq_list[i-len(pos_seq_list)]
                X[i] = sequence2positional_feature(k=k, sequence=neg_seq)
                y[i] = 0

    # Compositional feature sets
    if feature_type == 'compositional':

        # Preallocate a numpy array filled with 'None'
        X = np.zeros([len(pos_seq_list)+len(neg_seq_list), 2*(4**k)], dtype=int)

        # For each position in the preallocated feature set
        for i in range(len(pos_seq_list)+len(neg_seq_list)):

            # Construct the compositional feature set for all positive sample sequences
            if i < len(pos_seq_list):
                pos_seq = pos_seq_list[i]
                X[i] = sequence2compositional_feature(k=k, p=p, q=q, sequence=pos_seq)
                y[i] = 1
                
            # Construct the compositional feature set for all negative sample sequences
            else:
                neg_seq = neg_seq_list[i-len(pos_seq_list)]
                X[i] = sequence2compositional_feature(k=k, p=p, q=q, sequence=neg_seq)
                y[i] = 0

    # Coding potential feature sets
    if feature_type == 'potential':

        # Preallocate a numpy array filled with 'None'
        X = np.zeros([len(pos_seq_list)+len(neg_seq_list), 3*2*(4**k)], dtype=int)

        # For each position in the preallocated feature set
        for i in range(len(pos_seq_list)+len(neg_seq_list)):

            # Construct the coding potential feature set for all positive sample sequences
            if i < len(pos_seq_list):
                pos_seq = pos_seq_list[i]
                X[i] = sequence2coding_potential(p=p, q=q, sequence=pos_seq)
                y[i] = 1
                
            # Construct the coding potential set for all negative sample sequences
            else:
                neg_seq = neg_seq_list[i-len(pos_seq_list)]
                X[i] = sequence2coding_potential(p=p, q=q, sequence=neg_seq)
                y[i] = 0

    # Return the manually extracted feature sets
    return X, y

