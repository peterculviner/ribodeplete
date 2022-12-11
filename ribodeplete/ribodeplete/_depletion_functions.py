# load nessasary functions and libraries
import numpy as np
from scipy.stats import norm
from Bio.SeqUtils.MeltingTemp import Tm_NN as tm
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq
from Bio import SeqIO
import os,subprocess
import ribodeplete as rd

# define single functions

def optimizeprobes(fasta_path=None, n_probes=25, probe_mutation_steps=25, optimization_cycles=10,
                   target_tm=62.5, tm_dist=norm, tm_dist_kwargs={'loc':62.5,'scale':2,},
                   probe_start_len=(15,25), show_plots=False):
    # generate alignment object from aligned fasta file
    align_obj = rd.AlignedrRNA(fasta_path)
    # generate probes
    probes_left = n_probes
    while probes_left != 0:
        try:
            probe_length = np.random.randint(probe_start_len[0], probe_start_len[1]+1) # randomly determine probe length between parameters
            probe_start  = np.random.randint(len(align_obj.rrna[0])-probe_length) # randomly decide on a probe position (may not be successful if crosses an unaligned region)
            align_obj.addProbe(probe_start, probe_length, template='chimera') # add probes, will through error if unsuccessful
            probes_left-=1 # probe successfully added, tick forward
        except ValueError:
            pass # probe threw a value error, try again
    # generate the distribution for probe picks
    distribution_function = tm_dist(**tm_dist_kwargs)
    # start optimization cycles
    cycle_number = 0 # set to 0 initially for storage of original probe positions etc
    if show_plots is not False:
        if cycle_number in show_plots:
            align_obj.snapshotProbes()
    # store probe Tms
    align_obj.snapshotProbeTms()
    while cycle_number < optimization_cycles: # track number of cycles
        for probe_i in np.arange(len(align_obj.probes)):
            # now diversify given probe
            new_probes = align_obj.diversifyProbe(probe_i, probe_mutation_steps)
            new_tm = np.asarray([np.min(p.getTm()) for p in new_probes])
            if np.all(np.isnan(new_tm)): # if all Tms are undefined
                picked_probe = new_probes[np.random.choice(np.arange(len(new_probes)))] # just pick a random primer
            else: # if one or more of them has a real Tm, use the distribution to pick best(ish) primer
                pdf_outputs = distribution_function.pdf(new_tm)
                pdf_outputs[np.isnan(pdf_outputs)] = 0
                # if all probabilities end up being zero (some things are too far from the pdf), pick closest to the target tm
                if all(pdf_outputs == 0):
                    picked_probe = new_probes[np.argmin(np.abs(target_tm - new_tm))]
                else:
                    picked_probe = new_probes[np.random.choice(np.arange(len(pdf_outputs)), p=pdf_outputs / np.sum(pdf_outputs))]
            align_obj.probes[probe_i] = picked_probe
        cycle_number += 1 # optimization cycle completed, tick up by 1
        if show_plots is not False:
            if cycle_number in show_plots:
                align_obj.snapshotProbes()
        # store probe Tms
        align_obj.snapshotProbeTms()
    return align_obj

def extractrrna(genbank_path, species_label, verbose = False, additional_nt = 0):
    genome = SeqIO.read(genbank_path, 'genbank')
    # in order: 16S, 23S, 5S
    output_names = [[],[],[]]
    output_sequences = [[],[],[]]
    n_sequences = [0,0,0]
    n_duplicate = [0,0,0]
    for feat in genome.features:
        if feat.type == 'rRNA':
            # extract gene sequence
            gene_sequence = genome.seq[feat.location.start-additional_nt:feat.location.end+additional_nt]
            if str(feat.strand) == '-1':
                gene_sequence = gene_sequence.reverse_complement()
            gene_sequence = str(gene_sequence)
            try: # generate a gene name from either the 'gene' or 'locus_tag' qualifier
                gene_name = '%s_%s'%(species_label, feat.qualifiers['gene'][0])
            except KeyError:
                gene_name = '%s_%s'%(species_label, feat.qualifiers['locus_tag'][0])
            # determine which list to append it to, append it to the list
            gene_product = (feat.qualifiers['product'][0]).lower()
            for i,rrna_type in enumerate(['16s','23s','5s']):
                if rrna_type in gene_product: # try to add to the proper list
                    n_sequences[i] += 1
                    if gene_sequence in output_sequences[i]: # if exact sequence already exists, don't add a duplicate
                        n_duplicate[i] += 1
                        break
                    output_sequences[i].append(gene_sequence)
                    if gene_name in output_names[i]:
                        name_increment = 1
                        while True:
                            if gene_name+str(name_increment) in output_names[i]:
                                name_increment+=1
                            else:
                                break
                        output_names[i].append(gene_name+str(name_increment))
                    else:
                        output_names[i].append(gene_name)
                    break
            else:
                raise TypeError('Gene product %s is not a recognized rRNA type!'%gene_product)
    if verbose: # print summary information about genome
        # description, number of each type of loci found
        print('%s\nnumber of loci: %i/%i/%i (16S/23S/5S)\nduplicate: %i/%i/%i'%(genome.description,
                                                                            n_sequences[0],n_sequences[1],n_sequences[2],
                                                                            n_duplicate[0],n_duplicate[1],n_duplicate[2]))
    return output_names, output_sequences

def alignrRNA(genome_paths, verbose = False):
    # in order: 16S, 23S, 5S
    input_names = [[],[],[]]
    input_sequences = [[],[],[]]
    for gpath in genome_paths: # for each genome, extract rRNA of each type
        species_label = gpath.split('/')[-1].split('_')[0]
        names, sequences = extractrrna(gpath, species_label, verbose = verbose)
        for i in range(3): # add to list
            input_names[i] = names[i]+input_names[i]
            input_sequences[i] = sequences[i]+input_sequences[i]
    # output 3 alignments of the rRNA
    output_names = []
    output_sequences = []
    for i in range(3):
        names, sequences = muscleAlign(input_names[i],input_sequences[i])
        output_names.append(names)
        output_sequences.append(sequences)
    return output_names, output_sequences

def muscleAlign(names, sequences):
    # write a temporary fasta file for muscle to read / align
    writeFasta('_temp.fasta', names, sequences)
    # conduct a muscle alignment on the temporary fasta
    subprocess.call(['muscle','-in','_temp.fasta','-out','_temp_aligned.fasta'])
    # import alignments back into python
    names, sequences = readFasta('_temp_aligned.fasta',True)
    # clean up temporary file
    os.remove('_temp.fasta')
    return names, sequences

def writeFasta(filepath, names, sequences):
    # accepts list of names (spaces will be replaced with underscores) and list of sequences, writes to filepath
    with open(filepath,'w') as f:
        for n,seq in zip(names,sequences):
            f.write('>%s\n%s\n'%(n.replace(' ','_'),str(seq)))
    return filepath

def readFasta(filepath, delete=False):
    # returns names and records
    with open(filepath,'r') as f:
        lines = f.readlines()
        # now take out the records
        records = []
        names = []
        current_record = ''
        for line in lines:
            if line[0] == '>':
                names.append(line[1:-1])
                records.append(current_record)
                current_record = ''
            else:
                current_record += line[:-1]
        records.append(current_record)
        names = np.asarray(names)
        records = np.asarray(records[1:])
    if delete: # burn after reading
        os.remove(filepath)
    return names, records

def masktoregions_single(in_mask):
    current_strand = in_mask.copy().astype(float)
    current_strand[-1] = np.nan # set final position to np.nan to avoid overlap issues
    transitions = current_strand - np.roll(current_strand,1)
    true_start = np.where(transitions == 1)[0]
    true_end   = np.where(transitions == -1)[0] - 1
    if current_strand[0] == 1: # if starts on True, add True start to front end
        true_start = np.r_[0,true_start]
    if in_mask[-1] == True: # if ends on True, add True end to back end
        true_end = np.r_[true_end, len(current_strand)-1]
        if in_mask[-2] == False: # if the one before is False, it's a single point True
            true_start = np.r_[true_start,len(current_strand)-1]
    if np.all(in_mask[-2:] == [True, False]):
        true_end = np.r_[true_end, len(current_strand)-2]
    regions = np.asarray([true_start,true_end]).T
    return regions


def NNTm(oligo_seq, rna_seq, melting_table = mt.R_DNA_NN1):
    # rna_seq given in coding and 5'->3' (with T, not U!) --> mapped to seq in tm function
    # oligo_seq also given in coding and 5'->3' --> mapped to c_seq in tm function
    if '-' in rna_seq: # if a gap is present, return a np.nan Tm
        return np.nan
    rna_seq = Seq(rna_seq)
    oligo_seq = Seq(oligo_seq).complement()
    try:
        return tm(seq=rna_seq, c_seq=oligo_seq, nn_table=melting_table)
    except ValueError: # if multiple mismatches in a row, this will happen
        return np.nan
    
def argoverlappingregions_single(input_region, region_array):
    overlap_i = np.where(np.all([input_region[0] <= region_array[:,1],
                                 input_region[1] >= region_array[:,0]],axis=0))
    return overlap_i[0]