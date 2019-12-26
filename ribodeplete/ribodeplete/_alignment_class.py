from ribodeplete import extractrrna, alignrRNA, muscleAlign, writeFasta, readFasta, masktoregions_single, NNTm, argoverlappingregions_single
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.colorbar import ColorbarBase as cbb
import seaborn as sns
from Bio.Seq import Seq
from Bio.Alphabet.IUPAC import unambiguous_dna, unambiguous_rna

class AlignedrRNA():
    def readfasta(self, fasta_path):
        # read data from alignment file and convert it to:
        # rrna: list of strings of each sequence
        # species: infer species by part of name before '_'. Used to ensure equal biases between species.
        # unique_nt: list of possible nt at each position in alignment
        # gap_mask: mask True if any gaps in alignments at position
        # nogap_regions: list of regions without gaps
        rrna_fasta = np.asarray(readFasta(fasta_path)) # load rRNA sequences from the alignments
        rrna = rrna_fasta[1,:] # get individual rRNA string sequences
        species = np.asarray([name.split('_')[0] for name in rrna_fasta[0,:]]) # infer species names from part before '_'
        nt_array = np.asarray([[c for c in rna_string] for rna_string in rrna]) # character by character list of all nt in rrna
        # determine unique nt at each position and record if there is a gap for any individual species in alignment
        gap_mask  = []
        unique_nt = []
        for position_nts in nt_array.T:
            position_unique = np.unique(position_nts)
            unique_nt.append(position_unique)
            gap_mask.append('-' in position_unique)
        # re-format gap_mask into nogap_regions
        gap_mask = np.asarray(gap_mask)
        nogap_regions = masktoregions_single(~gap_mask) # find contiguous regions without gaps (inclusive, inclusive)
        gap_regions = masktoregions_single(gap_mask) # # find contiguous gap regions for plotting (inclusive, inclusive)
        return rrna, species, nt_array, unique_nt, gap_mask, gap_regions, nogap_regions
    
    def addProbe(self, left, length, template=None, add_to_list=True, allow_gaps=False):
        # add a primer object at a given location:
        # probe object stored in probelist, initiated on given (or randomly assigned) rRNA
        if template is None: # randomly select (equal speices weight!) a template from list of species
            picked_species = np.random.choice(np.unique(self.species)) # pick a species from list
            rrna_i = np.random.choice(np.where(self.species == picked_species)[0]) # pick an index from rrna list
            # set sequence to perfect match of rrna from rrna[rrna_i]
            sequence = self.rrna[rrna_i][left:left+length]
        elif template == 'chimera': # generate template from randomly selected nt
            sequence = ''.join([np.random.choice(nt_choices) for nt_choices in
                                self.unique_nt[left:left+length]])
        else:
            sequence = template # sequence can also be directly defined by template arg
        # check for gaps, probes cannot exist in gapped regions!
        if np.any(self.gap_mask[left:left+length]) and (allow_gaps == False):
            raise ValueError('Probes in gapped regions not supported.')
        if np.any(self.gap_mask[left:left+1]) and allow_gaps:
            raise ValueError('Probes *starting* in gapped regions not supported (even if gaps are supported).')
        if left < 0 or left+length > len(self.gap_mask):
            raise ValueError('Probes oustide of aligned regions not supported.')
        if add_to_list == True:
            self.probes.append(Probe(left, length, sequence, self))
            return len(self.probes)-1 # added probe to list, return current index of probe
        else:
            return Probe(left, length, sequence, self) # if not adding to list, return the probe itself
    
    def removeProbe(self, probe_i):
        # remove a probe from the set
        self.probes.pop(probe_i)
        
    def estimateAlignmentInformationContent(self):
        rep_rrna = self.rrna[[np.where(self.species==spec)[0][0] for spec in np.unique(self.species)]]
        rep_nt_array = np.asarray([[nt for nt in rrna] for rrna in rep_rrna]).T
        information = []
        for pos in rep_nt_array:
            if '-' in pos:
                information.append(np.nan) # undefined at position
            else:
                nt_counts = np.unique(pos, return_counts = True)[1]
                p_nt = nt_counts/np.sum(nt_counts).astype(float)
                information.append(2+np.nansum(p_nt*np.log2(p_nt)))
        return np.asarray(information)
    
    def diversifyProbe(self, probe_i, iterations,
                       functions=['extendProbe','shrinkProbe','moveProbe','mutateProbe'], p_values=[1,1,1,1],):
        # now iteratively conduct functions on probes
        function_choices = np.random.choice(functions, p=np.asarray(p_values).astype(float)/np.sum(p_values), size=iterations)
        probes_generated = [self.probes[probe_i]]
        for chosen_function in function_choices:
            chosen_probe = np.random.choice(probes_generated)
            func = getattr(chosen_probe, chosen_function)
            try:
                probes_generated.append(func(add_to_list=False))
            except ValueError:
                pass
        return probes_generated
        
    def getProbeData(self):
        # [[left, length, min(Tm)], ....]
        return np.asarray([[probe.left, probe.length, np.min(probe.getTm())] for probe in self.probes])
    
    def exportProbeData(self, filepath):
        columns = ['left','length','sequence', 'min_tm', 'max_tm'] + list(self.species)
        data = []
        for probe in self.probes:
            Tms = probe.getTm()
            if np.any(np.isnan(Tms)): # pass over any probes which have an undefined Tm
                continue
            row = [probe.left, probe.length, str(Seq(probe.sequence,alphabet=unambiguous_dna).reverse_complement()),
                   np.min(Tms), np.max(Tms)] + Tms
            data.append(row)
        pd.DataFrame(data=data, columns=columns).to_csv(filepath)
    
    def plotStoredProbesAlignment(self, height_ratios = [3,5,2], figwidth=10, figheights=[1.5,3.5], cmap_name = 'winter', c_below='purple', cb_frac_height=.7,
                                 cmap_lim = (50,65), cmap_ticks=[50,55,60,65], window_size=25):
        # store a few things to the object needed to late add probes
        n_steps = len(self.probe_plot_snapshots)
        self.probe_axes = [] # storage for probe axes
        self.colormap = plt.cm.get_cmap(cmap_name)
        self.cmap_calc = lambda x: self.colormap(float(x-cmap_lim[0])/(cmap_lim[1] - cmap_lim[0]))
        self.cmap_lim = cmap_lim
        self.c_below = c_below
        # intitalize figure gridspecs
        fig = plt.figure(figsize=(figwidth, figheights[0]*n_steps+figheights[1]))
        grid = plt.GridSpec(n_steps+2, 1, height_ratios=[height_ratios[0] for i in range(n_steps)] + height_ratios[1:])
        # generate subplots in gridspec blocks
        self.probe_axes.append(plt.subplot(grid[0]))
        for ax_i in range(1,n_steps):
            self.probe_axes.append(plt.subplot(grid[ax_i],sharex=self.probe_axes[0]))
        try:
            align_ax = plt.subplot(grid[ax_i+1],sharex=self.probe_axes[0])
            diversity_ax = plt.subplot(grid[ax_i+2],sharex=self.probe_axes[0])
        except UnboundLocalError: # only one to initialize
            align_ax = plt.subplot(grid[1],sharex=self.probe_axes[0])
            diversity_ax = plt.subplot(grid[2],sharex=self.probe_axes[0])
        # despine axes using seaborne
        [sns.despine(ax=p_ax,left=True,bottom=True) for p_ax in self.probe_axes]
        sns.despine(ax=align_ax,left=True,bottom=True)
        sns.despine(ax=diversity_ax)
        # remove ticks for all but diversity axis
        [p_ax.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False) for p_ax in self.probe_axes]
        align_ax.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        align_ax.invert_yaxis()
        plt.tight_layout(h_pad=0)
        # now plot alignments by iterating through them and doing the thing, only plot one per species!
        species_indexes = [np.where(self.species==spec)[0][0] for spec in np.unique(self.species)]
        for plot_i,rrna_i in enumerate(species_indexes):
            local_gap_mask = self.nt_array[rrna_i] == '-'
            # plot all gapped regions
            for region in masktoregions_single(local_gap_mask):
                align_ax.plot([region[0]-.5,region[1]+.5],[plot_i, plot_i], color='r', lw=2.5, solid_capstyle='butt')
            # plot all ungapped regions
            for region in masktoregions_single(~local_gap_mask):
                align_ax.plot([region[0]-.5,region[1]+.5],[plot_i, plot_i], color='k', lw=1.5, solid_capstyle='butt')
            align_ax.text(-.01*len(self.rrna[0]),plot_i,self.species[rrna_i],
                     horizontalalignment='right',verticalalignment='center',fontsize=10)
        # now plot regions with any gaps behind
        for region in self.gap_regions:
            align_ax.fill_between([region[0]-.5,region[1]+.5],0,len(species_indexes)-1, color='r', lw=0, alpha=.25)
        # now plot information content in a scatter plot
        ic = self.estimateAlignmentInformationContent()
        diversity_ax.scatter(np.arange(len(ic)),ic,s=1,alpha=.15,color='k',rasterized=True)
        # also plot a windowed average
        win_x = np.arange(len(ic)-window_size) + window_size/2
        win_avg = []
        for left in np.arange(len(ic)-window_size):
            window_slice = ic[left:left+window_size]
            if np.sum(np.isnan(window_slice)) < (.5*window_size):
                win_avg.append(np.nanmean(window_slice))
            else:
                win_avg.append(np.nan)
        diversity_ax.plot(win_x,win_avg,color='k',lw=1.5,rasterized=True)
        # put some reasonable limits down
        diversity_ax.set_ylim(0,2.05)
        diversity_ax.set_yticks([0,1,2])
        # place colormap
        norm = colors.Normalize(vmin=cmap_lim[0], vmax=cmap_lim[1])
        alignment_coor = align_ax.get_position()
        colorbar_ax = plt.gcf().add_axes([0.01,alignment_coor.y1,0.02,(alignment_coor.y1-alignment_coor.y0)*cb_frac_height])
        cbb(colorbar_ax, cmap=self.colormap, norm=norm, ticks=cmap_ticks)
        colorbar_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        colorbar_ax.tick_params(axis='y', right=False, labelright=False, left=True, labelleft=True)
        # plot all probe steps
        for probe_axis_i,probe_data in enumerate(self.probe_plot_snapshots):
            probe_data = probe_data[np.argsort(probe_data[:,0])] # sort by left position
            plotted_probes = np.empty((0,2)) # start a list of plotted probes for lane determination
            probe_lanes = np.asarray([]).astype(int) # record lane of previous probes for overlap plotting
            spacer = len(self.rrna[0])/130 # spacer for easy visualization of separation of probes
            # plot each probe
            for pd in probe_data:
                # determine lanes used by overlapping probes
                lanes_used = probe_lanes[argoverlappingregions_single([pd[0],pd[0]+pd[1]], plotted_probes)]
                try:
                    unused_lanes = np.setdiff1d(np.arange(np.max(lanes_used)+1),lanes_used)
                    used_lane = unused_lanes[0] # try to select first member of list as unused lane
                except IndexError: # if all lanes used
                    used_lane = np.max(lanes_used) + 1
                except ValueError: # if no lanes used
                    used_lane = 0
                if np.isnan(pd[2]):
                    # plot ill-defined Tm (np.nan) as dotted lines with greyed out backing
                    self.probe_axes[probe_axis_i].plot([pd[0], pd[0]+pd[1]],[used_lane,used_lane],c='k',alpha=.4,lw=1.5,solid_capstyle='butt')
                elif pd[2] <= self.cmap_lim[0]:
                    self.probe_axes[probe_axis_i].plot([pd[0], pd[0]+pd[1]],[used_lane,used_lane],c=self.c_below,lw=1.5,solid_capstyle='butt')
                else:
                    self.probe_axes[probe_axis_i].plot([pd[0], pd[0]+pd[1]],[used_lane,used_lane],c=self.cmap_calc(pd[2]),lw=1.5,solid_capstyle='butt')
                plotted_probes = np.r_[plotted_probes,[[pd[0],pd[0]+pd[1]+spacer]]]
                probe_lanes = np.r_[probe_lanes,used_lane]
            self.probe_axes[probe_axis_i].axhline(-0.4,1,0,lw=.5,c='k')
            self.probe_axes[probe_axis_i].set_ylim(-1,max(probe_lanes)+1)
        return self.probe_axes, align_ax, diversity_ax 
    
    def snapshotProbes(self):
        # gather probe data, prepare plots
        self.probe_plot_snapshots.append(self.getProbeData())
        
    def snapshotProbeTms(self):
        # record all probe data
        self.probe_tm_snapshots.append(np.asarray([[probe.left, probe.length, np.min(probe.getTm())] for probe in self.probes]))
        
    def __init__(self, fasta_path):
        # read alignment fasta and store results
        rrna, species, nt_array, unique_nt, gap_mask, gap_regions, nogap_regions = self.readfasta(fasta_path)
        self.rrna = rrna
        self.species = species
        self.nt_array = nt_array
        self.unique_nt = unique_nt
        self.gap_mask = gap_mask
        self.gap_regions = gap_regions
        self.nogap_regions = nogap_regions
        # initialize storage and other objects
        self.probes = []
        self.probe_plot_snapshots = []
        self.probe_tm_snapshots = []

class Probe():
    def getTm(self):
        tm_list = []
        for rrna_seq in self.parent_alignment.rrna:
            rna_sequence = rrna_seq[self.left:self.left+self.length]
            if '-' in rna_sequence: # deal with gap if found
                rna_sequence = ''
                position = self.left
                while len(rna_sequence) < self.length:
                    if rrna_seq[position] != '-':
                        rna_sequence += rrna_seq[position]
                    position+=1
            # now calculate it
            tm_list.append(NNTm(self.sequence, rna_sequence))
        return tm_list
    
    def getTm_nogap(self):
        tm_list = [NNTm(self.sequence, rrna_seq[self.left:self.left+self.length]) for rrna_seq in self.parent_alignment.rrna]
        return tm_list
    
    def getMismatches(self):
        # find all positions in probe with mismatches to any aligned rRNA and possible mutations to each site
        possible_mutational_steps = []
        for i,current_nt,nt_choices in zip(np.arange(len(self.sequence)),
                              self.sequence, self.parent_alignment.unique_nt[self.left:self.left+self.length]):
            [possible_mutational_steps.append((i,choice))
             for choice in np.setdiff1d(nt_choices, current_nt)]
        return possible_mutational_steps
    
    def extendProbe(self, distance = [1,2,3,4], p_values = [10,5,2.5,1.25], template = 'single', user_def = None, **kwargs):
        # grow probe in the left or right direction
        # template options are 'single' or 'any'
        # first determine how far to extend, direction of extension
        if user_def is None: 
            p_values = np.asarray(p_values)/np.sum(p_values)
            extend_distance = np.random.choice(distance,p=p_values)
            is_right = np.random.choice([True,False])
        elif user_def is not None: # should be a tuple: (<right/left>, distance)
            if user_def[0] == 'right':
                is_right = True
            elif user_def[0] == 'left':
                is_right = False
            else:
                raise ValueError('input should be a tuple: (<right/left>, distance)')
            extend_distance = user_def[1]
        if is_right: # extend right?
            if template == 'single':
                picked_species = np.random.choice(np.unique(self.parent_alignment.species)) # pick a species from list
                rrna_i = np.random.choice(np.where(self.parent_alignment.species == picked_species)[0]) # rrna index
                new_part = self.parent_alignment.rrna[rrna_i][self.left+self.length:self.left+self.length+extend_distance]
            new_sequence = self.sequence + new_part
            return self.parent_alignment.addProbe(self.left, self.length+extend_distance, new_sequence, **kwargs)
        else: # extend left
            if template == 'single':
                picked_species = np.random.choice(np.unique(self.parent_alignment.species)) # pick a species from list
                rrna_i = np.random.choice(np.where(self.parent_alignment.species == picked_species)[0]) # rrna index
                new_part = self.parent_alignment.rrna[rrna_i][self.left-extend_distance:self.left]
            new_sequence = new_part + self.sequence
            return self.parent_alignment.addProbe(self.left-extend_distance, self.length+extend_distance, new_sequence, **kwargs)
    
    def shrinkProbe(self, distance = [1,2,3,4], p_values = [10,5,2.5,1.25], user_def = None, **kwargs):
        # shrink probe from the left or right side
        # first determine how far to shrink
        if user_def is None:
            p_values = np.asarray(p_values)/np.sum(p_values)
            shrink_distance = np.random.choice(distance,p=p_values)
            is_right = np.random.choice([True,False])
        elif user_def is not None: # should be a tuple: (<right/left>, distance)
            if user_def[0] == 'right':
                is_right = True
            elif user_def[0] == 'left':
                is_right = False
            else:
                raise ValueError('input should be a tuple: (<right/left>, distance)')
            shrink_distance = user_def[1]
        # verify that probe is not shrunk into non-existance (less than 5 nt in length is not allowed!)
        if self.length - shrink_distance < 5:
            raise ValueError('cannot shrink below 5 nt in length.')
        if is_right: # shrink right?
            new_sequence = self.sequence[:-shrink_distance]
            return self.parent_alignment.addProbe(self.left, self.length-shrink_distance, new_sequence, **kwargs )
        else: # shrink left
            new_sequence = self.sequence[shrink_distance:]
            return self.parent_alignment.addProbe(self.left+shrink_distance, self.length-shrink_distance, new_sequence, **kwargs)
        
    def moveProbe(self, distance = [1,2,3,4], p_values = [10,5,2.5,1.25], user_def = None, **kwargs):
        # get move distance and direction
        if user_def is None:
            p_values = np.asarray(p_values)/np.sum(p_values)
            move_distance = np.random.choice(distance,p=p_values)
            is_right = np.random.choice([True,False])
        elif user_def is not None: # should be a tuple: (<right/left>, distance)
            if user_def[0] == 'right':
                is_right = True
            elif user_def[0] == 'left':
                is_right = False
            else:
                raise ValueError('input should be a tuple: (<right/left>, distance)')
            move_distance = user_def[1]
        # grow first, then shrink so we don't have issues with probes which are < 5 nt in length (will raise errors!)
        if is_right:
            intermediate_probe = self.extendProbe(user_def = ('right', move_distance), add_to_list = False)
            return intermediate_probe.shrinkProbe(user_def = ('left', move_distance), **kwargs)
        else:
            intermediate_probe = self.extendProbe(user_def = ('left', move_distance), add_to_list = False)
            return intermediate_probe.shrinkProbe(user_def = ('right', move_distance), **kwargs)
            
    def mutateProbe(self, **kwargs):
        # generate a new probe with a random mutation from choices in getMismatches
        # return index of new probe or np.nan if probe is already a perfect match to all sequences
        possible_steps = self.getMismatches()
        try:
            chosen_step = possible_steps[np.random.choice(np.arange(len(possible_steps)))]
        except ValueError:
            raise ValueError('Probe is perfect match.')
        # now make the sequence edit and generate a new probe
        pos,nt = chosen_step
        new_sequence = self.sequence[:pos] + nt + self.sequence[pos+1:]
        return self.parent_alignment.addProbe(self.left,self.length,new_sequence, **kwargs)
    
    def __init__(self, left, length, sequence, alignment):
        self.left = left
        self.length = length
        self.sequence = sequence
        self.parent_alignment = alignment