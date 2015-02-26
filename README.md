# pyproffx

A tool set of python modules for the performance profiler on
Fujitsu's PRIMEHPC FX10 supercomputer

## Install

    $ pip install pyproffx

## Quick guide

First, I strongly recommend that you use this module with
IPython, which has a powerful interactive shell, featured by
tab-completion.

If you have installed pyproffx successfuly, you can import
the module.

    $ ipython
    >>> import pyproffx as pfx

Before loading yor application's profiling results, you may need
the label strings.

    >>> %ls ~/path/to/data/
    output_prof_1.csv    output_prof_2.csv    output_prof_3.csv ...
    >>> fp = '~/path/to/data/'
    >>> pfx.program_info(fp)
    {'labels': ['__for_accumulate_estimates', '__flip_operator_and_spins'],
     'num_procs': 4,
     'num_threads': 16}

Then identify the label you want and give it to the loader
function.

    >>> p, d = pfx.load_pa(fp, '__for_accumulate_estimates')

Now you can access the PC counts mesured by the profiler.
The example below extracts the counts of L2 demand miss of
a thread, which is identified by process ID 1 and thread ID 0.

    >>> d.L2_miss_dm('T', 1, 0)
    array([193983660])

The string 'T' means 'Thread' monitor level, and there are two
more monitor levels: process ('P' or 'Process') and
application ('A' or 'Application').

Note that the type of return value is always numpy.array.


If you want to know which counters are available,
**tab-completion** in IPython is helpful to show you all.

    >>> d.[tab-completion]
    d.L1D_miss        d.Reserved31                   d.cse_window_empty_sp_full   d.end1op
    d.L1D_thrashing   d.Reserved32                   d.cycle_counts1              d.end2op
    d.L1I_miss        d.SIMD_fl_load_instructions    d.cycle_counts2              d.end3op
    d.L1I_thrashing   d.SIMD_fl_store_instructions   d.cycle_counts3              d.eu_comp_wait
    ...

You can get multi-process/multi-thread results by giving the
range of process/thread id.

    >>> d.L2_miss_dm('T', (0, 3), (0, 8))
    array([193234830, 191787314, 191129047, 192478642, 166060260, 191627643,
           ...                                               ..., 193335331)]

Thre are some performance indices computed by using the
combination of the PC counters.

    >>> dperf = pfx.Performance(p, d)
    >>> dperf.elapsed_time('T', 0, (0, 8))
    array([ 62.5973813 ,  62.09905636,  61.46894915,  62.87457782,
            61.68163378,  62.35979694,  60.15607663,  60.92197067])
