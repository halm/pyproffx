# Copyright (C) 2015 Haruhiko Matsuo <halm.matsuo@gmail.com>
#
#  Distributed under the MIT License.
#  See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT

"""Performance"""

from common import monitor_level_checker
import numpy as np


class Performance(object):
    def __init__(self, profenv, rawdata):
        self.e = profenv
        self.d = rawdata
        max_cycle = {}

        self.max_cycle_counts1_per_process = []
        if self.d.is_hybrid or self.d.is_hybrid:
            for i in range(self.d.num_processes):
                counts = self.d.cycle_counts1('Thread', i, (0, self.d.num_threads))
                self.max_cycle_counts1_per_process.append(counts.max())

        self.max_cycle_counts1_per_application = []
        if self.d.is_hybrid:
            _tmp = self.max_cycle_counts1_per_process
            self.max_cycle_counts1_per_application = max(_tmp)
        elif self.d.is_thread:
            _tmp = self.d.cycle_counts1('Thread', 0, (0, self.d.num_threads))
            self.max_cycle_counts1_per_application = max(_tmp)
        elif self.d.is_flatmpi:
            counts = self.d.cycle_counts1('Process', (0, self.d.num_processes))
            self.max_cycle_counts1_per_application.append(counts.max())

    def max_cycle_counts(self, *tag):
        """ """
        if tag[0][0] == 'T':
            if self.d.is_hybrid:
                max_val = self.d.cycle_counts1(*tag)
            elif self.d.is_thread:
                max_val = self.d.cycle_counts1(*tag)
        elif tag[0][0] == 'P':
            if self.d.is_hybrid:
                proc_id = tag[1]
                if hasattr(proc_id, '__iter__'):
                    pid_itr = pid_itr = (i for i in xrange(proc_id[0], proc_id[1]))
                else:
                    pid_itr = (proc_id,)
                max_val = [self.max_cycle_counts1_per_process[i] for i in pid_itr]
            elif self.d.is_flatmpi:
                max_val = self.d.cycle_counts1(*tag)
        elif tag[0][0] == 'A':
            if self.d.is_hybrid:
                max_val = max(self.max_cycle_counts1_per_process)
            elif self.d.is_thread:
                max_val = max(self.d.cycle_counts1())
            elif self.d.is_flatmpi:
                max_val = self.max_cycle_count1_per_application
            elif self.d.is_single:
                max_val = self.d.cycle_counts1(*tag)
        return np.array(max_val)

    def num_floating_ops(self, *tag):
        """Sum of the number of floating operations"""
        f = self.d.floating_instructions(*tag)
        fma = self.d.fma_instructions(*tag)
        simd_f = self.d.SIMD_floating_instructions(*tag)
        simd_fma = self.d.SIMD_fma_instructions(*tag)
        return f + fma*2 + simd_f*2 + simd_fma*4

    def mflops(self, *tag):
        """MFLOPS"""
        flop = self.num_floating_ops(*tag)
        cycle = self.max_cycle_counts(*tag)
        cpu_clock = self.e[0][2]
        return flop / cycle.astype(float) * cpu_clock

    def mips(self, *tag):
        """MIPS"""
        cycle = self.max_cycle_counts(*tag)
        ef_inst = self.d.effective_instruction_counts(*tag)
        cpu_clock = self.e[0][2]
        return ef_inst / cycle.astype(float) * cpu_clock

    def memory_throughput(self, *tag):
        """Memory throughput [GB/sec]"""
        L2_miss = self.d.L2_miss_dm(*tag) + self.d.L2_miss_pf(*tag)
        L2_wb = self.d.L2_wb_dm(*tag) + self.d.L2_wb_pf(*tag)
        cpu_clock = self.e[0][2]
        line_size = 128  # [Byte] L2 cache line
        trans_byte = (L2_miss + L2_wb) * line_size
        cycle = self.max_cycle_counts(*tag)
        return trans_byte / cycle.astype(float) * cpu_clock / 1.0e3

    def L2_throughput(self, *tag):
        """L2 throughput [GB/sec]"""
        L1D_miss = self.d.L1D_miss(*tag)
        cpu_clock = self.e[0][2]
        line_size = 128  # [Byte] L1 cache line
        trans_byte = L1D_miss * line_size
        cycle = self.max_cycle_counts(*tag)
        return trans_byte / cycle.astype(float) * cpu_clock / 1.0e3
