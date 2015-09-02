# Copyright (C) 2015 Haruhiko Matsuo <halm.matsuo@gmail.com>
#
#  Distributed under the MIT License.
#  See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT

"""Cache"""

from common import monitor_level_checker
import numpy as np


class Cache(object):
    def __init__(self, profenv, rawdata):
        self.e = profenv
        self.d = rawdata


    def L1Imiss_ratio(self, *tag):
        """ L1I misses / effective instructions"""
        l1imiss = self.d.L1I_miss(*tag)
        ef_inst = self.d.effective_instruction_counts(*tag)
        return l1imiss / ef_inst.astype(float) * 100


    def L1Dmiss_ratio(self, *tag):
        """Calculate L1 data miss::
        L1D miss / load-store operations."""
        simd_ls = self.d.SIMD_load_store_instructions(*tag)
        ls = self.d.load_store_instructions(*tag)
        l1d_miss = self.d.L1D_miss(*tag)
        return l1d_miss / (ls + simd_ls*2).astype(float) * 100


    def L1Dmiss_dm_ratio(self, *tag):
        """L1D miss dm / L1D miss"""
        l1d_miss = self.d.L1D_miss(*tag)
        l1d_miss_dm = self.d.L1D_miss_dm(*tag)
        return l1d_miss_dm / l1d_miss.astype(float) * 100


    def L1Dmiss_hwpf_ratio(self, *tag):
        """L1D miss hwpf / L1D miss"""
        l1d_miss = self.d.L1D_miss(*tag)
        l1d_miss_hwpf = self.d.L1D_miss_hwpf(*tag)
        return l1d_miss_hwpf / l1d_miss.astype(float) * 100


    def L1Dmiss_swpf_ratio(self, *tag):
        """L1D miss swpf / L1D miss"""
        l1d_miss = self.d.L1D_miss(*tag)
        l1d_miss_swpf = self.d.L1D_miss_swpf(*tag)
        return l1d_miss_swpf / l1d_miss.astype(float) * 100


    def L2miss_ratio(self, *tag):
        """L2 miss / load-store"""
        simd_ls = self.d.SIMD_load_store_instructions(*tag)
        ls = self.d.load_store_instructions(*tag)
        l2_miss_dm = self.d.L2_miss_dm(*tag)
        l2_miss_pf = self.d.L2_miss_pf(*tag)
        return (l2_miss_dm + l2_miss_pf) / (ls + simd_ls*2).astype(float) * 100


    def L2miss_dm_ratio(self, *tag):
        """L2 miss dm / L2 miss"""
        l2_miss_dm = self.d.L2_miss_dm(*tag)
        l2_miss_pf = self.d.L2_miss_pf(*tag)
        return l2_miss_dm / (l2_miss_dm + l2_miss_pf).astype(float) * 100


    def L2miss_pf_ratio(self, *tag):
        """L2 miss dm / L2 miss"""
        l2_miss_dm = self.d.L2_miss_dm(*tag)
        l2_miss_pf = self.d.L2_miss_pf(*tag)
        return l2_miss_pf / (l2_miss_dm + l2_miss_pf).astype(float) * 100


    def uDTLBmiss_ratio(self, *tag):
        """uDTLB miss / load-store"""
        simd_ls = self.d.SIMD_load_store_instructions(*tag)
        ls = self.d.load_store_instructions(*tag)
        udtlb_miss = self.d.uDTLB_miss(*tag)
        return udtlb_miss / (ls + simd_ls*2).astype(float) * 100


    def mDTLBmiss_ratio(self, *tag):
        """mDTLB miss / load-store"""
        simd_ls = self.d.SIMD_load_store_instructions(*tag)
        ls = self.d.load_store_instructions(*tag)
        dmmu_miss = self.d.trap_DMMU_miss(*tag)
        return dmmu_miss / (ls + simd_ls*2).astype(float) * 100
