/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(pimreg/cg/lr1,FixPIMRegCGLR1);
// clang-format on
#else

#ifndef LMP_FIX_PIMREGCGLR1_H
#define LMP_FIX_PIMREGCGLR1_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPIMRegCGLR1 : public Fix {
 public:
  FixPIMRegCGLR1(class LAMMPS *, int, char **);
  ~FixPIMRegCGLR1() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void pre_force(int) override;
  double memory_usage() override;

 protected:
  class PairPIM *pim;
  class NeighList *list;
  class Ewald *ewald;

 private:
  int maxatom;
  int maxiteration;
  //int ilevel_respa;

  int pair_compute_flag;      // 0 if pair->compute is skipped
  int kspace_compute_flag;    // 0 if kspace->compute is skipped
  int init_ek_flag = 0;

  char *id_pe;
  class Compute *pe;
  //class Pair *pim;

  double **tempk;
  double **bk;
  double **rk;
  double **rkp1;
  double **pk;
  double **tempf;
  
  double g_ewald;


  double dot_local(double **a, double **b, int maxatom);
  void calculate_dipoles();
  double update_energy();
  void grad_clear(double **);
  void reallocate();

  void printA(PairPIM *pim);
  //void printb();

  void calculate_diff(double **dedmu, PairPIM *pim);
  void calculate_AdotV(double **tempk, double **v, PairPIM *pim);
  void calculate_b(double **tempk, double **v, PairPIM *pim);

  void calculate_AdotV_one(double **dedmu, double **x, double **mu, int i, int *jlist, int jnum, PairPIM *pim);
  void calculate_b_one(double **dedmu, double **x, double **mu, int i, int *jlist, int jnum, PairPIM *pim);

  int pack_forward_comm(int, int *, double *, int, int *) override;
  //int pack_reverse_comm(int, int, double *) override;
  void unpack_forward_comm(int, int, double *) override;
  void init_list(int, class NeighList *) override;
  //void unpack_reverse_comm(int, int *, double *) override;

};

}    // namespace LAMMPS_NS

#endif
#endif
