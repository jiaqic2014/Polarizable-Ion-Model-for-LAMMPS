/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Jiaqi Chen (jiaqic@usst.edu.cn)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(pim/lr,PairPIMLR)

#else

#ifndef LMP_PAIR_PIMLR_H
#define LMP_PAIR_PIMLR_H

#include "pair.h"

namespace LAMMPS_NS {
class PairPIMLR : public Pair {
 friend class FixPIMRegCG;
 friend class FixPIMRegCGLR;
 friend class FixPIMRegCGLR1;
 friend class FixPIMRegCGLRCUT;
 public:
 //Constructer
  PairPIMLR(class LAMMPS *);
  //Destructer
  ~PairPIMLR() override;
  //Compute the forces on a atom
  void compute(int, int) override;
  //Process the args to pair_style command, must work with allocate()
  void settings(int, char **) override;
  //Process the args to pair_coeff or read from data file
  void coeff(int, char **) override;
  //style initialization: request neighbor list(s), error checks
  //It is unclear to me where this function is called, though it is likely excuted once in the initialization phase
  void init_style() override;
  //I think this one happens after simulation setup and before run is performed. It seems to be used for coefficient setup for the j,i 
  double init_one(int, int) override;

  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;

  //Write diagonal coefficients
  void write_data(FILE *) override;
  //Write all coefficients
  void write_data_all(FILE *) override;

  //called to calculate the force between i,j only. Only applicable if the potentials are not addible. Analyse the potential again to determine. Set flag off is needed. 
  //double single(int, int, int, int, double, double, double, double &) override;

  //Return a general pointer to the call. Expose the internal data to other part of the code. 
  void *extract(const char *, int &) override;

 protected:
  double cut_lj_global;
  //For dispersion and dipole
  double **cut_lj,**cut_ljsq;
  //For coul force
  double cut_coul,cut_coulsq;
  //Coeff in the first layer of RIM
  double **biga,**a,**bigc6,**bigc8;
  //Coeff in the T-T damping for dispersion C6 and C8
  double **b6, **b8;
  //Coeff in the T-T damping for dipole interaction
  double **bd, **cd, **cdinv;
  //Coeff for polarizability alpha, this is supposed to be a per atom type, not a pair thing. Use a 2D array for compatibility. 
  double **alpha;
  //Used in ewald summation. Check later in kSpace
  double g_ewald;
  //used in ewald/disp
  int ewald_order;
  //Allocate memeries. 
  virtual void allocate();
  //Some kind of advanced algorithm support. Not gonna touch. 
  double *cut_respa;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Pair style born/coul/long requires atom attribute q

An atom style that defines this attribute must be used.

E: Pair style requires a KSpace style

No kspace style is defined.

*/
