/* ----------------------------------------------------------------------
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

#include "pair_pim.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairPIM::PairPIM(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = 1;
  dipoleflag = 1;
  vflag_atom = 1;
  ftable = NULL;
  writedata = 1;
  cut_respa = NULL;
}

/* ---------------------------------------------------------------------- */

PairPIM::~PairPIM()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(biga);
    memory->destroy(a);
    memory->destroy(bigc6);
    memory->destroy(bigc8);
    memory->destroy(b6);
    memory->destroy(b8);
    memory->destroy(bd);
    memory->destroy(cd);
    memory->destroy(cdinv);
    memory->destroy(alpha);

  }
  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

void PairPIM::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itable,itype,jtype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul;
  double r,rsq,rinv,r2inv,r3inv,r5inv,r7inv;
  double forceborn,forcecoul;
  double forcecoulx,forcecouly,forcecoulz;
  //Factors to calculate short range coulomb force using Ewald summation
  double grij,expm2,prefactor,t,erfc;
  double b0, b1, b2, b3,pre1,pre2,pre3,pre4;
  //electircal parameters
  double pdotp,pidotr,pjdotr, g0, g1, g2;
  //Parameters in the PIM short range dampening 
  double bdtmp, dgijdr, dgjidr, gdampij, gdampji, edamp;
  double fdamp6, fdamp8, dfdr, btmp;

  //double DDE = 0;
  //double DCE = 0;
  //double DSE = 0;
  //double frepulsion,fadhesion;
  //double *special_coul = force->special_coul;
  //double *special_lj = force->special_lj;
  //double factor_lj, factor_coul;
  

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **mu = atom->mu;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  //I think it is used for special neighbors, where a scaling factor is given. 
  //double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  //Used in kSpace Coul, though it is unclear where in force->newton_pair
  double qqrd2e = force->qqrd2e;
  int debugflag = 0;
  double cdtemp, cdinvtemp;
  //double ecouldiag = 0, evdwldiag = 0;
  
  
  //Number of local atoms, I believe the local here means domain decomposition. 
  inum = list->inum;
  //ID of local atoms
  ilist = list->ilist;
  //Number of neighbours of the atom at hand
  numneigh = list->numneigh;
  //Pointer to the neighbour array
  firstneigh = list->firstneigh;
  
  // loop over neighbors of my atoms
  //i is global ID, ii is local ID, if I understand correctly. 

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      //factor_lj = special_lj[sbmask(j)];  
      // I really doubt if we'll have a factor !=1, this causes so much overhead. 
      //factor_coul = special_coul[sbmask(j)];
      //factor_lj = special_lj[sbmask(j)];
      //factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK; //I think only this one is mandatory. Let's check out. 
      
      //Relative displacement
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      //prevent unintended dividing by zero
      //rsq = MAX(1e-16,rsq);
      jtype = type[j];

      r = sqrt(rsq);
      r2inv = 1.0/rsq;
      rinv = sqrt(r2inv);
      r3inv = r2inv*rinv;
      r5inv = r3inv*r2inv;
      r7inv = r5inv*r2inv;


      if (rsq < cut_coulsq) {

          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;

          pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
          pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
          
          //temp parameters from T-T dampening
          bdtmp = bd[itype][jtype];
          //Note that if bdtmp is zero, we are facing the case where we don't have damping, yet gdampij in the current 


          //We have cdinv[j][i] = cdinv[i][j] to accomodate the symmetric requirement on initialization, which means we have to set cd or cdinv based on the comparison between jtype and itype. 
          //I should figure out a better way if we have more than two atom types, maybe take care of the coefficient initialization?
          if (itype<jtype){
            cdtemp = cd[itype][jtype];
            cdinvtemp = cdinv[itype][jtype];
          } else{
            cdtemp = cdinv[itype][jtype];
            cdinvtemp = cd[itype][jtype];
          }
          dgijdr = bdtmp*cdtemp*pow(bdtmp*r,4)*exp(-bdtmp*r)/24;
          dgjidr = bdtmp*cdinvtemp*pow(bdtmp*r,4)*exp(-bdtmp*r)/24;
          gdampij = 1 - cdtemp*exp(-bdtmp*r)*(1 + bdtmp*r + pow(bdtmp*r,2)/2 + pow(bdtmp*r,3)/6 + pow(bdtmp*r,4)/24);
          gdampji = 1 - cdinvtemp*exp(-bdtmp*r)*(1 + bdtmp*r + pow(bdtmp*r,2)/2 + pow(bdtmp*r,3)/6 + pow(bdtmp*r,4)/24);
          

          //if(debugflag<2){
          //  if(itype<jtype){
          //    auto mesg = fmt::format("[{:1d}][{:1d}]:dgij:{:3e}|dgji:{:3e}|gij:{:3e}|gji:{:3e}|cdij:{:3e}|cdji:{:3e}\n",itype,jtype,dgijdr,dgjidr,gdampij,gdampji,cd[itype][jtype],cdinv[itype][jtype]);
          //    utils::logmesg(lmp,mesg);
          //    debugflag += 1; 
          //  }
          //}
          
          //This is consistent with only q force. We don't do Ewald for dipole, I also believe I forget to add qqrd2e for the force in previous implementation.
          if (qtmp != 0.0 && q[j] != 0.0) {
            prefactor = qtmp*q[j]/r;
            forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            forcecoulx = delx*forcecoul*r2inv;
            forcecouly = dely*forcecoul*r2inv;
            forcecoulz = delz*forcecoul*r2inv;
          }

          if (mu[i][3] > 0.0 && mu[j][3] > 0.0) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            r7inv = r5inv*r2inv;

            pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
            pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
            pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;

            pre1 = 3.0*r5inv*pdotp - 15.0*r7inv*pidotr*pjdotr;
            pre2 = 3.0*r5inv*pjdotr;
            pre3 = 3.0*r5inv*pidotr;
            pre4 = -1.0*r3inv;
            //forcecoulx = delx * (g0*b1 + g1*b2 + g2*b3) - b1 * (qtmp*mu[j][0]*gdampij - q[j]*mu[i][0]*gdampji + qtmp*pjdotr*dgijdr - q[j]*pidotr*dgjidr) + b2 * (pjdotr*mu[i][0] + pidotr*mu[j][0]);
          
            forcecoulx += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0];
            forcecouly += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1];
            forcecoulz += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2];
          }
//
          if (mu[i][3] > 0.0 && q[j] != 0.0) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
            pre1 = 3.0*q[j]*r5inv * pidotr;
            pre2 = q[j]*r3inv;

            forcecoulx += (pre2*mu[i][0] - pre1*delx)*gdampji + (q[j]*pidotr*r3inv)*rinv*dgjidr*delx;
            forcecouly += (pre2*mu[i][1] - pre1*dely)*gdampji + (q[j]*pidotr*r3inv)*rinv*dgjidr*dely;
            forcecoulz += (pre2*mu[i][2] - pre1*delz)*gdampji + (q[j]*pidotr*r3inv)*rinv*dgjidr*delz;
          }
//
          if (mu[j][3] > 0.0 && qtmp != 0.0) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
            pre1 = 3.0*qtmp*r5inv * pjdotr;
            //add damping term
            pre2 = qtmp*r3inv;

            forcecoulx += (pre1*delx - pre2*mu[j][0])*gdampij - (qtmp*pjdotr*r3inv)*rinv*dgijdr*delx;
            forcecouly += (pre1*dely - pre2*mu[j][1])*gdampij - (qtmp*pjdotr*r3inv)*rinv*dgijdr*dely;
            forcecoulz += (pre1*delz - pre2*mu[j][2])*gdampij - (qtmp*pjdotr*r3inv)*rinv*dgijdr*delz;
          }

          g0 = qtmp*q[j];
          b0 = erfc * rinv;

      } else{
        forcecoulx = forcecouly = forcecoulz = 0.0;
      }//Finishing calculating coulomb force, don't forget the contribution of self energy of dipoles when doing minimization. But we can't add that here as this is updating pair-wise energy. I'll figure out where to do that later. 
        
      //Short range forces
      
      if (rsq < cut_ljsq[itype][jtype]) {
        //here we use force born factor, take out the deltar
        //forceborn = 0.01;
        forceborn = biga[itype][jtype]*a[itype][jtype]*exp(-a[itype][jtype]*r)*rinv;
        btmp = b6[itype][jtype];
        fdamp6 = 1 - exp(-btmp*r) * (1 + r*btmp + pow(r*btmp,2)/2.0 + pow(r*btmp,3)/6.0 + pow(r*btmp,4)/24.0 + pow(r*btmp,5)/120.0 + pow(r*btmp,6)/720.0);
        
        //fdamp6 = 1;
        dfdr = btmp*exp(-btmp*r)*pow(r*btmp,6)/720.0;
        //dfdr = 0;
        forceborn += (dfdr*bigc6[itype][jtype]*r5inv*r2inv - fdamp6*6*bigc6[itype][jtype]*r5inv*r3inv);
        
        btmp = b8[itype][jtype];
        fdamp8 = 1 - exp(-btmp*r) * (1 + r*btmp + pow(r*btmp,2)/2.0 + pow(r*btmp,3)/6.0 + pow(r*btmp,4)/24.0 + pow(r*btmp,5)/120.0 + pow(r*btmp,6)/720.0 + pow(r*btmp,7)/5040.0 + pow(r*btmp,8)/40320.0);
        //fdamp8 = 1;
        dfdr = btmp*exp(-btmp*r)*pow(r*btmp,8)/40320.0;
        //dfdr = 0;
        forceborn += (dfdr*bigc8[itype][jtype]*r5inv*r2inv*r2inv - fdamp8*8*bigc8[itype][jtype]*r5inv*r5inv);
      } else forceborn = 0.0;
        
      //forceborn = 0;
      //forcecoulx = 0;
      //forcecouly = 0;
      //forcecoulz = 0;

      f[i][0] += delx*forceborn + qqrd2e*forcecoulx;
      f[i][1] += dely*forceborn + qqrd2e*forcecouly;
      f[i][2] += delz*forceborn + qqrd2e*forcecoulz;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delx*forceborn + qqrd2e*forcecoulx;
        f[j][1] -= dely*forceborn + qqrd2e*forcecouly;
        f[j][2] -= delz*forceborn + qqrd2e*forcecoulz;
      }

      if (eflag) {
        if (rsq < cut_coulsq) {
          //The real part summation of coul energy. 
          //qqrd2e is a conversion, the b0*g0 has the dimension of e*e/L, while the conversion goes to E/epsilon
          ecoul = b0*g0;
          //note that m[i][3] is not updated automatically, don't use it other than check if it exist

          if (mu[i][3] > 0.0 && mu[j][3] > 0.0){
            ecoul += r3inv*pdotp - 3.0*r5inv*pidotr*pjdotr;
            //DDE += r3inv*pdotp - 3.0*r5inv*pidotr*pjdotr;
          }
            
            
          if (mu[i][3] > 0.0 && q[j] != 0.0){
            ecoul += -q[j]*r3inv*pidotr*gdampji;
            //DCE += -q[j]*r3inv*pidotr*gdampji;
          }
            
            
          if (mu[j][3] > 0.0 && qtmp != 0.0){
            ecoul += qtmp*r3inv*pjdotr*gdampij;
            //DCE += qtmp*r3inv*pjdotr*gdampij;
          }
          ecoul *= qqrd2e;
          //dipoleEnergy *= qqrd2e;
        } else ecoul = 0.0; //remember this is only the interaction part. 
        //updating short range energy.

        if (rsq < cut_ljsq[itype][jtype]) {
          evdwl = biga[itype][jtype]*exp(-a[itype][jtype]*r) - fdamp6*bigc6[itype][jtype]*r3inv*r3inv - fdamp8*bigc8[itype][jtype]*r5inv*r3inv;
        } else evdwl = 0.0;
      }

      //evdwl = 0;
      //ecoul = 0;
        //updating virial, I doubt if I'll ever use this
        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
                             evdwl,ecoul,delx*forceborn + qqrd2e*forcecoulx,dely*forceborn + qqrd2e*forcecouly,delz*forceborn + qqrd2e*forcecoulz,delx,dely,delz);
      }
      //Add dipole self energy, which is not a pair-wise energy, but I'm not sure if there is a better place for it. 
      //There is a dipole self energy in kspace calculation. That kspace can't be used with charge
      if (evflag) {
        if (eflag_either) {
          if (eflag_global) {
            if (i < nlocal) {
              
              eng_coul += 0.5*(mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2])/alpha[itype][itype]*qqrd2e;
              //DSE += 0.5*(mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2])/alpha[itype][itype];
              }
          }
       }
      }
    }
    //DSE *= qqrd2e;
    //DDE *= qqrd2e;
    //DCE *= qqrd2e;
    //auto mesg = fmt::format("PIMPair Diagnostic: Coul E:{:e}|VDWL E:{:e}|DDE:{:e}|DCE:{:e}|DSE:{:e}\n",eng_coul + force->kspace->energy, eng_vdwl,DDE,DCE,DSE);
    //utils::logmesg(lmp,mesg);
    if (vflag_fdotr) virial_fdotr_compute();
  }


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairPIM::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(biga,n+1,n+1,"pair:biga");
  memory->create(a,n+1,n+1,"pair:a");
  memory->create(bigc6,n+1,n+1,"pair:bigc6");
  memory->create(bigc8,n+1,n+1,"pair:bigc8");
  memory->create(b6,n+1,n+1,"pair:b6");
  memory->create(b8,n+1,n+1,"pair:b8");
  memory->create(bd,n+1,n+1,"pair:bd");
  memory->create(cd,n+1,n+1,"pair:cd");
  memory->create(cdinv,n+1,n+1,"pair:cd");
  memory->create(alpha,n+1,n+1,"pair:alpha");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPIM::settings(int narg, char **arg)
{
  //what option do we want? we want two cutoff range, this is used in pair_style args for global settings
  //set cutoffs
  if (narg < 1 || narg > 2) error->all(FLERR,"Illegal pair_style command");

  cut_lj_global = utils::numeric(FLERR,arg[0],false,lmp);;
  if (narg == 1) cut_coul = cut_lj_global;
  else cut_coul = utils::numeric(FLERR,arg[1],false,lmp);;

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPIM::coeff(int narg, char **arg)
{
  //The arg we have are a, biga, b6, b8, bd, cd, bigc6, bigc8, alpha, cut_lj
  //There could also be a pair-wise cut_lj parameter. 
  //Note that cut_coul is not pair-wise, there is only a global parameter set using pair_style args
  if (narg < 12 || narg > 13) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double a_one = utils::numeric(FLERR,arg[2],false,lmp);
  double biga_one = utils::numeric(FLERR,arg[3],false,lmp);
  double b6_one = utils::numeric(FLERR,arg[4],false,lmp);
  double b8_one = utils::numeric(FLERR,arg[5],false,lmp);
  double bd_one = utils::numeric(FLERR,arg[6],false,lmp);
  double cd_one = utils::numeric(FLERR,arg[7],false,lmp);
  double cdinv_one = utils::numeric(FLERR,arg[8],false,lmp);
  double bigc6_one = utils::numeric(FLERR,arg[9],false,lmp);
  double bigc8_one = utils::numeric(FLERR,arg[10],false,lmp);
  double alpha_one = utils::numeric(FLERR,arg[11],false,lmp);
  double cut_lj_one = cut_lj_global;
  //int debugflag = 0;
  if (narg == 13) cut_lj_one =  utils::numeric(FLERR,arg[12],false,lmp);

  //auto mesg = fmt::format("[{:1d}][{:1d}] Set:cdij:{:3e}|cdji:{:3e}\n",i,j,cd[itype][jtype],cd[jtype][itype]);
  //auto mesg = fmt::format("ilow:[{:1d}]|ihigh:[{:1d}]|jlow:[{:1d}]|jhigh:[{:1d}]|",ilo,ihi,jlo,jhi);
  //utils::logmesg(lmp,mesg);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
    //for (int j = jlo; j <= jhi; j++) {
      a[i][j] = a_one;
      biga[i][j] = biga_one;
      b6[i][j] = b6_one;
      b8[i][j] = b8_one;
      bd[i][j] = bd_one;
      cd[i][j] = cd_one;
      cdinv[i][j] = cdinv_one;
      bigc6[i][j] = bigc6_one;
      bigc8[i][j] = bigc8_one;
      cut_lj[i][j] = cut_lj_one;
      if (i!=j) {
        alpha[i][j] = 0;}
      else {
        alpha[i][j] = alpha_one;
      }
      setflag[i][j] = 1;
      count++;
      //auto mesg = fmt::format("[{:1d}][{:1d}] Set:cdij:{:3e}|cdji:{:3e}\n",i,j,cd[itype][jtype],cd[jtype][itype]);
      //utils::logmesg(lmp,mesg);
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPIM::init_one(int i, int j)
{ 
  //or we could use mix rule to set ij, for our case we couldn't
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  double cut = MAX(cut_lj[i][j],cut_coul);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];
  cut_ljsq[j][i] = cut_ljsq[i][j];
  a[j][i] = a[i][j];
  biga[j][i] = biga[i][j];
  b6[j][i] = b6[i][j]; 
  b8[j][i] = b8[i][j]; 
  bd[j][i] = bd[i][j]; 
  cd[j][i] = cd[i][j];
  cdinv[j][i] = cdinv[i][j];
  bigc6[j][i] = bigc6[i][j];
  bigc8[j][i] = bigc8[i][j];
  alpha[j][i] = alpha[i][j];
  setflag[j][i] = 1;
  

  return cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairPIM::init_style()
{
  if (!atom->q_flag || !atom->mu_flag)
    error->all(FLERR,"Pair style PIM requires atom attribute q, mu");

  cut_coulsq = cut_coul * cut_coul;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;
  neighbor->request(this,instance_me);
  //neighbor->add_request(this, NeighConst::REQ_FULL);

}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPIM::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a[i][j],sizeof(double),1,fp);
        fwrite(&biga[i][j],sizeof(double),1,fp);
        fwrite(&b6[i][j],sizeof(double),1,fp);
        fwrite(&b8[i][j],sizeof(double),1,fp);
        fwrite(&bd[i][j],sizeof(double),1,fp);
        fwrite(&cd[i][j],sizeof(double),1,fp);
        fwrite(&cdinv[i][j],sizeof(double),1,fp);
        fwrite(&bigc6[i][j],sizeof(double),1,fp);
        fwrite(&bigc8[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
        fwrite(&alpha[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPIM::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,NULL,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&a[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&biga[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&b6[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&b8[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&bd[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cd[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cdinv[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&bigc6[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&bigc8[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&cut_lj[i][j],sizeof(double),1,fp,NULL,error);
          utils::sfread(FLERR,&alpha[i][j],sizeof(double),1,fp,NULL,error);

        }
        MPI_Bcast(&a[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&biga[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b6[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b8[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&bd[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cd[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cdinv[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&bigc6[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&bigc8[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&alpha[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPIM::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  //fwrite(&offset_flag,sizeof(int),1,fp);
  //fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPIM::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_lj_global,sizeof(double),1,fp,NULL,error);
    utils::sfread(FLERR,&cut_coul,sizeof(double),1,fp,NULL,error);
    //utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,NULL,error);
    //utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,NULL,error);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  //MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  //MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairPIM::write_data(FILE *fp)
{ 
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g %g %g %g %g %g\n",i,
            a[i][i],biga[i][i],b6[i][i],b8[i][i],bd[i][i],cd[i][i], cdinv[i][i], bigc6[i][i], bigc8[i][i], alpha[i][i], cut_lj[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairPIM::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g %g %g %g %g %g\n",i,j,
              a[i][j],biga[i][j],b6[i][j],b8[i][j],bd[i][j],cd[i][j],cdinv[i][j],bigc6[i][j], bigc8[i][j], alpha[i][j], cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

void *PairPIM::extract(const char *str, int &dim)
{ //give other functions access to internal data. 
  //Maybe needed for energy minimization
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  return NULL;
}
