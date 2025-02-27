/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Jiaqi Chen (jiaqic@usst.edu.cn)
------------------------------------------------------------------------- */

#include "fix_pimreg_cg_lr1.h"
#include "pair_pim.h"

#include "neighbor.h"
#include "ewald_cg.h"
#include "ewald.h"
#include "neigh_list.h"
#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "compute.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "respa.h"
#include "update.h"
#include "math_const.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

//this fix is used to perform energy minimization with respect to dipole moment for PIM model
//This one uses gradient descent method, not very effective and seems to experience numerical problem at times. 
/* ---------------------------------------------------------------------- */

FixPIMRegCGLR1::FixPIMRegCGLR1(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), id_pe(nullptr), pe(nullptr), rk(nullptr), 
    tempk(nullptr), pk(nullptr), pim(nullptr), ewald(nullptr), rkp1(nullptr), bk(nullptr), tempf(nullptr)
{
  if (narg < 3|| narg >5) error->all(FLERR, "Illegal fix pimreg command");
  peratom_flag = 1;
  if (narg < 4) {
    nevery = 1;
  }
  else {
    nevery = utils::inumeric(FLERR, arg[3], false, lmp);
    if (narg==5){
      maxiteration = utils::inumeric(FLERR, arg[4], false, lmp);
    } else {
      maxiteration = 20;
      utils::logmesg(lmp,"Default Maximum iteration\n");
    }
  }
  peratom_freq = nevery;
  size_peratom_cols = 3;
  if (nevery <= 0) error->all(FLERR, "Illegal fix pimreg command");
  std::string cmd = id + std::string("_pe");
  id_pe = utils::strdup(cmd);
  cmd += " all pe";
  modify->add_compute(cmd);

  maxatom = 0;

  reallocate();
  grad_clear(tempk);
  grad_clear(rk);
  grad_clear(rkp1);
  grad_clear(pk);
  grad_clear(bk);
  grad_clear(tempf);

  //check, this may be insufficient
  comm_forward = 3;
  //comm_reverse = 3;
}

/* ---------------------------------------------------------------------- */

FixPIMRegCGLR1::~FixPIMRegCGLR1()
{
  memory->destroy(tempk);
  memory->destroy(bk);
  memory->destroy(rk);
  memory->destroy(rkp1);
  memory->destroy(pk);
  memory->destroy(tempf);
  modify->delete_compute(id_pe);
  delete[] id_pe;
}
/* -----------------------------------------------------------------
This is neighbor list call back from add_neighbor. list points to the requested neighbor list. 
------------------------------------------------------------------ */

void FixPIMRegCGLR1::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixPIMRegCGLR1::init()
{
  // require consecutive atom IDs

  if (!atom->tag_enable || !atom->tag_consecutive())
    error->all(FLERR, "Fix pimreg requires consecutive atom IDs");

  // check for PE compute

  pe = modify->get_compute_by_id(id_pe);
  if (!pe) error->all(FLERR, "PE compute ID for fix pimreg does not exist");

  if (force->pair && force->pair->compute_flag)
    pair_compute_flag = 1;
  else
    pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag)
    kspace_compute_flag = 1;
  else
    kspace_compute_flag = 0;

  // Get a pointer to the PairPIM class to obtain pair-wise coefficient
  pim = dynamic_cast<PairPIM *>(force->pair_match("^pim",0));
  if (pim == nullptr)
  error->all(FLERR,"Must use pair_style pim with fix pimreg/cg/lr");

  //This shit doesn't seem to work as I expected. I couldn't call method in this ewald function. 

  // Get a pointer to the Ewald class to obtain long range coulombic
  ewald = dynamic_cast<Ewald *>(force->kspace_match("^ewald",0));
  if (ewald == nullptr)
  error->all(FLERR,"Must use kspace_style ewald with fix pimreg/cg/lr");

  g_ewald = force->kspace->g_ewald;

  // Add neighbor request, we use full neighbor list for this fix
  neighbor->add_request(this,NeighConst::REQ_FULL);

}

/* ---------------------------------------------------------------------- */

void FixPIMRegCGLR1::setup(int vflag)
{ 
  pre_force(vflag);
  init_ek_flag = 1;
}

/* ----------------------------------------------------------------------------
This setup the time when this fix is called. 
----------------------------------------------------------------------------- */

int FixPIMRegCGLR1::setmask()
{
  datamask_read = datamask_modify = 0;
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPIMRegCGLR1::pre_force(int /* vflag */)
{
  if (update->ntimestep % nevery) return;
  //add compute request for pe for the next timestep
  pe->addstep(update->ntimestep+1);
  calculate_dipoles();
}

/* ----------------------------------------------------------------------
   compute and set minimum dipoles
------------------------------------------------------------------------- */

void FixPIMRegCGLR1::calculate_dipoles()
{
  int i, j, inum, idim, itype, jnum;
  double energy, energym1, energyDiag;
  double relativeE = 1;
  double tempdot,tempdot1;
  
  double **mu=atom->mu;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int nmax = atom->nmax;
  int *type = atom->type;
  double alpha_n,beta_n;

  // grow arrays if necessary, I think we build arrays larger than needed, which is fine by me, just make sure the additional elements are zeros. 
  if (atom->nlocal + atom->nghost > maxatom) reallocate();

  grad_clear(tempk);
  grad_clear(bk);
  grad_clear(rk);
  grad_clear(rkp1);
  grad_clear(pk);
  grad_clear(tempf);

  //in the EF(LR) version, b is the electrical field from test dipoles * alpha, AdotV is the test dipoles
  calculate_AdotV(tempk,mu,pim);
  calculate_b(bk,mu,pim);

  //note that whenever update energy is called, the force is counted once, we need to restore it. 
  for (i = 0; i < nmax; i++)
    for (j = 0; j < 3; j++) {
      tempf[i][j] = f[i][j];
    }

  //Initialize rk and pk for the CG method. 
  for (i = 0; i < nlocal; i++) {
    for(idim=0; idim<3; idim++){
      //set rk = b - A*mu to be consistent with the algorithm
      rk[i][idim] = bk[i][idim] - tempk[i][idim];
      //initialize pk to rk
      pk[i][idim] = rk[i][idim];
    }
    //if(comm->me==0){
    //  auto mesg = fmt::format("rk[{:d}]:{:e},{:e},{:e}|mu[i][3]={:e}\n",i,rk[i][0],rk[i][1],rk[i][2],mu[i][3]);
    //  utils::logmesg(lmp,mesg);
    //}
  }

  int iter = 0;
  //int *sametag = atom->sametag;

  //find the energy minimization dipole set
  maxatom = atom->nmax;
  //while((relativeE>1e-5||iter<5) && iter<maxiteration){
  while(iter<maxiteration){
    
    iter++;

    //energy = update_energy();
    //energym1 = energy;
    //energyDiag = 0.5*dot_local(tempk,mu,nlocal);
    //calculate_b(bk,mu,pim);
    ////only test it serially, we need add MPI_AllReduce to add on the energyDiag if run in parallel
    //energyDiag -= dot_local(bk,mu,nlocal);
    //energyDiag *= force->qqrd2e;
    //
    //if(comm->me==0){
    //  auto mesg = fmt::format("Analytical Energy = {:e}|C-D:{:e}|D-D:{:e}\n",energyDiag,dot_local(bk,mu,nlocal)*force->qqrd2e,0.5*dot_local(tempk,mu,nlocal)*force->qqrd2e);
    //  utils::logmesg(lmp,mesg);
    //}
    //check between nlocal and nmax
    //rk*rk
    tempdot = dot_local(rk,rk,nlocal);
    calculate_AdotV(tempk,pk,pim);
    //pk*A*pk
    tempdot1 = dot_local(tempk,pk,nlocal);
    alpha_n = tempdot/tempdot1;

    //if(comm->me==0){
    //  auto mesg = fmt::format("CG Diag|rk*rk:{:e},pk*A*pk:{:e}|alpha_n:{:e}\n",tempdot,tempdot1,alpha_n);
    //  utils::logmesg(lmp,mesg);
    //}
    
    //update muk and rk
    if (std::isnan(alpha_n)){
      auto mesg = fmt::format("Warning, Nan alpha_n found\n");
      utils::logmesg(lmp,mesg);
      break;
    } else {
      for (i = 0; i < nlocal; i++) {
        j = i;
        for(idim=0; idim<3; idim++){
          mu[i][idim] = mu[i][idim] + alpha_n*pk[i][idim];
          rkp1[i][idim] = rk[i][idim];
          rk[i][idim] = rk[i][idim] - alpha_n*tempk[i][idim];
      }
      ////updating ghost atoms
      int *sametag = atom->sametag;
      
      while (sametag[j] >= 0) {
        j = sametag[j];
        for(idim=0; idim<3; idim++){
            mu[j][idim] = mu[i][idim];
        }
      }
      }
    }
    

    //tempdot = dot_local(rk,rkp1,nlocal); //Diagnostic, rk should be perpendicular with rkp1
    //
    //if(comm->me==0){
    //  auto mesg = fmt::format("PIMReg Diagnostic: rk*rkp1 = {:e}\n",tempdot);
    //  utils::logmesg(lmp,mesg);
    //}
    //
    //energy = update_energy(); //updating energy under the new configuration. 
//
    //calculate_AdotV(tempk,mu,pim);
    //energyDiag = 0.5*dot_local(tempk,mu,nlocal);
    //calculate_b(bk,mu,pim);
//
    //energyDiag -= dot_local(bk,mu,nlocal); //only test it serially, we need add MPI_AllReduce to add on the energyDiag if run in parallel
    //energyDiag *= force->qqrd2e;
    //if(comm->me==0){
    //  auto mesg = fmt::format("Analytical Energy = {:e}|C-D:{:e}|D-D:{:e}\n",energyDiag,dot_local(bk,mu,nlocal)*force->qqrd2e,0.5*dot_local(tempk,mu,nlocal)*force->qqrd2e);
    //  utils::logmesg(lmp,mesg);
    //}
//
    //relativeE = energy-energym1;
    //relativeE = relativeE/energy;
    //if (relativeE<0){
    //  relativeE = -relativeE;
    //}
//
    //if(comm->me==0){
    //  auto mesg = fmt::format("PIMReg Diagnostic: Timestep:{:4d}|Iteration:{:4d}|rk2:{:3e}|Energy:{:3e}|Relative:{:3e}\n",update->ntimestep,iter,tempdot,energy,relativeE);
    //  utils::logmesg(lmp,mesg);
    //}

    //update pk as needed.
    tempdot1 = dot_local(rkp1,rkp1,nlocal);
    tempdot = dot_local(rk,rk,nlocal);
    beta_n = tempdot/tempdot1;
    
    for (i = 0; i < nlocal; i++) {
      for(idim=0; idim<3; idim++){
        pk[i][idim] = rk[i][idim] + beta_n*pk[i][idim];
      }
    }
  }
  //printA(pim);
  for (i = 0; i < nmax; i++)
    for (j = 0; j < 3; j++) {
      f[i][j] = tempf[i][j];
    }
  return;
}

/* ----------------------------------------------------------------------
   calculate the temp = b at the local node
---------------------------------------------------------------------- */

void FixPIMRegCGLR1::calculate_b(double **b, double **v, PairPIM *pim)
{
  int ii, i, j, inum, jnum, jj, itype, jtype, idim;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *mask = atom->mask;
  int *type = atom->type;
  double **x = atom->x;
  double *q = atom->q;
  double **ek = ewald->ek;
  double **cdinv = pim->cdinv;
  double **cd = pim->cd;
  double **bd = pim->bd;

  double tempSum;
  double xtmp,ytmp,ztmp;
  double delx,dely,delz;
  double rsq, r, r3inv,r5inv;
  double cdinvtemp, gdampji, bdtmp;
  double alpha_i;
  double vjdotr;

  double grij,expm2,prefactor,t,erfc;
  

  //Number of local atoms, I believe the local here means domain decomposition. 
  inum = list->inum;
  //ID of local atoms
  ilist = list->ilist;
  //Number of neighbours of the atom at hand
  numneigh = list->numneigh;
  //Pointer to the neighbour array
  firstneigh = list->firstneigh;

  if(!init_ek_flag) return;
  //kspace ewald is written in a way that (404,404) only works to evaluate ek, this is dumb.
  //I need to refer to c++ books later to figure out how to overcome this without modifying other dependencies
  ewald->compute_field();

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    //calculate_b_one(bk,x,mu,i,jlist,jnum,pim);
    for (idim = 0; idim < 3; idim++){ 
      //Consider the vector b[i,m], which is a summation over j, we are setting the element [i,m] over m(idim)
      tempSum = 0;
      for(jj=0; jj<jnum; jj++){ 

        j = jlist[jj];
        j &= NEIGHMASK;
        jtype = type[j];

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        //rsq = MAX(rsq,1e-20);

        if (rsq < (pim->cut_coulsq)){
          r = sqrt(rsq);
          r3inv = (1/rsq)/r;
          //We have cdinv[j][i] = cdinv[i][j] to accomodate the symmetric requirement on initialization, which means we have to set cd or cdinv based on the comparison between jtype and itype. 
          if (itype<jtype){
            cdinvtemp = cdinv[itype][jtype];
          } else{
            cdinvtemp = cd[itype][jtype];
          }
          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;

          bdtmp = bd[itype][jtype];
          gdampji = 1 - cdinvtemp*exp(-bdtmp*r)*(1 + bdtmp*r + pow(bdtmp*r,2)/2 + pow(bdtmp*r,3)/6 + pow(bdtmp*r,4)/24);

          //note that the both terms actually contribute, and they are identical. I made a mistake in previous derivation.
          if (idim==0){
            tempSum +=(q[j]*r3inv*gdampji)*delx*(erfc + EWALD_F*grij*expm2);
          } else if(idim ==1){
            tempSum +=(q[j]*r3inv*gdampji)*dely*(erfc + EWALD_F*grij*expm2);
          } else {
            tempSum +=(q[j]*r3inv*gdampji)*delz*(erfc + EWALD_F*grij*expm2);
          }
        }
      }
      
      if (std::isnan(ek[i][idim])){
        b[i][idim] = tempSum;
        auto mesg = fmt::format("Warning, Nan ek found\n");
        utils::logmesg(lmp,mesg);
      } else{
        b[i][idim] = tempSum + ek[i][idim];
      }
      
    }
  }
  return;
}

/* ----------------------------------------------------------------------
   Dummy conversion at the local node
---------------------------------------------------------------------- */

void FixPIMRegCGLR1::calculate_AdotV(double **Adotv, double **v, PairPIM *pim)
{

  int ii, i, inum, jnum,jj, itype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *mask = atom->mask;
  double **x = atom->x;
  double tempSum;
  int j;
  double xtmp,ytmp,ztmp;
  double delx,dely,delz;
  double rsq, r, r3inv,r5inv;
  double alpha_i;
  double vjdotr;

  //Number of local atoms, I believe the local here means domain decomposition. 
  inum = list->inum;
  //ID of local atoms
  ilist = list->ilist;
  //Number of neighbours of the atom at hand
  numneigh = list->numneigh;
  //Pointer to the neighbour array
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    //Consider the product A[i,m,j,n]*v[j,n], this is getting the element [j,m], iterating over i
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    itype = atom->type[i];
    alpha_i = pim->alpha[itype][itype];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    //we expect to calculate the vector component i, with the neighbor list for atom i
    for (int idim = 0; idim < 3; idim++){ 
      //Consider the product A[i,m,j,n]*v[j,n], this is getting the element [j,m], iterating over m
      tempSum = 0;
      for(int jj=0; jj<jnum; jj++) { 
        //Consider the product A[i,m,j,n]*v[j,n], this is the summation over j
        j = jlist[jj];
        j &= NEIGHMASK;
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < (pim->cut_coulsq)){
          r = sqrt(rsq);
          r3inv = (1/rsq)/r;
          r5inv = r3inv/rsq;
          vjdotr = delx*v[j][0] + dely*v[j][1] + delz*v[j][2];
          //Consider the product A[i,m,j,n]*v[j,n], this is the summation over n, m is idim
          if (idim == 0){
            tempSum += (v[j][idim]*r3inv - 3.0*delx*r5inv*vjdotr);
          } else if (idim == 1){
            tempSum += (v[j][idim]*r3inv - 3.0*dely*r5inv*vjdotr);
          } else {
            tempSum += (v[j][idim]*r3inv - 3.0*delz*r5inv*vjdotr);
          }
        }
      }
      //special case of j = i, which is not considered within the neighbor list
      tempSum += 1/alpha_i*v[i][idim];
      //A*mu
      Adotv[i][idim] = tempSum;
    }
  }
  return;
}

/* ----------------------------------------------------------------------
   calculate dot of the vectors
---------------------------------------------------------------------- */
double FixPIMRegCGLR1::dot_local(double **a, double **b, int maxatom)
{ 
  //different version, only use the neighbor list atoms
  int i;
  double tempsum = 0;
  for (i = 0; i < maxatom; i++) {
    tempsum += (a[i][0]*b[i][0] + a[i][1]*b[i][1] + a[i][2]*b[i][2]);
  }
  return tempsum;
}

/* ----------------------------------------------------------------------
   evaluate potential energy and forces
   same logic as in Verlet
------------------------------------------------------------------------- */

double FixPIMRegCGLR1::update_energy()
{

  int eflag = 1;
  comm->forward_comm(this);
  if (pair_compute_flag) force->pair->compute(eflag, 0);
  if (kspace_compute_flag) force->kspace->compute(eflag, 0);
  //get energy from newly set parameters
  double energy = pe->compute_scalar();
  return energy;
}

/* ----------------------------------------------------------------------
   clear array needed, this only works on vector arrays. 
------------------------------------------------------------------------- */

void FixPIMRegCGLR1::grad_clear(double **array)
{
  size_t nbytes = sizeof(double) * atom->nlocal;
  if (force->newton) nbytes += sizeof(double) * atom->nghost;
  if (nbytes) memset(&array[0][0], 0, 3 * nbytes);
}

/* ----------------------------------------------------------------------
   reallocated local per-atoms arrays
------------------------------------------------------------------------- */

void FixPIMRegCGLR1::reallocate()
{
  memory->destroy(tempk);
  memory->destroy(rk);
  memory->destroy(pk);
  memory->destroy(rkp1);
  memory->destroy(bk);
  memory->destroy(tempf);
  maxatom = atom->nmax;
  memory->create(tempk, maxatom, 3, "pimreg:tempk");
  memory->create(rk, maxatom, 3, "pimreg:rk");
  memory->create(pk, maxatom, 3, "pimreg:pk");
  memory->create(rkp1, maxatom, 3, "pimreg:rkp1");
  memory->create(bk, maxatom, 3, "pimreg:bk");
  memory->create(tempf, maxatom, 3, "pimreg:tempf");
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixPIMRegCGLR1::memory_usage()
{
  double bytes = 0.0;
  //remember to update this to match the created arrays
  bytes += (double) 3 * maxatom * 6 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
  Communacation, send local to ghost
---------------------------------------------------------------------- */

int FixPIMRegCGLR1::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;
  double **mu = atom->mu;
  m = 0;

  for (i = 0; i < n; i ++) {
    j = list[i];
    buf[m++] = mu[j][0];
    buf[m++] = mu[j][1];
    buf[m++] = mu[j][2];
  }

  return m;
}

/* ----------------------------------------------------------------------
  Communacation, receive local to ghost
---------------------------------------------------------------------- */

void FixPIMRegCGLR1::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  double **mu = atom->mu;
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    mu[i][0] = buf[m++];
    mu[i][1] = buf[m++];
    mu[i][2] = buf[m++];
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/* ----------------------------------------------------------------------
   calculate the A*mu in dE/dmu = A*mu - b at a single i atom, depreciated, not tested to be correct
---------------------------------------------------------------------- */

void FixPIMRegCGLR1::calculate_AdotV_one(double **Adotv, double **x, double **v, int i, int *jlist, int jnum, PairPIM *pim)
{ 
  double tempSum;
  int j;
  double xtmp,ytmp,ztmp;
  double delx,dely,delz;
  double rsq, r;
  double r3inv,r5inv;
  double alpha_i;
  int itype;
  double vjdotr;

  itype = atom->type[i];
  alpha_i = pim->alpha[itype][itype];
  xtmp = x[i][0];
  ytmp = x[i][1];
  ztmp = x[i][2];
  //we expect to calculate the vector component i, with the neighbor list for atom i
  for (int idim = 0; idim < 3; idim++)
  { //Consider the product A[i,m,j,n]*v[j,n], this is getting the element [j,m], iterating over m
    tempSum = 0;
    //dedmu[i][idim] = 0;
    for(int jj=0; jj<jnum; jj++)
    { //Consider the product A[i,m,j,n]*v[j,n], this is the summation over j
      j = jlist[jj];
      j &= NEIGHMASK;
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      

      if (rsq < (pim->cut_coulsq)){
        r = sqrt(rsq);
        r3inv = (1/rsq)/r;
        r5inv = r3inv/rsq;
        vjdotr = delx*v[j][0] + dely*v[j][1] + delz*v[j][2];
        if (idim == 0){
          tempSum += (v[j][idim]*r3inv - 3.0*delx*r5inv*vjdotr);
        } else if (idim == 1){
          tempSum += (v[j][idim]*r3inv - 3.0*dely*r5inv*vjdotr);
        } else {
          tempSum += (v[j][idim]*r3inv - 3.0*delz*r5inv*vjdotr);
        }
      }
      //Consider the product A[i,m,j,n]*v[j,n], this is the summation over n, m is idim
      
    }
    //special case of j = i, which is not considered within the neighbor list

    //Self-energy calculation is correct, the problem is the interactive energy, it doesn't match the results in the pair-wise calculation
    tempSum += 1/alpha_i*v[i][idim];

    //A*mu
    Adotv[i][idim] = tempSum;
  }
  return;
}

/*----------------------------------------------------------------
Diagnostic printing 
-----------------------------------------------------------------*/
void FixPIMRegCGLR1::printA(PairPIM *pim)
{ 
  int i, j, idim, jdim;
  int nmax = atom->nmax;
  int nlocal = atom->nlocal;
  double A_one = 0;
  int ired,jred;

  for (i = 0; i < nlocal; i++) {
    
    for(j = 0; j< nlocal; j++)
    { //Consider the product A[i,m,j,n]*v[j,n], this is the summation over j
      for (idim = 0; idim<3;idim++){
        grad_clear(rk);
        rk[i][idim] = 1;
        for (jdim = 0; jdim<3; jdim++){
          grad_clear(tempk);
          grad_clear(bk);
          bk[j][jdim] = 1;
          calculate_AdotV(tempk,bk,pim);
          A_one = dot_local(rk,tempk,nmax);
          ired = 3*(i) + idim +1;
          jred = 3*(j) + jdim +1;
          //auto mesg = fmt::format("A[{:d}][{:d}][{:d}][{:d}] = {:e}\t",i+1,idim+1,j+1,jdim+1,A_one);
          //utils::logmesg(lmp,mesg);
          auto mesg1 = fmt::format("Ar({:d},{:d}) = {:e};\n",ired,jred,A_one);
          utils::logmesg(lmp,mesg1);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   calculate the b analytically in dE/dmu = A*mu - b at a single i atom, depreciated.
---------------------------------------------------------------------- */

void FixPIMRegCGLR1::calculate_b_one(double **b, double **x, double **mu, int i, int *jlist, int jnum, PairPIM *pim)
{
  double tempSum;
  int j;
  double xtmp,ytmp,ztmp;
  double delx,dely,delz;
  double rsq, r;
  double r3inv,r5inv;
  double *q = atom->q;
  double **cdinv = pim->cdinv;
  double **cd = pim->cd;
  double **bd = pim->bd;
  double gdampji;
  int *type = atom->type;
  int itype,jtype;
  double cdinvtemp,bdtmp;

  xtmp = x[i][0];
  ytmp = x[i][1];
  ztmp = x[i][2];
  itype = type[i];

  for (int idim = 0; idim < 3; idim++)
  { //Consider the vector b[i,m], which is a summation over j, we are setting the element [i,m] over m(idim)
    tempSum = 0;
    for(int jj=0; jj<jnum; jj++)
    { 
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];

      rsq = delx*delx + dely*dely + delz*delz;
      r = sqrt(rsq);
      r3inv = (1/rsq)/r;

      if (rsq < (pim->cut_coulsq)){
        //We have cdinv[j][i] = cdinv[i][j] to accomodate the symmetric requirement on initialization, which means we have to set cd or cdinv based on the comparison between jtype and itype. 
        if (itype<jtype){
          cdinvtemp = cdinv[itype][jtype];
        } else{
          cdinvtemp = cd[itype][jtype];
        }
        bdtmp = bd[itype][jtype];
        gdampji = 1 - cdinvtemp*exp(-bdtmp*r)*(1 + bdtmp*r + pow(bdtmp*r,2)/2 + pow(bdtmp*r,3)/6 + pow(bdtmp*r,4)/24);
        //note that the both terms actually contribute, and they are identical. I made a mistake in previous derivation.
        if (idim==0){
          tempSum +=(q[j]*r3inv*gdampji)*delx;
        } else if(idim ==1){
          tempSum +=(q[j]*r3inv*gdampji)*dely;
        } else {
          tempSum +=(q[j]*r3inv*gdampji)*delz;
        }
      }
    }
    // b
    b[i][idim] = tempSum;
  }

  return;
}

/* ----------------------------------------------------------------------
   calculate the diff analytically using dE/dmu = A*mu - b at the local， depreciated
---------------------------------------------------------------------- */

void FixPIMRegCGLR1::calculate_diff(double **dedmu, PairPIM *pim)
{
  int ii, i, inum, jnum,jj, itype;
  double xtmp,ytmp,ztmp;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double **mu=atom->mu;
  int *mask = atom->mask;
  double **x = atom->x;

  //Number of local atoms, I believe the local here means domain decomposition. 
  inum = list->inum;
  //ID of local atoms
  ilist = list->ilist;
  //Number of neighbours of the atom at hand
  numneigh = list->numneigh;
  //Pointer to the neighbour array
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    //calculate computes the [i,0], [i,1] and [i,2] components of dedmu using A*mu-b with neighbor cutoff
    //dedmu =A*mu - b; x is the location vector.
    //contains initialization, it has to be run before calculate_b_one
    calculate_AdotV_one(dedmu,x,mu,i,jlist,jnum,pim);
    calculate_b_one(bk,x,mu,i,jlist,jnum,pim);
    for(int idim =0; idim<3;idim++){
      dedmu[i][idim] -= bk[i][idim];
    }

  }
  return;
}