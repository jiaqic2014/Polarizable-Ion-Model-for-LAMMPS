# Polarizable-Ion-Model-for-LAMMPS
The Polarizable Ion Model for molten salt is implemented in LAMMPS. The work is realted to a study of noble metal behaviors in molten salt, which has been submitted for the NURETH-21 meeting. 

## TLDR

Use pair style pim, fix style pimreg/cg, and kspace ewald for verified performance. 

# Description of files:
## Extra-Pair

Contains two pair type. The pim and pim/lr pairstyle. The pim/lr style is not verified, use with caution. 

1. The pim pair use Ewald summation on the charge-charge interaction. The pim style should be used with the fix style pimreg_cg or pimreg/lr1 and the kspace ewald style. 

2. The pim/lr style addionally includes the long range modeling for charge-dipole and dipole-dipole interaction. The pim/lr style needs to be used with fix style pimreg/cg/lr or pimreg/cg. For kspace, the pair should be used with ewald_cg or ewald_disp, read the code to determine their usage. 

## Extra-Fix

Contains the fix styles used to calculate the induced dipoles. 

1. fix pimreg/cg style use short range electrical force and the conjugate gradient method to calculate the induced dipoles. This code has been verified with test cases. The description is given in the NURETH-21 paper.

2. fix pimreg/lr and the rest styles include the long range electric forces in the induced dipole calculation.
   
   2.1. fix pimreg/lr/cut. Include long range dipole and long range charge-dipole in the pair style, but not in the induced dipole calculation.

   2.2. fix pimreg/lr. Include long range dipole and long range charge-dipole in the pair style. Include long range charge force in the calculation of induced dipole. Used with kspcae ewald/cg style.
   
   2.3. fix pimreg/lr1. Include long range charge force, but not long range dipole force in the pair style. Use with pair/pim. Include long range charge force in the calculation of indluced dipole.

## KSPACE

1. Ewald. Modified a bit for the simulation. This is not good practice as the built in ewald style is slightly modified.
2. Ewald/cg. Modified from ewald disp to be used with the charge atom style. 
