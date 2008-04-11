/* @(#)root/pythia6:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TPythia6Calls
#define ROOT_TPythia6Calls
//
//           converted with i2h by P.Murat
//
//------------------------------------------------------------------------------
//...This file contains a complete listing of all PYTHIA
//...commonblocks, and additionally some recommended other
//...declarations. You may copy this to the top of your
//...mina program and then eliminate unnecessary parts.
//  Jun 19 1998 P.Murat(CDF): add implicit for integers
//-----------------------------------------------------------------
//...All real arithmetic in double precision.
//      IMPLICIT DOUBLE  PRECISION(A-H, O-Z)
//      implicit integer (i-n)
//...Three Pythia functions return integers, so need declaring.
//...Parameter statement to help give large particle numbers
//...(left- and righthanded SUSY, excited fermions).
//...Commonblocks.
//...The event record.
//...Parameters.
//...Particle properties + some flavour parameters.
//...Decay information.
//...Particle names
//...Random number generator information.
//...Selection of hard scattering subprocesses.
//...Parameters.
//...Internal variables.
//...Process information.
//...Parton distributions and cross sections.
//...Resonance width and secondary decay treatment.
//...Generation and cross section statistics.
//...Process names.
//...Total cross sections.
//...Photon parton distributions: total and valence only.
//...Setting up user-defined processes.
//...Supersymmetry parameters.
//...Supersymmetry mixing matrices.
//...Parameters for Gauss integration of supersymmetric widths.
//...Histogram information.
//------------------------------------------------------------------------------

int   const KSUSY1  =  1000000;
int   const KSUSY2  =  2000000;
int   const KEXCIT  =  4000000;
int   const KNDCAY  =  8000; //should be 4000 for pythia61


struct Pyjets_t {
  int    N;
  int    NPAD;
  int    K[5][4000];
  double P[5][4000];
  double V[5][4000];
};

struct Pydat1_t {
  int    MSTU[200];
  double PARU[200];
  int    MSTJ[200];
  double PARJ[200];
};

struct Pydat2_t {
  int    KCHG[4][500];
  double PMAS[4][500];
  double PARF[2000];
  double VCKM[4][4];
};

struct Pydat3_t {
  int    MDCY[3][500];
  int    MDME[2][KNDCAY];
  double BRAT[KNDCAY];
  int    KFDP[5][KNDCAY];
};

struct Pydat4_t {
  char  CHAF[2][500][16];	// here I needed manual intervention
};

struct Pydatr_t {
  int    MRPY[6];
  double RRPY[100];
};

struct Pysubs_t {
  int    MSEL;
  int    MSELPD;
  int    MSUB[500];
  int    KFIN[81][2];		//
  double CKIN[200];
};

struct Pypars_t {
  int    MSTP[200];
  double PARP[200];
  int    MSTI[200];
  double PARI[200];
};

struct Pyint1_t {
  int    MINT[400];
  double VINT[400];
};

struct Pyint2_t {
  int    ISET[500];
  int    KFPR[2][500];
  double COEF[20][500];
  int    ICOL[2][4][40];
};

struct Pyint3_t {
  double XSFX[81][2];		//
  int    ISIG[3][1000];
  double SIGH[1000];
};

struct Pyint4_t {
  int    MWID[500];
  double WIDS[5][500];
};

struct Pyint5_t {
  int    NGENPD;
  int    NGEN[3][501];
  double XSEC[3][501];
};

struct Pyint6_t {
  char PROC[501][28];
};

struct Pyint7_t {
  double SIGT[6][7][7];
};

struct Pyint8_t {
  double XPVMD[13];
  double XPANL[13];
  double XPANH[13];
  double XPBEH[13];
  double XPDIR[13];
};

struct Pyint9_t {
  double VXPVMD[13];
  double VXPANL[13];
  double VXPANH[13];
  double VXPDGM[13];
};

struct Pymssm_t {
  int    IMSS[100];
  double RMSS[100];
};

struct Pyssmt_t {
  double ZMIX[4][4];
  double UMIX[2][2];
  double VMIX[2][2];
  double SMZ[4];
  double SMW[2];
  double SFMIX[4][16];
  double ZMIXI[4][4];
  double UMIXI[2][2];
  double VMIXI[2][2];
};

struct Pyints_t {
  double XXM[20];
};

struct Pybins_t {
  int    IHIST[4];
  int    INDX[1000];
  double BIN[20000];
};

#endif
