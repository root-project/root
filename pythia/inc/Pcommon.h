/* @(#)root/pythia:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_PCommon
#define ROOT_PCommon

#ifndef __CFORTRAN_LOADED
#include "cfortran.h"
#endif


extern "C" {
/*========================================================*/
/* COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)  */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    mstp[200];
   Float_t  parp[200];
   Int_t    msti[200];
   Float_t  pari[200];
} PyparsCommon;

#define PYPARS COMMON_BLOCK(PYPARS,pypars)
COMMON_BLOCK_DEF(PyparsCommon,PYPARS);

/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Provides information on latest event generated,        */
/* statistics. Contains status codes and parameters       */
/* regulating the performance of program.                 */
/*========================================================*/

/*========================================================*/
/* COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200)  */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    msel;
   Int_t    msub[200];
   Int_t    kfin[81][2];
   Float_t  ckin[200];
} PysubsCommon;

#define PYSUBS COMMON_BLOCK(PYSUBS,pysubs)
COMMON_BLOCK_DEF(PysubsCommon,PYSUBS);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Allows to run program with desired subset of process,  */
/* or restrict flavour and kinematics.                    */
/*                                                        */
/*  MSEL - switches between full user-control and pre-    */
/*          programed alternatives (look at documentation)*/
/*  MSUB - selects which subset of subprocesses to include*/
/*         in the generation(ordering follows ISUB code)  */
/*  KFIN[J][I]-provides an option to switch contributions  */
/*         to the cross-sections ->allows restriction on  */
/*         final state flavour. I=0->beam side of event;  */
/*         I=1->target side.J-enumerates flavours:        */
/*         WARNING!!!:In original F77-version flavours are*/
/*         enumerated from -40 to 40 -according to the KF */
/*         code. In 'C' there isn't a possibility to have */
/*         an array indexes start from number different   */
/*         from 0. Thus, one have to add 40 to desired KF */
/*         code to obtain proper index in 'C'-array.      */
/*  CKIN - kinematics cuts settings - see documentation.  */
/*========================================================*/

/*========================================================*/
/* COMMON/PYINT1/MINT(400),VINT(400)                      */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    mint[400];
   Float_t  vint[400];
} Pyint1Common;

#define PYINT1 COMMON_BLOCK(PYINT1,pyint1)
COMMON_BLOCK_DEF(Pyint1Common,PYINT1);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Worksapce arrays.                                      */
/* These arrays collects a host of integer and real values*/
/* used internaly during initialization/event generation. */
/*========================================================*/

/*========================================================*/
/* COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),      */
/*                ICOL(40,4,2)                            */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    iset[200];
   Int_t    kfpr[2][200];
   Float_t  coef[20][200];
   Int_t    icol[2][4][40];
} Pyint2Common;

#define PYINT2 COMMON_BLOCK(PYINT2,pyint2)
COMMON_BLOCK_DEF(Pyint2Common,PYINT2);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Workspace arrays.                                      */
/* These arrays are necessary to store Jacobians, etc.    */
/*========================================================*/

/*========================================================*/
/* COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)   */
/*--------------------------------------------------------*/
typedef struct {
   Float_t  xsfx[81][2];
   Int_t    isig[3][1000];
   Float_t  sigh[1000];
} Pyint3Common;

#define PYINT3 COMMON_BLOCK(PYINT3,pyint3)
COMMON_BLOCK_DEF(Pyint3Common,PYINT3);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Stores information about crosssections and parton dis- */
/* tribution and relative final state weights             */
/* WARNING!!! Values must not be changed by a user!!!     */
/*========================================================*/

/*========================================================*/
/* COMMON/PYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),      */
/*                WIDS(21:40,3)               */
/*--------------------------------------------------------*/
typedef struct {
   Float_t    widp[41][20];
   Float_t    wide[41][20];
   Float_t    wids[3][20];
} Pyint4Common;

#define PYINT4 COMMON_BLOCK(PYINT4,pyint4)
COMMON_BLOCK_DEF(Pyint4Common,PYINT4);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Stores decay wieths for resonances                     */
/* WARNING!!! Values must not be changed by a user!!!     */
/*========================================================*/


/*========================================================*/
/* COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3)   */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    ngen[3][201];
   Float_t  xsec[3][201];
} Pyint5Common;

#define PYINT5 COMMON_BLOCK(PYINT5,pyint5)
COMMON_BLOCK_DEF(Pyint5Common,PYINT5);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Stores information necessary for cross-section         */
/* calculation.                                           */
/* WARNING!!! Values must not be changed by a user!!!     */
/*========================================================*/

/***************************************************************/
/*                      LUJETS part                                             */
/***************************************************************/

/*========================================================*/
/* COMMON/LUJETS/N,K(4000,5),P(4000,5),V(4000,5)          */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    n;
   Int_t    k[5][4000];
   Float_t  p[5][4000];
   Float_t  v[5][4000];
} LujetsCommon;

#define LUJETS COMMON_BLOCK(LUJETS,lujets)
COMMON_BLOCK_DEF(LujetsCommon,LUJETS);
/***********************************************************/
/*           D E S C R I P T I O N :                       */
/*---------------------------------------------------------*/
/*  N      - number of lines in K,P,V matrices occupied by */
/*           current event                                 */
/*  K[0][I] - Status Code KS (look at documentation)       */
/*  K[1][I] - Parton/Particle KF code                      */
/*  K[2][I] - Line number of Parrent Particle              */
/*  K[3][I] - Line number of first daugher[or internal use]*/
/*  K[4][I] - Line number of last daughter[or internal use]*/
/*                                                         */
/*  P[0][I] - Px - momentum in the x direction [GeV/c]     */
/*  P[1][I] - Py - momentum in the y direction [GeV/c]     */
/*  P[2][I] - Pz - momentum in the z direction [GeV/c]     */
/*  P[3][I] - E  - energy [GeV]                            */
/*  P[4][I] - m  - mass [Gev/c^2]                          */
/*                                                         */
/*  V[0][I] - x position of production vertex [mm]         */
/*  V[1][I] - y position of production vertex [mm]         */
/*  V[2][I] - z position of production vertex [mm]         */
/*  V[3][I] - time of production [mm/c]=[3.33E-12 s]       */
/*  V[4][I] - proper lifetime of particle [mm/c]           */
/*=========================================================*/

/*========================================================*/
/* COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)  */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    mstu[200];
   Float_t  paru[200];
   Int_t    mstj[200];
   Float_t  parj[200];
} Ludat1Common;

#define LUDAT1 COMMON_BLOCK(LUDAT1,ludat1)
COMMON_BLOCK_DEF(Ludat1Common,LUDAT1);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* This common regulates the performance of program and   */
/*  gives access to somea number of status codes.         */
/*                                                        */
/*  MSTU,MPAR - related to utility function and Standard  */
/*              Model (look at documentation)             */
/*  MSTJ,PARJ - underlying physics assumptions            */
/*========================================================*/

/*========================================================*/
/* COMMON/LUDAT2/KCHG(500,3),PMASS(500,4),PARF(2000),     */
/*               VCKM(4,4)                                */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    kchg[3][500];
   Float_t  pmas[4][500];
   Float_t  parf[2000];
   Float_t  vckm[4][4];
} Ludat2Common;

#define LUDAT2 COMMON_BLOCK(LUDAT2,ludat2)
COMMON_BLOCK_DEF(Ludat2Common,LUDAT2);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* This gives access to a number of flavour treatment     */
/*  constants/parameters and particle/parton data.        */
/*                                                        */
/* Look at the documentation for details...               */
/*========================================================*/

/*========================================================*/
/* COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),     */
/*               KFDP(2000,5)                             */
/*--------------------------------------------------------*/
typedef struct {
   Int_t   mdcy[3][500];
   Int_t   mdme[2][2000];
   Float_t brat[2000];
   Int_t   kfdp[5][2000];
} Ludat3Common;

#define LUDAT3 COMMON_BLOCK(LUDAT3,ludat3)
COMMON_BLOCK_DEF(Ludat3Common,LUDAT3);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Gives access to particle decay data and parameters.    */
/* Look at the documentation for details...               */
/*========================================================*/

/*========================================================*/
/* COMMON/LUDAT4/CHAF(500)                                */
/* CHARACTER CHAF*8                                       */
/*--------------------------------------------------------*/
typedef struct {
   Char_t  chaf[500][8];
} Ludat4Common;

#define LUDAT4 COMMON_BLOCK(LUDAT4,ludat4)
COMMON_BLOCK_DEF(Ludat4Common,LUDAT4);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Gives access to character types variables:             */
/*                                                        */
/*  chaf - particle names(excluding charge) according to  */
/*         KC-code                                        */
/*========================================================*/

/*========================================================*/
/* COMMON/LUDATR/MRLU(6),RRLU(100)                        */
/*--------------------------------------------------------*/
typedef struct {
   Int_t    mrlu[6];
   Float_t  rrlu[100];
} LudatrCommon;

#define LUDATR COMMON_BLOCK(LUDATR,ludatr)
COMMON_BLOCK_DEF(LudatrCommon,LUDATR);
/**********************************************************/
/*           D E S C R I P T I O N :                      */
/*--------------------------------------------------------*/
/* Contains the state of random number generator.         */
/*========================================================*/

}

#endif
