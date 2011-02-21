/* @(#)root/eg:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Hepevt
#define ROOT_Hepevt

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

extern "C" {

#ifndef __CFORTRAN_LOADED
#include "cfortran.h"
#endif

typedef struct {
   Int_t    nevhep;
   Int_t    nhep;
   Int_t    isthep[4000];
   Int_t    idhep[4000];
   Int_t    jmohep[4000][2];
   Int_t    jdahep[4000][2];
   Double_t phep[4000][5];
   Double_t vhep[4000][4];
} HEPEVT_DEF;

#define HEPEVT COMMON_BLOCK(HEPEVT,hepevt)

COMMON_BLOCK_DEF(HEPEVT_DEF,HEPEVT);

HEPEVT_DEF HEPEVT;

}

#endif
