/* @(#)root/eg:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/



#ifndef ROOT_HepEvt
#define ROOT_HepEvt

extern "C" {

#ifndef __CFORTRAN_LOADED
#include "cfortran.h"
#endif

typedef struct {
	Int_t	 nevhep;
        Int_t    nhep;
        Int_t    isthep[2000];
        Int_t    idhep[2000];
        Int_t    jmohep[2000][2];
        Int_t    jdahep[2000][2];
        Double_t phep[2000][5];
        Double_t vhep[2000][4];
} HEPEVT_DEF;

#define HEPEVT COMMON_BLOCK(HEPEVT,hepevt)

COMMON_BLOCK_DEF(HEPEVT_DEF,HEPEVT);

HEPEVT_DEF HEPEVT;

}

#endif
