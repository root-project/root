// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef PQ2_ping
#define PQ2_ping

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// pq2ping                                                              //
//                                                                      //
// Prototypes for functions used in PQ2 functions to check daemons      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

Int_t checkUrl(const char *url, const char *flog, bool def_proof = 0);
Int_t pingXproofdAt();
Int_t pingXrootdAt();
Int_t pingServerAt();

#endif
