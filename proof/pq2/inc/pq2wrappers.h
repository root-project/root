// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef PQ2_wrappers
#define PQ2_wrappers

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// pq2wrappers                                                          //
//                                                                      //
// Prototypes for wrapper functions used in PQ2                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFileCollection.h"
#include "TMap.h"

void             DataSetCache(bool clear, const char *ds);
void             ShowDataSets(const char *ds, const char *opt = "");
TFileCollection *GetDataSet(const char *ds, const char *server = "");
TMap            *GetDataSets(const char *owner = "", const char *server = "", const char *opt = 0);
Int_t            RemoveDataSet(const char *dsname);
Int_t            VerifyDataSet(const char *dsname, const char *opt = 0, const char *redir = 0);
Bool_t           ExistsDataSet(const char *dsname);
Int_t            RegisterDataSet(const char *dsname, TFileCollection *fc, const char *opt = "");

#endif
