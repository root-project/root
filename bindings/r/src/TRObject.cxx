// Author: Omar Zapata  Omar.Zapata@cern.ch   2014

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRObject.h>
#include<vector>

using namespace ROOT::R;
ClassImp(TRObject);

//______________________________________________________________________________
TRObject::TRObject(SEXP robj): TObject(), fObj(robj), fStatus(kTRUE) { }


//______________________________________________________________________________
void TRObject::operator=(SEXP robj)
{
   fStatus = kTRUE;
   fObj = robj;
}

//______________________________________________________________________________
TRObject::TRObject(SEXP robj, Bool_t status): fObj(robj), fStatus(status) {}
