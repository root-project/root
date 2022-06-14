// Author: Omar Zapata  Omar.Zapata@cern.ch   2015

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRFunctionExport.h>

using namespace ROOT::R;
ClassImp(TRFunctionExport);


//______________________________________________________________________________
TRFunctionExport::TRFunctionExport(): TObject()
{
   f = NULL;
}

//______________________________________________________________________________
TRFunctionExport::TRFunctionExport(const TRFunctionExport &fun): TObject(fun)
{
   f = fun.f;
}

