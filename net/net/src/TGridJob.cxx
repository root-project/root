// @(#)root/net:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   06/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridJob                                                             //
//                                                                      //
// Abstract base class defining interface to a GRID job.                //
//                                                                      //
// Related classes are TGridJobStatus.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGridJob.h"

ClassImp(TGridJob);


////////////////////////////////////////////////////////////////////////////////
/// Must be implemented by actual GRID job implementation. Returns -1 in
/// case of error, 0 otherwise.

Int_t TGridJob::GetOutputSandbox(const char *, Option_t *)
{
   MayNotUse("GetOutputSandbox");
   return -1;
}
