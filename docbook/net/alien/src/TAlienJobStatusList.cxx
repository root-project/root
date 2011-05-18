// @(#)root/alien:$Id$
// Author: Andreas-Joachim Peters  10/12/2006

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJobStatusList                                                  //
//                                                                      //
// Alien implementation of TGridJobStatusList                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienJobStatusList.h"
#include "TAlienJobStatus.h"
#include "TROOT.h"

ClassImp(TAlienJobStatusList)

//______________________________________________________________________________
void TAlienJobStatusList::PrintCollectionEntry(TObject* entry, Option_t* /*option*/,
                                               Int_t /*recurse*/) const
{
   // Print information about jobs.

   TAlienJobStatus* jobstatus = (TAlienJobStatus*) entry;
   TString split(jobstatus->GetKey("split"));
   TString queueid(jobstatus->GetKey("queueId"));
   TROOT::IndentLevel();
   printf("JobId = %s Split = %s\n", queueid.Data(), split.Data());
}
