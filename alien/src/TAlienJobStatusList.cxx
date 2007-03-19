// @(#)root/alien:$Name:  $:$Id: TAlienJobStatus.cxx,v 1.3 2006/05/19 07:30:04 brun Exp $
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

ClassImp(TAlienJobStatusList)

//______________________________________________________________________________
void TAlienJobStatusList::Print(const Option_t* options) const
{
  // Extract the master jobs.

   TIter next(this);
   TAlienJobStatus* jobstatus=0;

   while ( ( jobstatus = (TAlienJobStatus*) next() ) ) {
      TString split;
      TString queueid;
      queueid = jobstatus->GetKey("queueId");
      split = jobstatus->GetKey("split");
      printf("JobId = %s Split = %s\n",queueid.Data(),split.Data());
   }
}
