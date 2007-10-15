// @(#)root/alien:$Id$
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
// TAlienJob                                                            //
//                                                                      //
// Alien implentation of TGridJob                                       //
//                                                                      //
// Related classes are TAlienJobStatus.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGrid.h"
#include "TAlienJob.h"
#include "TAlienJobStatus.h"
#include "TObjString.h"
#include "gapi_job_operations.h"

ClassImp(TAlienJob)


//______________________________________________________________________________
TGridJobStatus *TAlienJob::GetJobStatus() const
{
   // Queries the job for its status and returns a TGridJobStatus object.
   // Returns 0 in case of failure.

   TString jobID;
   jobID = fJobID;

   GAPI_JOBARRAY *gjobarray = gapi_queryjobs("-", "%", "-", "-", "-", "-",
                                             jobID.Data(), "-", "-");

   if (!gjobarray)
      return 0;

   if (gjobarray->size() == 0) {
      delete gjobarray;
      return 0;
   }

   TAlienJobStatus *status = new TAlienJobStatus();

   GAPI_JOB gjob = gjobarray->at(0);
   std::map<std::string, std::string>::const_iterator iter = gjob.gapi_jobmap.begin();
   for (; iter != gjob.gapi_jobmap.end(); ++iter) {
      status->fStatus.Add(new TObjString(iter->first.c_str()), new TObjString(iter->second.c_str()));
   }

   delete gjobarray;

   return status;
}

//______________________________________________________________________________
Bool_t TAlienJob::Cancel()
{
   // Cancels a job e.g. sends a kill command.
   // Returns kFALSE in case of failure, otherwise kTRUE.

   if (gGrid) {
      return gGrid->Kill((TGridJob*)this);
   }
   Error("Cancel","No GRID connection (gGrid=0)");
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TAlienJob::Resubmit()
{
   // Resubmits a job.
   // Returns kFALSE in case of failure, otherwise kTRUE.

   if (gGrid) {
      return gGrid->Resubmit((TGridJob*)this);
   }
   Error("Cancel","No GRID connection (gGrid=0)");
   return kFALSE;
}
