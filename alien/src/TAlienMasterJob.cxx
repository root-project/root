// @(#)root/alien:$Name:  $:$Id: TAlienMasterJob.cxx,v 1.5 2004/11/01 17:38:09 jgrosseo Exp $
// Author: Jan Fiete Grosse-Oetringhaus   27/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienMasterJob                                                      //
//                                                                      //
// Special Grid job which contains a master job which controls          //
// underlying jobs resulting from job splitting.                        //
//                                                                      //
// Related classes are TAlienJobStatus.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienJobStatus.h"
#include "TAlienMasterJob.h"
#include "TAlienMasterJobStatus.h"
#include "TAlienJob.h"
#include "TObjString.h"
#include "glite_job_operations.h"

ClassImp(TAlienMasterJob)

//______________________________________________________________________________
TGridJobStatus *TAlienMasterJob::GetJobStatus() const
{
   // Gets the status of the master job and all its sub jobs.
   // Returns a TAlienMasterJobStatus object, 0 on failure.

   TString jobID;
   jobID += (static_cast<ULong_t>(fJobID));

   GLITE_JOBARRAY* gjobarray = glite_queryjobs("-", "-", "-", "-", jobID.Data(),
                                               "-", "-", "-", "-");

   if (!gjobarray)
      return 0;

   if (gjobarray->size() == 0) {
      delete gjobarray;
      return 0;
   }

   TAlienMasterJobStatus *status = new TAlienMasterJobStatus();

   TAlienJob masterJob(fJobID);
   status->fMasterJob = dynamic_cast<TAlienJobStatus*>(masterJob.GetJobStatus());

   std::vector<GLITE_JOB>::const_iterator jobIter = gjobarray->begin();
   for (; jobIter != gjobarray->end(); ++jobIter) {

      GLITE_JOB gjob = *jobIter;
      TAlienJobStatus* jobStatus = new TAlienJobStatus();
      TObjString* jobID = 0;

      std::map<std::string, std::string>::const_iterator iter = gjob.glite_jobmap.begin();
      for (; iter != gjob.glite_jobmap.end(); ++iter) {
         jobStatus->fStatus.Add(new TObjString(iter->first.c_str()), new TObjString(iter->second.c_str()));
         if (strcmp(iter->first.c_str(), "queueId") == 0)
            jobID = new TObjString(iter->second.c_str());
      }

      if (jobID != 0)
         status->fJobs.Add(jobID, jobStatus);
      else
         delete jobStatus;
   }

   return status;
}
