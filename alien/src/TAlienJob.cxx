// @(#)root/alien:$Name:  $:$Id: TAlienJob.cxx,v 1.5 2004/11/01 17:38:09 jgrosseo Exp $
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

#include "TAlienJob.h"
#include "TAlienJobStatus.h"
#include "TObjString.h"
#include "glite_job_operations.h"

ClassImp(TAlienJob)


//______________________________________________________________________________
TGridJobStatus *TAlienJob::GetJobStatus() const
{
   // Queries the job for its status and returns a TGridJobStatus object.
   // Returns 0 in case of failure.

   TString jobID;
   jobID += (static_cast<ULong_t>(fJobID));

   GLITE_JOBARRAY *gjobarray = glite_queryjobs("-", "-", "-", "-", "-", "-",
                                               jobID.Data(), "-", "-");

   if (!gjobarray)
      return 0;

   if (gjobarray->size() == 0) {
      delete gjobarray;
      return 0;
   }

   TAlienJobStatus *status = new TAlienJobStatus();

   GLITE_JOB gjob = gjobarray->at(0);
   std::map<std::string, std::string>::const_iterator iter = gjob.glite_jobmap.begin();
   for (; iter != gjob.glite_jobmap.end(); ++iter) {
      status->fStatus.Add(new TObjString(iter->first.c_str()), new TObjString(iter->second.c_str()));
   }

   delete gjobarray;

   return status;
}
