// @(#) root/glite:$Id$
// Author: Anar Manafov <A.Manafov@gsi.de> 2006-04-10

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/************************************************************************/
/*! \file TGLiteJobStatus.h
Class defining interface to a gLite result set.
Objects of this class are created by TGrid methods.*//*

         version number:    $LastChangedRevision: 1678 $
         created by:        Anar Manafov
                            2006-04-10
         last changed by:   $LastChangedBy: manafov $ $LastChangedDate: 2008-01-21 18:22:14 +0100 (Mon, 21 Jan 2008) $

         Copyright (c) 2006-2008 GSI GridTeam. All rights reserved.
*************************************************************************/

//glite-api-wrapper
#include <glite-api-wrapper/gLiteAPIWrapper.h>
// STD
#include <string>
// ROOT
#include "TGridJobStatus.h"
#include "TGLiteJobStatus.h"
//////////////////////////////////////////////////////////////////////////
//
// The TGLiteJobStatus class is a part of RGLite plug-in and
// represents a status of Grid jobs.
// Actually this class is responsible to retrieve a Grid job status and
// translate it to a TGridJobStatus::EGridJobStatus statuses.
//
// Related classes are TGLite.
//
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLiteJobStatus)

using namespace std;
using namespace glite_api_wrapper;


//______________________________________________________________________________
TGLiteJobStatus::TGLiteJobStatus(TString _jobID): m_sJobID(_jobID)
{
}


//______________________________________________________________________________
TGridJobStatus::EGridJobStatus TGLiteJobStatus::GetStatus() const
{
   // The GetStat() method retrieves a gLite job status and
   // translates it to a TGridJobStatus::EGridJobStatus type.
   // RETURN:
   //      a TGridJobStatus::EGridJobStatus status value.

   string sStatusName;
   string sStatusString;
   // Gets the status of the job reduced to the subset defined in TGridJobStatus.
   glite::lb::JobStatus::Code code(glite::lb::JobStatus::UNDEF);
   try {
      code = CGLiteAPIWrapper::Instance().GetJobManager().JobStatus(m_sJobID, &sStatusName, &sStatusString);
   } catch (const exception &e) {
      Error("GetStatus", "Exception: %s", e.what());
      return kUNKNOWN;
   }

   Info("GetStatus", "JobID = %s", m_sJobID.c_str());
   Info("GetStatus",
        "Job status is [%d]; gLite status code is \"%s\"; gLite status string is \"%s\"",
        code, sStatusName.c_str(), sStatusString.c_str());
   switch (code) {
      case glite::lb::JobStatus::DONE:
      case glite::lb::JobStatus::CLEARED:
      case glite::lb::JobStatus::PURGED:
         Info("GetStatus", "Job status is kDONE");
         return kDONE;
      case glite::lb::JobStatus::SUBMITTED:
      case glite::lb::JobStatus::WAITING:
      case glite::lb::JobStatus::READY:
         Info("GetStatus", "Job status is kWAITING");
         return kWAITING;
      case glite::lb::JobStatus::SCHEDULED:
      case glite::lb::JobStatus::RUNNING:
         Info("GetStatus", "Job status is kRUNNING");
         return kRUNNING;
      case glite::lb::JobStatus::ABORTED:
         Info("GetStatus", "Job status is kABORTED");
         return kABORTED;
      case glite::lb::JobStatus::CANCELLED:
         Info("GetStatus", "Job status is kFAIL");
         return kFAIL;
      default:
         // glite::lb::JobStatus::CODE_MAX:
         // glite::lb::JobStatus::UNKNOWN:
         // glite::lb::JobStatus::UNDEF:
         Info("GetStatus", "Job status is kUNKNOWN");
         return kUNKNOWN;
   }
}
