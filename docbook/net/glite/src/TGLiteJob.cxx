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
/*! \file TGLiteJob.cxx
gLite implementation of TGridJob*//*

         version number:    $LastChangedRevision: 1678 $
         created by:        Anar Manafov
                            2006-04-10
         last changed by:   $LastChangedBy: manafov $ $LastChangedDate: 2008-01-21 18:22:14 +0100 (Mon, 21 Jan 2008) $

         Copyright (c) 2006-2008 GSI GridTeam. All rights reserved.
*************************************************************************/

// glite-api-wrapper
#include <glite-api-wrapper/gLiteAPIWrapper.h>
// ROOT RGLite
#include "TGLiteJob.h"
#include "TGLiteJobStatus.h"

//////////////////////////////////////////////////////////////////////////
//
// The TGLiteJob class is a part of RGLite plug-in and
// represents a Grid job and offers a possibility to
// query the job status and retrieve its output sandbox.
//
// Related classes are TGLite.
//
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLiteJob)

using namespace std;
using namespace glite_api_wrapper;
using namespace MiscCommon;

//______________________________________________________________________________
TGridJobStatus* TGLiteJob::GetJobStatus() const
{
   // The GetJobStatus() method queries the job for its status.
   // RETURN:
   //      a TGridJobStatus object.

   // Returns 0 in case of failure.
   return dynamic_cast<TGridJobStatus*>(new TGLiteJobStatus(fJobID));
}


//______________________________________________________________________________
Int_t TGLiteJob::GetOutputSandbox(const char* _localpath, Option_t* /*opt*/)
{
   // Retrieving the output sandbox files.
   // INPUT:
   //      _localpath  [in] - a local destination path for output sandbox.
   // NOTE:
   //      The other parameter is unsupported.
   // RETURN:
   //      The method returns -1 in case of errors and 0 otherwise.

   // TODO: Add Info message;
   // TODO: Add option "nopurge" to TGLiteJob::GetOutputSandbox, since it's supported now by glite-api-wrapper
   try {
      CJobManager::delivered_output_t joboutput_path;
      CGLiteAPIWrapper::Instance().GetJobManager().JobOutput(string(fJobID), _localpath, &joboutput_path);

      // Print all output directories
      CJobManager::delivered_output_t::const_iterator iter = joboutput_path.begin();
      CJobManager::delivered_output_t::const_iterator iter_end = joboutput_path.end();
      Info("GetOutputSandbox", "The output has been delivered [ job ] -> [local output directory]");
      for (; iter != iter_end; ++iter) {
         stringstream ss;
         ss << "[" << iter->first << "] -> [" << iter->second << "]";
         Info("GetOutputSandbox", ss.str().c_str());
      }
      return 0;
   } catch (const exception &_e) {
      Error("GetOutputSandbox", "Exception: %s", _e.what());
      return -1;
   }
}


//______________________________________________________________________________
Bool_t TGLiteJob::Resubmit()
{
   // Not implemented for RGLite.

   MayNotUse("Resubmit");
   return kFALSE;
}


//______________________________________________________________________________
Bool_t TGLiteJob::Cancel()
{
   // The Cancel() method cancels a gLite job, which was assigned to the class.
   // RETURN:
   //      kTRUE if succeeded and kFALSE otherwise.

   try {
      CGLiteAPIWrapper::Instance().GetJobManager().JobCancel(string(fJobID));
   } catch (const exception &_e) {
      Error("Cancel", "Exception: %s", _e.what());
      return kFALSE;
   }
   return kTRUE;
}
