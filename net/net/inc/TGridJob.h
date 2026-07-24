// @(#)root/net:$Id$
// Author: Jan Fiete Grosse-Oetringhaus  06/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridJob
#define ROOT_TGridJob

#include "TObject.h"
#include "TString.h"


class TGridJobStatus;

//////////////////////////////////////////////////////////////////////////
///                                                                     
/// Abstract base class defining interface to a GRID job.               
///                                                                      
/// \note This class is deprecated. It is kept for backward compatibility
/// but it should not be used in new code.                               
///                                                                     
/// Related classes are TGridJobStatus.                                 
///                                                                     
//////////////////////////////////////////////////////////////////////////
class TGridJob : public TObject {

protected:
   TString  fJobID;  // the job's ID

public:
   TGridJob(TString jobID) : fJobID(jobID) { }
   virtual ~TGridJob() { }

   virtual TString GetJobID() { return fJobID; }

   virtual TGridJobStatus *GetJobStatus() const = 0;
   virtual Int_t           GetOutputSandbox(const char *localpath, Option_t *opt = nullptr);

   virtual Bool_t          Resubmit() = 0;
   virtual Bool_t          Cancel () = 0;
   ClassDefOverride(TGridJob,1)  // ABC defining interface to a GRID job
};

#endif
