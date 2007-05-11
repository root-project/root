// @(#)root/net:$Name:  $:$Id: TGridJob.h,v 1.2 2005/05/20 09:59:35 rdm Exp $
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridJob                                                             //
//                                                                      //
// Abstract base class defining interface to a GRID job.                //
//                                                                      //
// Related classes are TGridJobStatus.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TGridJobStatus;

class TGridJob : public TObject {

protected:
   TString  fJobID;  // the job's ID

public:
   TGridJob(TString jobID) : fJobID(jobID) { }
   virtual ~TGridJob() { }

   virtual TString GetJobID() { return fJobID; }

   virtual TGridJobStatus *GetJobStatus() const = 0;

   ClassDef(TGridJob,1)  // ABC defining interface to a GRID job
};

#endif
