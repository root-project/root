// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus  06/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienJob
#define ROOT_TAlienJob

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJob                                                            //
//                                                                      //
// Alien implentation of TGridJob                                       //
//                                                                      //
// Related classes are TAlienJobStatus.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridJob
#include "TGridJob.h"
#endif


class TAlienJob : public TGridJob {

public:
   TAlienJob(TString jobID) : TGridJob(jobID) { }
   virtual ~TAlienJob() { }

   virtual TGridJobStatus *GetJobStatus() const;
   virtual Bool_t          Resubmit();
   virtual Bool_t          Cancel();

   ClassDef(TAlienJob,1)  // Alien implementation of TGridJob
};

#endif
