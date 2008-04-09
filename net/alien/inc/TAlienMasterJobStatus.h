// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienMasterJobStatus
#define ROOT_TAlienMasterJobStatus

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienMasterJobStatus                                                //
//                                                                      //
// Status of a MasterJob.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridJobStatus
#include "TGridJobStatus.h"
#endif
#ifndef ROOT_TMap
#include "TMap.h"
#endif

class TAlienJobStatus;
class TAlienMasterJob;


class TAlienMasterJobStatus : public TGridJobStatus {

friend class TAlienMasterJob;

private:
   TAlienJobStatus *fMasterJob;  // Status of the master job
   TMap             fJobs;       // Map which contains the sub jobs,
                                 // key is the job ID, values are
                                 // TAlienJobStatus objects

public:
   TAlienMasterJobStatus(const char* jobid) : fMasterJob(0)
      { TString name; name = jobid; SetName(name); SetTitle(name); }
   virtual ~TAlienMasterJobStatus();

   EGridJobStatus GetStatus() const;
   void Print(Option_t *) const;

   Float_t PercentFinished();

   Bool_t IsFolder() const { return kTRUE; }
   void   Browse(TBrowser *b);
   TMap  *GetJobs() { return &fJobs; }
   Int_t  GetNSubJobs() const { return fJobs.GetSize(); }

   ClassDef(TAlienMasterJobStatus,1)  // Status of Alien master job
};

#endif
