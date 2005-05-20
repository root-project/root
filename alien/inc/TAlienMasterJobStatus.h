// @(#)root/alien:$Name:  $:$Id: TAlienMasterJobStatus.h,v 1.3 2004/11/01 17:38:08 jgrosseo Exp $
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
   TAlienMasterJobStatus() : fMasterJob(0) { }
   virtual ~TAlienMasterJobStatus();

   virtual EGridJobStatus GetStatus() const;
   virtual void Print(Option_t *) const;

   Float_t PercentFinished();

   Bool_t IsFolder() const { return kTRUE; }
   void Browse(TBrowser *b);

   ClassDef(TAlienMasterJobStatus,1)  // Status of Alien master job
};

#endif
