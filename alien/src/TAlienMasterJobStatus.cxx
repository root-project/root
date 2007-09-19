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
// TAlienMasterJobStatus                                                //
//                                                                      //
// Status of a MasterJob                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienJobStatus.h"
#include "TAlienMasterJobStatus.h"
#include "TObjString.h"
#include "TBrowser.h"

ClassImp(TAlienMasterJobStatus)

//______________________________________________________________________________
TAlienMasterJobStatus::~TAlienMasterJobStatus()
{
   // Cleanup.

   fJobs.DeleteAll();

   if (fMasterJob)
      delete fMasterJob;
}

//______________________________________________________________________________
void TAlienMasterJobStatus::Browse(TBrowser* b)
{
   // Browser interface.

   if (b) {
     //      TString status("");
     //      status += GetStatus();
     //      b->Add(new TNamed(status, TString("overall status")));
     //      status = "";
     //      status += PercentFinished();
     //      b->Add(new TNamed(status, TString("percentage finished")));

      TIterator* iter = fJobs.MakeIterator();

      TObject* obj = 0;
      while ((obj = iter->Next()) != 0) {
         TObjString* keyStr = dynamic_cast<TObjString*>(obj);
         TObject* value = fJobs.GetValue(obj);

         if (keyStr && value)
            b->Add(value, keyStr->GetString().Data());
      }
      delete iter;
   }
}

//______________________________________________________________________________
TGridJobStatus::EGridJobStatus TAlienMasterJobStatus::GetStatus() const
{
   // Returns the status of the master job reduced to the subset defined
   // in TGridJobStatus.

   if (!fMasterJob)
      return kUNKNOWN;

   return fMasterJob->GetStatus();
}

//______________________________________________________________________________
Float_t TAlienMasterJobStatus::PercentFinished()
{
   // Returns the percentage of finished subjobs, only DONE is considered
   // as finished.

   if (fJobs.GetSize() == 0)
      return 0;

   TIterator* iter = fJobs.MakeIterator();

   Int_t done = 0;

   TObject* obj = 0;
   while ((obj = iter->Next()) != 0) {
      TObject* value = fJobs.GetValue(obj);
      TAlienJobStatus* jobStatus = dynamic_cast<TAlienJobStatus*>(value);

      if (jobStatus) {
         if (jobStatus->GetStatus() == kDONE)
            ++done;
      }
   }

   delete iter;

   return (Float_t) done / fJobs.GetSize();
}

//______________________________________________________________________________
void TAlienMasterJobStatus::Print(Option_t *) const
{
   // Prints information of the master job and the sub job. Only the status is printed.

   if (fMasterJob) {
      printf("Printing information for the master job: ");
      fMasterJob->PrintJob(kFALSE);
   }

   TIterator* iter = fJobs.MakeIterator();

   TObject* obj = 0;
   while ((obj = iter->Next()) != 0) {
      TObjString* keyStr = dynamic_cast<TObjString*>(obj);

      TObject* value = fJobs.GetValue(obj);
      TAlienJobStatus* jobStatus = dynamic_cast<TAlienJobStatus*>(value);

      if (keyStr && jobStatus) {
         printf("Printing info for subjob %s: ", keyStr->GetString().Data());
         jobStatus->PrintJob(kFALSE);
      }
   }
   delete iter;
}
