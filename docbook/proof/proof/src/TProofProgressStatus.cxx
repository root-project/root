// @(#)root/proof:$Id$
// Author: Jan Iwaszkiewicz    08/08/08

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TProofProgressStatus.h"
#include "TObject.h"
#include "TString.h"
#include "TSystem.h"
#include "TTime.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofProgressStatus                                                 //
//                                                                      //
// Small class including processing statistics                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TProofProgressStatus)

//______________________________________________________________________________
TProofProgressStatus::TProofProgressStatus(Long64_t entries,
                                           Long64_t bytesRead,
                                           Long64_t readCalls,
                                           Double_t procTime,
                                           Double_t cpuTime): TObject()
{
   // Main and default constructor

   fLastEntries = 0;
   fEntries = entries;
   fBytesRead = bytesRead;
   fReadCalls = readCalls;
   fLearnTime = 0.;
   fLastProcTime = 0;
   fProcTime = procTime;
   fCPUTime = cpuTime;
   SetLastUpdate();
}

//______________________________________________________________________________
TProofProgressStatus& TProofProgressStatus::operator+=(const TProofProgressStatus &st)
{
   // '+=' operator

   fLastEntries += st.fEntries;
   fEntries += st.fEntries;
   fBytesRead += st.fBytesRead;
   fReadCalls += st.fReadCalls;
   if (st.fLearnTime > fLearnTime)
      fLearnTime = st.fLearnTime;
   fLastProcTime = st.fProcTime;
   fProcTime += st.fProcTime;
   fCPUTime += st.fCPUTime;
   SetLastUpdate();
   return *this;
}

//______________________________________________________________________________
TProofProgressStatus& TProofProgressStatus::operator-=(const TProofProgressStatus &st)
{
   // '-=' operator

   fEntries -= st.fEntries;
   fBytesRead -= st.fBytesRead;
   fReadCalls -= st.fReadCalls;
   if (st.fLearnTime < fLearnTime)
      fLearnTime = st.fLearnTime;
   fProcTime -= st.fProcTime;
   fCPUTime -= st.fCPUTime;
   SetLastUpdate();
   return *this;
}

//______________________________________________________________________________
TProofProgressStatus TProofProgressStatus::operator-(TProofProgressStatus &st)
{
   // '-' operator

   return TProofProgressStatus(*this) -= st;
}

//______________________________________________________________________________
void TProofProgressStatus::Print(Option_t*option) const
{
   // Dump the content

   Printf("TProofProgressStatus:%s: Ents:(%lld,%lld), Bytes:%lld, Calls:%lld,"
          " Learn:%.3g s, Proc:(%.3g,%.3g) s, CPU:%.3g s",
          option, fEntries, fLastEntries, fBytesRead, fReadCalls,
          fLearnTime, fProcTime, fLastProcTime, fCPUTime);
}

//______________________________________________________________________________
void TProofProgressStatus::SetLastUpdate(Double_t updtTime)
{
   // Update time stamp either with the passed value (if > 0) or with
   // the current time

   if (updtTime > 0) {
      fLastUpdate = updtTime;
   } else {
      TTime tnow = gSystem->Now();
      fLastUpdate = (Double_t) (Long64_t(tnow)) / (Double_t)1000.;
   }
}

//______________________________________________________________________________
Double_t TProofProgressStatus::GetCurrentRate() const
{
   // Get current rate. Rteunr the average rate if the current is not defined

   if (fLastProcTime > 0) {
      return fLastEntries / fLastProcTime;
   }
   return GetRate();
}
