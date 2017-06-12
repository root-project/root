// @(#)root/proof:$Id$
// Author: Jan Iwaszkiewicz    08/08/08

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TProofProgressStatus
\ingroup proofkernel

Container class for processing statistics

*/

#include "TProofProgressStatus.h"
#include "TObject.h"
#include "TString.h"
#include "TSystem.h"
#include "TTime.h"

ClassImp(TProofProgressStatus);

////////////////////////////////////////////////////////////////////////////////
/// Main and default constructor

TProofProgressStatus::TProofProgressStatus(Long64_t entries,
                                           Long64_t bytesRead,
                                           Long64_t readCalls,
                                           Double_t procTime,
                                           Double_t cpuTime): TObject()
{
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

////////////////////////////////////////////////////////////////////////////////
/// '+=' operator

TProofProgressStatus& TProofProgressStatus::operator+=(const TProofProgressStatus &st)
{
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

////////////////////////////////////////////////////////////////////////////////
/// '-=' operator

TProofProgressStatus& TProofProgressStatus::operator-=(const TProofProgressStatus &st)
{
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

////////////////////////////////////////////////////////////////////////////////
/// '-' operator

TProofProgressStatus TProofProgressStatus::operator-(TProofProgressStatus &st)
{
   return TProofProgressStatus(*this) -= st;
}

////////////////////////////////////////////////////////////////////////////////
/// Dump the content

void TProofProgressStatus::Print(Option_t*option) const
{
   Printf("TProofProgressStatus:%s: Ents:(%lld,%lld), Bytes:%lld, Calls:%lld,"
          " Learn:%.3g s, Proc:(%.3g,%.3g) s, CPU:%.3g s",
          option, fEntries, fLastEntries, fBytesRead, fReadCalls,
          fLearnTime, fProcTime, fLastProcTime, fCPUTime);
}

////////////////////////////////////////////////////////////////////////////////
/// Update time stamp either with the passed value (if > 0) or with
/// the current time

void TProofProgressStatus::SetLastUpdate(Double_t updtTime)
{
   if (updtTime > 0) {
      fLastUpdate = updtTime;
   } else {
      TTime tnow = gSystem->Now();
      fLastUpdate = (Double_t) (Long64_t(tnow)) / (Double_t)1000.;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get current rate. Rteunr the average rate if the current is not defined

Double_t TProofProgressStatus::GetCurrentRate() const
{
   if (fLastProcTime > 0) {
      return fLastEntries / fLastProcTime;
   }
   return GetRate();
}
