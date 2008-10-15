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
                                           Double_t procTime,
                                           Double_t cpuTime): TObject()
{
   // Main and default constructor

   fEntries = entries;
   fBytesRead = bytesRead;
   fProcTime = procTime;
   fCPUTime = cpuTime;
}

//______________________________________________________________________________
TProofProgressStatus& TProofProgressStatus::operator+=(const TProofProgressStatus &st)
{
   // '+=' operator

   fEntries += st.GetEntries();
   fBytesRead += st.GetBytesRead();
   fProcTime += st.GetProcTime();
   fCPUTime += st.GetCPUTime();
   return *this;
}

//______________________________________________________________________________
TProofProgressStatus& TProofProgressStatus::operator-=(const TProofProgressStatus &st)
{
   // '-=' operator

   fEntries -= st.GetEntries();
   fBytesRead -= st.GetBytesRead();
   fProcTime -= st.GetProcTime();
   fCPUTime -= st.GetCPUTime();
   return *this;
}

//______________________________________________________________________________
TProofProgressStatus TProofProgressStatus::operator-(TProofProgressStatus &st)
{
   // '-' operator

   return TProofProgressStatus(*this) -= st;
}

//______________________________________________________________________________
void TProofProgressStatus::Print(Option_t* /*option*/) const
{
   // Dump the content

   Printf("TProofProgressStatus: Entries:%lld, BytesRead:%lld, ProfTime:%.3g s, CPUTime:%.3g s",
          fEntries, fBytesRead, fProcTime, fCPUTime);
}
