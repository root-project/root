// @(#)root/proof:$Id$
// Author: Jan Iwaszkiewicz    08/08/08

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofProgressStatus
#define ROOT_TProofProgressStatus

#ifndef ROOT_TObject
#include "TObject.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofProgressStatus                                                 //
//                                                                      //
// Small class including processing statistics                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TProofProgressStatus : public TObject {
public:
   enum EStatusBits { kFileNotOpen     = BIT(15),
                      kFileCorrupted   = BIT(16)
   };
private:
   Long64_t fLastEntries; // Last chunck
   Long64_t fEntries;
   Long64_t fBytesRead;
   Long64_t fReadCalls;
   Double_t fLearnTime; // the time spent in learning phase
   Double_t fLastProcTime; // the wallclock time of the last addition
   Double_t fProcTime; // the wallclock time
   Double_t fCPUTime;
   Double_t fLastUpdate; // Timestamp of last change
public:
   TProofProgressStatus(Long64_t fEntries = 0, Long64_t fBytesRead = 0,
                        Long64_t fReadCalls = 0,
                        Double_t fProcTime = 0, Double_t fCPUTime = 0);
   void     Reset() { fEntries = 0,
                      fBytesRead = 0, fProcTime = 0, fCPUTime = 0; SetLastUpdate();}

   Long64_t GetEntries() const { return fEntries; }
   Long64_t GetBytesRead() const { return fBytesRead; }
   Long64_t GetReadCalls() const { return fReadCalls; }
   Double_t GetLearnTime() const { return fLearnTime; }
   Double_t GetProcTime() const { return fProcTime; }
   Double_t GetCPUTime() const { return fCPUTime; }
   Double_t GetLastUpdate() const { return fLastUpdate; }
   Double_t GetRate() const { return ((fProcTime > 0) ? fEntries/fProcTime : 0); }
   Double_t GetCurrentRate() const;
   void     SetLastEntries(Long64_t entries) { fLastEntries = entries;}
   void     SetEntries(Long64_t entries) { fEntries = entries; SetLastUpdate();}
   void     IncEntries(Long64_t entries = 1) { fLastEntries = entries; fEntries += entries; SetLastUpdate();}
   void     IncBytesRead(Long64_t bytesRead) { fBytesRead += bytesRead; SetLastUpdate();}
   void     SetBytesRead(Long64_t bytesRead) { fBytesRead = bytesRead; SetLastUpdate();}
   void     IncReadCalls(Long64_t readCalls) { fReadCalls += readCalls; SetLastUpdate();}
   void     SetReadCalls(Long64_t readCalls) { fReadCalls = readCalls; SetLastUpdate();}
   void     SetLearnTime(Double_t learnTime) { fLearnTime = learnTime; }
   void     SetLastProcTime(Double_t procTime) { fLastProcTime = procTime; }
   void     SetProcTime(Double_t procTime) { fProcTime = procTime; }
   void     IncProcTime(Double_t procTime) { fLastProcTime = procTime; fProcTime += procTime; }
   void     SetCPUTime(Double_t procTime) { fCPUTime = procTime; }
   void     IncCPUTime(Double_t procTime) { fCPUTime += procTime; }
   void     SetLastUpdate(Double_t updtTime = 0);
   void     Print(Option_t* option = "") const;

   TProofProgressStatus operator-(TProofProgressStatus &st);
   TProofProgressStatus &operator+=(const TProofProgressStatus &st);
   TProofProgressStatus &operator-=(const TProofProgressStatus &st);

   ClassDef(TProofProgressStatus,2) // Proof progress status class
};

#endif
