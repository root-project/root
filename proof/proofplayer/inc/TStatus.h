// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   12/03/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStatus
#define ROOT_TStatus

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStatus                                                              //
//                                                                      //
// This class holds the status of a ongoing operation and collects      //
// error messages. It provides a Merge() operation allowing it to       //
// be used in PROOF to monitor status in the slaves.                    //
// No messages indicates success.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "THashList.h"

#include <set>
#include <string>

class TStatus : public TNamed {

public:
   enum EProcStatus {
      kNotOk = BIT(15)       // True if status of things are not OK
   };

private:
   TList       fMsgs;     // list of error messages
   TIter       fIter;     //!iterator in messages
   THashList   fInfoMsgs; // list of info messages

   Int_t       fExitStatus;  // Query exit status ((Int_t)TVirtualProofPlayer::EExitStatus or -1);
   Long_t      fVirtMemMax;  // Max virtual memory used by the worker
   Long_t      fResMemMax;   // Max resident memory used by the worker
   Long_t      fVirtMaxMst;  // Max virtual memory used by the master
   Long_t      fResMaxMst;   // Max resident memory used by the master

public:
   TStatus();
   virtual ~TStatus() { }

   inline Bool_t  IsOk() const { return TestBit(kNotOk) ? kFALSE : kTRUE; }
   void           Add(const char *mesg);
   void           AddInfo(const char *mesg);
   virtual Int_t  Merge(TCollection *list);
   virtual void   Print(Option_t *option="") const;
   void           Reset();
   const char    *NextMesg();

   Int_t          GetExitStatus() const { return fExitStatus; }
   Long_t         GetResMemMax(Bool_t master = kFALSE) const { return ((master) ? fResMaxMst : fResMemMax); }
   Long_t         GetVirtMemMax(Bool_t master = kFALSE) const { return ((master) ? fVirtMaxMst : fVirtMemMax); }

   void           SetExitStatus(Int_t est) { fExitStatus = est; }
   void           SetMemValues(Long_t vmem = -1, Long_t rmem = -1, Bool_t master = kFALSE);

   ClassDef(TStatus,5);  // Status class
};

#endif
