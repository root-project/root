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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#include <set>
#include <string>
#ifdef R__GLOBALSTL
namespace std { using ::set; using ::string; }
#endif


class TCollection;

class TStatus : public TNamed {

private:
   typedef std::set<std::string>                 MsgSet_t;
   typedef std::set<std::string>::const_iterator MsgIter_t;
   MsgSet_t    fMsgs;   // list of error messages
   MsgIter_t   fIter;   //!iterator in messages

   Int_t       fExitStatus;  // Query exit status ((Int_t)TVirtualProofPlayer::EExitStatus or -1);
   Long_t      fVirtMemMax;  // Max virtual memory used by the worker
   Long_t      fResMemMax;   // Max resident memory used by the worker

public:
   TStatus();
   virtual ~TStatus() { }

   Bool_t         IsOk() const { return fMsgs.empty(); }
   void           Add(const char *mesg);
   virtual Int_t  Merge(TCollection *list);
   virtual void   Print(Option_t *option="") const;
   void           Reset();
   const char    *NextMesg();

   Int_t          GetExitStatus() const { return fExitStatus; }
   Long_t         GetResMemMax() const { return fResMemMax; }
   Long_t         GetVirtMemMax() const { return fVirtMemMax; }
   
   void           SetExitStatus(Int_t est) { fExitStatus = est; }
   void           SetMemValues(Long_t vmem = -1, Long_t rmem = -1);

   ClassDef(TStatus,3);  // Status class
};

#endif
