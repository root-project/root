// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   7/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStatus                                                              //
//                                                                      //
// This class holds the status of an ongoing operation and collects     //
// error messages. It provides a Merge() operation allowing it to       //
// be used in PROOF to monitor status in the slaves.                    //
// No messages indicates success.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TStatus.h"
#include "Riostream.h"
#include "TClass.h"
#include "TList.h"
#include "TProofDebug.h"


ClassImp(TStatus)

//______________________________________________________________________________
TStatus::TStatus() : fVirtMemMax(-1), fResMemMax(-1)
{
   // Deafult constructor.

   SetName("PROOF_Status");
   fIter = fMsgs.begin();
}

//______________________________________________________________________________
void TStatus::Add(const char *mesg)
{
   // Add an error message.

   fMsgs.insert(mesg);
   Reset();
}

//______________________________________________________________________________
Int_t TStatus::Merge(TCollection *li)
{
   // PROOF Merge() function.

   TIter stats(li);
   PDB(kOutput,1)
      Info("Merge", "start: max virtual memory: %.2f MB \tmax resident memory: %.2f MB ",
                    GetVirtMemMax()/1024., GetResMemMax()/1024.);
   while (TObject *obj = stats()) {
      TStatus *s = dynamic_cast<TStatus*>(obj);
      if (s == 0) continue;

      MsgIter_t i = s->fMsgs.begin();
      MsgIter_t end = s->fMsgs.end();
      for (; i != end; i++)
         Add(i->c_str());
      
      SetMemValues(s->GetVirtMemMax(), s->GetResMemMax());
      PDB(kOutput,1)
         Info("Merge", "during: max virtual memory: %.2f MB \t"
                       "max resident memory: %.2f MB ",
                       GetVirtMemMax()/1024., GetResMemMax()/1024.);
   }

   return fMsgs.size();
}

//______________________________________________________________________________
void TStatus::Print(Option_t * /*option*/) const
{
   // Standard print function.

   Printf("OBJ: %s\t%s\t%s", IsA()->GetName(), GetName(), (IsOk() ? "OK" : "ERROR"));

   MsgIter_t i = fMsgs.begin();
   for (; i != fMsgs.end(); i++)
      Printf("\t%s", (*i).c_str());

   Printf(" Max virtual memory: %.2f MB \tMax resident memory: %.2f MB ",
          GetVirtMemMax()/1024., GetResMemMax()/1024.);
}

//______________________________________________________________________________
void TStatus::Reset()
{
   // Reset the iterator on the messages.

   fIter = fMsgs.begin();
}

//______________________________________________________________________________
const char *TStatus::NextMesg()
{
   // Return the next message or 0.

   if (fIter != fMsgs.end()) {
      return (*fIter++).c_str();
   } else {
      return 0;
   }
}

//______________________________________________________________________________
void TStatus::SetMemValues(Long_t vmem, Long_t rmem)
{
   // Set max memory values

   if (vmem > 0. && (fVirtMemMax < 0. || vmem > fVirtMemMax)) fVirtMemMax = vmem;
   if (rmem > 0. && (fResMemMax < 0. || rmem > fResMemMax)) fResMemMax = rmem;
}

