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


ClassImp(TStatus)

//______________________________________________________________________________
TStatus::TStatus()
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
   while (TObject *obj = stats()) {
      TStatus *s = dynamic_cast<TStatus*>(obj);
      if (s == 0) continue;

      MsgIter_t i = s->fMsgs.begin();
      MsgIter_t end = s->fMsgs.end();
      for (; i != end; i++)
         Add(i->c_str());
   }

   return fMsgs.size();
}

//______________________________________________________________________________
void TStatus::Print(Option_t * /*option*/) const
{
   // Standard print function.

   cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName()
        << "\t" << (IsOk() ? "OK" : "ERROR") << endl;

   MsgIter_t i = fMsgs.begin();
   for (; i != fMsgs.end(); i++)
      cout << "\t" << *i << endl;
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


