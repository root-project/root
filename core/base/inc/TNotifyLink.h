// @(#)root/base:$Id$
// Author: Philippe Canal 2019

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNotifyLink
#define ROOT_TNotifyLink

#include <TObject.h>
#include <TError.h> // for R__ASSERT

/** \class TNotifyLink
\ingroup Base

Links multiple listeners to be notified on TChain file changes.

Neither TChain::SetNotify() nor this TNotifyLink take ownership of the object to be notified.

eg.
```
auto notify = new TNotifyLink(object, fChain->GetNotify());
fChain->SetNotify(notify);
```
**/

class TNotifyLinkBase : public TObject {
protected:
   TNotifyLinkBase *fPrevious = nullptr;
   TObject         *fNext = nullptr;

public:
   // TTree status bits
   enum EStatusBits {
      kLinked = BIT(11) // Used when the TNotifyLink is connected to a TTree.
   };

   void Clear(Option_t * /*option*/ ="") override
   {
      auto current = this;
      do {
         auto next = dynamic_cast<TNotifyLinkBase*>(fNext);
         current->ResetBit(kLinked);
         current->fPrevious = nullptr;
         current->fNext = nullptr;
         current = next;
      } while(current);
   }

   template <class Chain>
   void PrependLink(Chain &chain)
   {
      SetBit(kLinked);

      fNext = chain.GetNotify();
      chain.SetNotify(this);
      if (auto next = dynamic_cast<TNotifyLinkBase *>(fNext))
         next->fPrevious = this;
   }

   template <class Chain>
   void RemoveLink(Chain &chain)
   {
      ResetBit(kLinked);

      if (chain.GetNotify() == this) { // this notify link is the first in the list
         R__ASSERT(fPrevious == nullptr && "The TNotifyLink head node should not have a previous element.");
         chain.SetNotify(fNext);
      } else if (fPrevious) {
         fPrevious->fNext = fNext;
      }
      if (auto next = dynamic_cast<TNotifyLinkBase *>(fNext))
         next->fPrevious = fPrevious;
      fPrevious = nullptr;
      fNext = nullptr;
   }

   Bool_t IsLinked()
   {
      return TestBit(kLinked);
   }

   ClassDefOverride(TNotifyLinkBase, 0);
};

template <class Type>
class TNotifyLink : public TNotifyLinkBase {
private:
   Type *fSubscriber;

public:
   TNotifyLink(Type *subscriber) : fSubscriber(subscriber) {}

   /// Call Notify on our subscriber and propagate the call to the next link.
   Bool_t Notify() override
   {
      bool result = true;
      if (fSubscriber)
         result &= fSubscriber->Notify();
      if (fNext)
         result &= fNext->Notify();
      return result;
   }

   ClassDefOverride(TNotifyLink, 0);
};

#endif // ROOT_TNotifyLink
