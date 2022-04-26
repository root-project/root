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

   template <class Notifier>
   void PrependLink(Notifier &notifier)
   {
      SetBit(kLinked);

      fNext = notifier.GetNotify();
      if (auto link = dynamic_cast<TNotifyLinkBase*>(fNext)) {
         link->fPrevious = this;
      }
      notifier.SetNotify(this);
   }

   template <class Notifier>
   void RemoveLink(Notifier &notifier)
   {
      ResetBit(kLinked);

      if (notifier.GetNotify() == this) {
         R__ASSERT(fPrevious == nullptr && "The TNotifyLink head node should not have a previous element.");
         notifier.SetNotify(fNext);
      } else if (fPrevious) {
         fPrevious->fNext = fNext;
      }
      if (auto link = dynamic_cast<TNotifyLinkBase*>(fNext)) {
         link->fPrevious = fPrevious;
      }
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
   Type *fCurrent;

public:
   TNotifyLink(Type *current) : fCurrent(current) {}

   // Call Notify on the current and next object.
   Bool_t Notify() override
   {
      auto result = fCurrent ? fCurrent->Notify() : kTRUE;
      if (fNext) result &= fNext->Notify();
      return result;
   }

   ClassDefOverride(TNotifyLink, 0);
};

#endif // ROOT_TNotifyLink
