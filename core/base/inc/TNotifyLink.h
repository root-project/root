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

A node in a doubly linked list of subscribers to TChain notifications.

TObject has a virtual TObject::Notify() method that takes no parameters and returns a boolean.
By default the method does nothing, and different objects in ROOT use this method for different purposes.

`TChain` uses `Notify` to implement a callback mechanism that notifies interested parties (subscribers) when
the chain switches to a new sub-tree.
In practice it calls the Notify() method of its fNotify data member from TChain::LoadTree().
However there could be several different objects interested in knowing that a given TChain switched to a new tree.
TNotifyLink can be used to build a linked list of subscribers: calling TNotifyLink::Notify() on the head
node of the list propagates the call to all subscribers in the list.

Example usage:
~~~{.cpp}
TNotifyLink l(subscriber); // subscriber must implement `Notify()`
l.PrependLink(chain); // prepends `l` to the list of notify links of the chain
~~~

\note TChain does not explicitly enforce that its fNotify data member be the head node of a list of
TNotifyLinks, but that is the case in practice at least when using TTreeReader or RDataFrame to process the chain.

\note TChain does not take ownership of the TNotifyLink and the TNotifyLink does not take ownership of the
      subscriber object.
**/

/// See TNotifyLink.
class TNotifyLinkBase : public TObject {
protected:
   /// Previous node in a TChain's list of subscribers to its notification.
   /// If null, this TNotifyLink is the head node of the list and the TChain::GetNotify() for the corresponding
   /// chain is expected to return `this`.
   TNotifyLinkBase *fPrevious = nullptr;
   /// Next node in a TChain's list of subscribers.
   /// For generality, it might be a generic TObject rather than another TNotifyLink: this makes it possible
   /// to call TChain::SetNotify() with a generic notifier exactly once before more TNotifyLinks are added.
   /// Null if this is the tail of the list.
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

   /// Set this link as the head of the chain's list of notify subscribers.
   /// Templated only to remove an include dependency from TChain: it expects
   /// a TChain as input (in practice anything that implements SetNotify and
   /// GetNotify will work, but in ROOT that is only TTree and its sub-classes).
   template <class Chain>
   void PrependLink(Chain &chain)
   {
      SetBit(kLinked);

      fNext = chain.GetNotify();
      chain.SetNotify(this);
      if (auto next = dynamic_cast<TNotifyLinkBase *>(fNext))
         next->fPrevious = this;
   }

   /// Remove this link from a chain's list of notify subscribers.
   /// Templated only to remove an include dependency from TChain: it expects
   /// a TChain as input (in practice anything that implements SetNotify and
   /// GetNotify will work, but in ROOT that is only TTree and its sub-classes).
   /// \note No error is emitted if the TNotifyLink is not part of the linked list
   /// for the chain passed as argument. The TNotifyLink will still remove itself
   /// from the doubly linked list.
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

   TObject *GetNext() const { return fNext; }

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
