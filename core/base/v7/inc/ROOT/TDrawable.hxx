/// \file ROOT/TDrawable.h
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDrawable
#define ROOT7_TDrawable

#include <memory>

namespace ROOT {
namespace Experimental {
class TCanvas;

namespace Internal {

/** \class TDrawable
  Base class for drawable entities: objects that can be painted on a `TPad`.
  */

class TDrawable {
public:
  virtual ~TDrawable();

  /// Paint the object
  virtual void Paint(TCanvas& onCanv) = 0;
};

/// \class TAnyPtr
/// Models a shared pointer or a unique pointer.

template <class T>
class TUniWeakPtr {
   union {
      std::unique_ptr<T> fUnique;
      std::weak_ptr<T> fWeak;
   };
   bool fIsWeak; ///< fUnique or fWeak?

public:
   /// \class Accessor
   /// Gives transparent access to the shared or unique pointer.
   /// Locks if needed.
   class Accessor {
      union {
         std::shared_ptr<T> fShared;
         T* fRaw;
      };
      bool fIsShared;  ///< fRaw or fShared?

   public:
      Accessor(const TUniWeakPtr& uniweak):
         fIsShared(uniweak.fIsWeak) {
         if (fIsShared)
            fShared = uniweak.fWeak.lock();
         else
            fRaw = uniweak.fUnique.get();
      }

      Accessor(Accessor&& rhs): fIsShared(rhs.fIsShared) {
         if (fIsShared)
            fShared.swap(rhs.fShared);
         else
            fRaw = rhs.fRaw;
      }

      T* operator->() const { return fIsShared ? fRaw : fShared.get(); }
      T& operator*() const { return *operator->(); }
      operator bool() const { return fIsShared ? (bool)fRaw : (bool)fShared; }

      ~Accessor() {
         if (fIsShared)
            fShared.~shared_ptr();
      }
   };

   TUniWeakPtr(const std::shared_ptr<T>& ptr): fWeak(ptr), fIsWeak(true) {}
   TUniWeakPtr(std::unique_ptr<T>&& ptr): fUnique(std::move(ptr)), fIsWeak(false) {}
   TUniWeakPtr(TUniWeakPtr&& rhs): fIsWeak(rhs.fIsWeak) {
      if (fIsWeak)
         fWeak.swap(rhs.fWeak);
      else
         fUnique.swap(rhs.fUnique);
   }

   ~TUniWeakPtr() {
      if (fIsWeak)
         fWeak.~weak_ptr();
      else
         fUnique.~unique_ptr();
   }

   Accessor Get() const { return Accessor(*this); }
   void Reset() {
      if (fIsWeak)
         fWeak.reset();
      else
         fUnique.reset();
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
