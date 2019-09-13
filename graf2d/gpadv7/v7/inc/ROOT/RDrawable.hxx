/// \file ROOT/RDrawable.hxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDrawable
#define ROOT7_RDrawable

#include <memory>
#include <string>
#include <vector>


namespace ROOT {
namespace Experimental {

class RMenuItems;
class RPadBase;

namespace Internal {
class RPadPainter;

class RIOSharedBase {
public:
   virtual const void *GetIOPtr() const = 0;
   virtual bool HasShared() const = 0;
   virtual void *MakeShared() = 0;
   virtual void SetShared(void *shared) = 0;
   virtual ~RIOSharedBase() {}
};

using RIOSharedVector_t = std::vector<RIOSharedBase *>;

template<class T>
class RIOShared final : public RIOSharedBase {
   std::shared_ptr<T>  fShared;  ///<!   holder of object
   T* fIO{nullptr};              ///<    plain pointer for IO
public:
   const void *GetIOPtr() const final { return fIO; }
   virtual bool HasShared() const final { return fShared.get() != nullptr; }
   virtual void *MakeShared() final { fShared.reset(fIO); return &fShared; }
   virtual void SetShared(void *shared) final { fShared = *((std::shared_ptr<T> *) shared); }

   RIOShared() = default;

   RIOShared(const std::shared_ptr<T> &ptr) : RIOSharedBase()
   {
      fShared = ptr;
      fIO = ptr.get();
   }

   RIOShared &operator=(const std::shared_ptr<T> &ptr)
   {
      fShared = ptr;
      fIO = ptr.get();
      return *this;
   }

   operator bool() const { return !!fShared; }

   const T *get() const { return fShared.get(); }
   T *get() { return fShared.get(); }

   void Reset() { fShared.reset(); fIO = nullptr; }

};

}

/** \class RDrawable
  Base class for drawable entities: objects that can be painted on a `RPad`.
 */

class RDrawable {
friend class RPadBase;

private:

   std::string  fId; ///< object identifier, unique inside RCanvas

protected:

   virtual void CollectShared(Internal::RIOSharedVector_t &) {}

public:

   virtual ~RDrawable();

   virtual void Paint(Internal::RPadPainter &onPad);

   /** Method can be used to provide menu items for the drawn object */
   virtual void PopulateMenu(RMenuItems &){};

   virtual void Execute(const std::string &);

   std::string GetId() const { return fId; }
};


/// Classes to solve problem with shared_ptr inside RCanvas
/// Only objects inside canvas can be


namespace Internal {

/// \class TAnyPtr
/// Models a shared pointer or a unique pointer.

template <class T>
class TUniWeakPtr {
   // Needs I/O support for union (or variant, actually) {
      std::unique_ptr<T> fUnique;
      std::weak_ptr<T> fWeak; //! Cannot save for now :-(
      T* fWeakForIO = nullptr; // Hack to allow streaming *out* of fWeak (reading is still broken because we don't set fWeak)
   // };
   bool fIsWeak = false; ///< fUnique or fWeak?

public:
   /// \class Accessor
   /// Gives transparent access to the shared or unique pointer.
   /// Locks if needed.
   class Accessor {
      union {
         T *fRaw;                    ///< The raw, non-owning pointer accessing a TUniWeak's unique_ptr
         std::shared_ptr<T> fShared; ///< The shared_ptr accessing a TUniWeak's weak_ptr
      };
      bool fIsShared; ///< fRaw or fShared?

   public:
      Accessor(const TUniWeakPtr &uniweak): fIsShared(uniweak.fIsWeak)
      {
         if (fIsShared)
            new (&fShared) std::shared_ptr<T>(uniweak.fWeak.lock());
         else
            fRaw = uniweak.fUnique.get();
      }

      Accessor(Accessor &&rhs): fIsShared(rhs.fIsShared)
      {
         if (fIsShared)
            new (&fShared) std::shared_ptr<T>(std::move(rhs.fShared));
         else
            fRaw = rhs.fRaw;
      }

      T *operator->() const { return fIsShared ? fRaw : fShared.get(); }
      T &operator*() const { return *operator->(); }
      operator bool() const { return fIsShared ? (bool)fRaw : (bool)fShared; }

      ~Accessor()
      {
         if (fIsShared)
            fShared.~shared_ptr();
      }
   };

   TUniWeakPtr() = default;
   TUniWeakPtr(const std::shared_ptr<T> &ptr): fWeak(ptr), fWeakForIO(ptr.get()), fIsWeak(true) {}
   TUniWeakPtr(std::unique_ptr<T> &&ptr): fUnique(std::move(ptr)), fIsWeak(false) {}
   TUniWeakPtr(TUniWeakPtr &&rhs): fIsWeak(rhs.fIsWeak)
   {
      if (rhs.fIsWeak) {
         fWeak = std::move(rhs.fWeak);
         auto shptr = rhs.fWeak.lock();
         fWeakForIO = shptr.get();
      } else {
         fUnique = std::move(rhs.fUnique);
      }
   }

   ~TUniWeakPtr()
   {
   }

   Accessor Get() const { return Accessor(*this); }
   void Reset()
   {
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
