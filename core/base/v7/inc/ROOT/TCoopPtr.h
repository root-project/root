/// \file TCoopPtr
/// \ingroup Base
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-31

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TCoopPtr
#define ROOT7_TCoopPtr

#include <memory>
#include <type_traits>

namespace ROOT {

/**
  \class TCoopPtr
  \brief Several pointers point to the same object, any of them can delete
  the object, setting all of them to `nullptr`. Once all TCoopPtr-s have been
  destructed, the pointed-to object is also destructed.

  Internally, this is modelled as a `shared_ptr<unique_ptr<T>>`: all
  shared_ptr-s point to the same pointer (unique_ptr, actually), and a change to
  that pointer value is visible to all of them.

  This is a PAW-style ownership management, useful if deletion needs to be an
  explicit action (unlike `shared_ptr`), but shared resource handling is
  required (like `unique_ptr`).

  Example:
      TCoopPtr<TH1D> pHist(new TH1D({{10, 0., 1.}}));
      TCoopPtr<TCanvas> pad = gROOT.Create<TCanvas>();
      pad->Draw(pHist); // histogram is drawn; the pad co-owns the histogram
      pHist.Delete(); // histogram vanishes from pad.
      // Or the histogram could have been deleted on the pad, interactively.
**/

template <class POINTEE>
class TCoopPtr {
public:
  /// The type pointed to
  using Pointee_t = std::remove_reference_t<POINTEE>;
  /// The primitive pointer type
  using Pointer_t = std::add_pointer_t<Pointee_t>;
  /// The underlying std::shared_ptr
  using SharedPtr_t = std::shared_ptr<void*>;

  TCoopPtr() = default;
  /// Creates a synchronized TCoopPtr: once any of them call Delete(), all others
  /// will "contain" a `nullptr`. The last synchronized TCoopPtr to be destructed
  /// will delete the object they pointed to.
  TCoopPtr(const TCoopPtr& other) = default;
  TCoopPtr(TCoopPtr&& other) = default;

  /// Initialize from a raw pointer.
  TCoopPtr(Pointer_t ptr):
     fPtrBuf(std::make_shared<void*>(ptr)) {}

  /// Create a TCoopPtr from an object. Tries to invoke POINTEE's move
  /// constructor. Is only available if POINTEE is move constructable.
  template <class SELF = TCoopPtr,
     typename std::enable_if<
      std::is_move_constructible<typename SELF::Pointee_t>::value
     >::type* = nullptr>
  TCoopPtr(POINTEE&& obj):
     fPtrBuf(std::make_shared<void*>(new POINTEE(std::move(obj))))
  {}

  /// Initialize from a unique_ptr; takes ownership.
  ///
  /// Call as `TCoopPtr p(std::move(uniquePtr));`
  TCoopPtr(std::unique_ptr<POINTEE>&& ptr):
     TCoopPtr(ptr.release()) {}

  /// Conversion from a pointer to derived to a pointer to base.
  template <class DERIVED,
     typename std::enable_if<std::is_base_of<POINTEE, DERIVED>{}>::type* = nullptr>
  TCoopPtr(const TCoopPtr<DERIVED>& derived) {
    fPtrBuf = std::static_pointer_cast<void*>(derived.GetShared());
  }

  /// Get the raw pointer.
  Pointer_t Get() const {
    return fPtrBuf.get() ? static_cast<Pointer_t>(*fPtrBuf.get())
                         : nullptr;
  }
  const SharedPtr_t& GetShared() const { return fPtrBuf; }

  /// Delete the object pointed to. Safe to be called mutiple times.
  /// All other synchronized TCoopPtr will be notified.
  POINTEE* Delete() const {
    delete fPtrBuf.get();
    fPtrBuf.get() = nullptr;
  }

  /// Access the object pointed to.
  Pointer_t operator->() const { return Get(); }
  /// Dereference the object pointed to.
  POINTEE& operator*() const { return *Get(); }
  /// Returns `true` if the pointer is non-null.
  operator bool() const { return fPtrBuf.get(); }

private:
  SharedPtr_t fPtrBuf;
};


/// Create an object of type `POINTEE` on the heap, and build a `TCoopPtr` for
/// it.
///
/// \param args Arguments forwarded to the constructor of `POINTEE`.
template <class POINTEE, class... ARGS>
TCoopPtr<POINTEE> MakeCoop(ARGS&&... args) {
  return TCoopPtr<POINTEE>(new POINTEE(std::forward<ARGS>(args)...));
};


/// Move an object into a TCoopPtr. Rather efficient if there is a move
/// constructor for `POINTEE`.
///
/// \param obj - object to be moved into the TCoopPtr.
template <class POINTEE>
TCoopPtr<POINTEE> MakeCoop(POINTEE&& obj) {
  return TCoopPtr<POINTEE>(std::move(obj));
};


namespace Internal {
/**
 \class TCoopPtrTypeErasedBase
 To handle polymorphic `TCoopPtr<POINTEE>`s, convert them to
 `TCoopPtrTypeErased<POINTEE>`s and access them through their common base
 `TCoopPtrTypeErasedBase`. Example:

     auto pH1D = MakeCoOwnedHist<1, double>({{{10, 0., 1.}}});
     auto pH2I = MakeCoOwnedHist<2, int>({{{2, 0., 1.}, {2., 0., 1.}}});
     std::vector<Internal::TCoopPtrTypeErased<TDrawable>>
       coOwnedDrawables {pH1D, pH2I};

 The contained objects will be destructed once all TCoopPtrTypeErased and all
 other synchronized `TCoopPtr` are destructed.
 */
struct TCoopPtrTypeErasedBase {
  virtual ~TCoopPtrTypeErasedBase() {};
};


/**
 \class TCoopPtrTypeErased
 Type-specific derived implementation of a `TCoopPtrTypeErasedBase`, to invoke
 the destructor of the `TCoopPtr<POINTEE>` upon destruction of the
 `TCoopPtrTypeErased`. "Translates" from the abstract interface
 (`TCoopPtrTypeErasedBase`) to the specific implementation `TCoopPtr<POINTEE>`.
 */
template<class POINTEE>
class TCoopPtrTypeErased: public TCoopPtr<POINTEE>, public TCoopPtrTypeErasedBase {
public:
  TCoopPtrTypeErased(const TCoopPtr<POINTEE>& ptr): TCoopPtr<POINTEE>(ptr) {}
  ~TCoopPtrTypeErased() final = default;
};
} // namespace Internal
}

#endif
