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
#include <unordered_map>


namespace ROOT {
namespace Experimental {

class RDrawingOptsBase;
class RMenuItems;
class RPadBase;
class RDrawable;


namespace Internal {
class RPadPainter;
}

/** \class RDrawable
  Base class for drawable entities: objects that can be painted on a `RPad`.
 */


using RDrawableAttributesContainer = std::unordered_map<std::string, std::string>;

/** Access to drawable attributes, never should be stored */
class RDrawableAttributesNew {
   RDrawable &fDrawable; ///<! reference

   std::string fPrefix; ///<! name prefix for all attributes values

   const RDrawableAttributesContainer *fDefaults{nullptr}; ///<! default values for these attributes

   std::string GetFullName(const std::string &name) const { return fPrefix.empty() ? name : fPrefix + "." + name; }

protected:

   /** Should be used in the constructor */
   void SetDefaults(const RDrawableAttributesContainer &dflts) { fDefaults = &dflts; }

public:

   RDrawableAttributesNew(RDrawable &d, const std::string &prefix);
   virtual ~RDrawableAttributesNew() {}

   /** use const char* - nullptr means no value found */
   const char *Eval(const std::string &name) const;

   /** returns true when value exists */
   bool HasValue(const std::string &name) const { return Eval(name) != nullptr; }

   void SetValue(const std::string &name, const char *val);

   void SetValue(const std::string &name, const std::string &value);

   void ClearValue(const std::string &name) { SetValue(name, (const char *)nullptr); }

   void Clear();

   int GetInt(const std::string &name) const;
   void SetInt(const std::string &name, const int value);

   float GetFloat(const std::string &name) const;
   void SetFloat(const std::string &name, const float value);
};

class RDrawableDisplayItem;

class RDrawable {
friend class RPadBase;
friend class RDrawableAttributesNew;
friend class RDrawableDisplayItem;

private:

   std::string  fId; ///< object identifier, unique inside RCanvas

   RDrawableAttributesContainer fNewAttributes; ///< container for any kind of attribute associated with drawable, attributes can be styled, store as JSON_object

   const RDrawableAttributesContainer *fDefaults{nullptr}; ///<! default values for drawable attributes

protected:

   void ClearAttributes();

   void SetDefaults(const RDrawableAttributesContainer &dflts) { fDefaults = &dflts; }

public:

   RDrawable();

   virtual ~RDrawable();

   virtual void Paint(Internal::RPadPainter &onPad) = 0;

   /** Method can be used to provide menu items for the drawn object */
   virtual void PopulateMenu(RMenuItems &){};

   virtual void Execute(const std::string &);

   /// Get the reference to the drawing options as RDrawingOptsBase. Used e.g. to identify the RDrawable in
   /// the list of primitives.
   virtual RDrawingOptsBase& GetOptionsBase() = 0;

   std::string GetId() const { return fId; }

};

template <class DERIVED>
class RDrawableBase: public RDrawable {
public:
   RDrawingOptsBase& GetOptionsBase() override { return static_cast<DERIVED*>(this)->GetOptions(); }
};

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
