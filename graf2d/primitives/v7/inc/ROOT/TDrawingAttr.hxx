/// \file ROOT/TDrawingAttr.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDrawingAttr
#define ROOT7_TDrawingAttr

#include <ROOT/TStyle.hxx>

#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

class TDrawingOptsBase;

///\{
/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name, for diagnostic purposes.
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.
void InitializeAttrFromString(const std::string &name, const std::string &strval, int& val);
void InitializeAttrFromString(const std::string &name, const std::string &strval, long long& val);
void InitializeAttrFromString(const std::string &name, const std::string &strval, float& val);
void InitializeAttrFromString(const std::string &name, const std::string &strval, std::string& val);
///\}

/** \class ROOT::Experimental::TDrawingAttrOrRef
 A wrapper for a graphics attribute, for instance a `TColor`.
 The `TTopmostPad` keeps track of shared attributes used by multiple drawing options by means of
 `weak_ptr`s; `TDrawingAttrOrRef`s hold `shared_ptr`s to these.
 The reference into the table of the shared attributes is wrapped into the reference of the `TDrawingAttrOrRef`
 to make them type-safe (i.e. distinct for `TColor`, `long long` and `double`).
 */

template <class ATTR>
class TDrawingAttrValOrRef {
private:
   /// The shared_ptr, shared with the relevant attribute table of `TTopmostPad`.
   std::shared_ptr<ATTR> fPtr; //!
   ATTR fAttr; ///< The attribute value, used if `!fPtr`.

   /// Whether this attribute is shared (through `TTopmostPad`'s attribute table) with other `TDrawingAttrOrRef`
   /// objects.
   bool IsShared() const { return (bool) fPtr; }

   /// Share the attribute, potentially transforming this into a shared attribute.
   std::shared_ptr<ATTR> GetSharedPtr() {
      if (!IsShared())
         fPtr = std::make_shared<ATTR>(std::move(fAttr));
      return fPtr;
   }

public:
   /// Construct a default, non-shared attribute. The default value gets read from the default style,
   /// given the attribute's name.
   TDrawingAttrValOrRef(const char *name) {
      InitializeAttrFromString(name, TStyle::GetCurrent().GetAttribute(name), fAttr);
   }

   /// Construct a default, non-shared attribute. The default value gets read from the default style,
   /// given the attribute's name and arguments for the default attribute constructor, should no
   /// style entry be found.
   template <class...ARGS>
   TDrawingAttrValOrRef(const char *name, ARGS... args): fAttr(args...) {
      InitializeAttrFromString(name, TStyle::GetCurrent().GetAttribute(name), fAttr);
   }

   /// Construct a *non-shared* attribute, copying the attribute's value.
   TDrawingAttrValOrRef(const TDrawingAttrValOrRef &other): fAttr(other.Get()) {}

   /// Move an attribute.
   TDrawingAttrValOrRef(TDrawingAttrValOrRef &&other) = default;

   /// Create a shared attribute.
   TDrawingAttrValOrRef Share() {
      return GetSharedPtr();
   }

   /// Update fAttr from the value of the shared state
   void SyncFromShared() {
      if (IsShared())
         fAttr = Get();
   }

   /// Get the const attribute, whether it's shared or not.
   const ATTR &Get() const {
      if (IsShared())
         return *fPtr;
      return fAttr;
   }

   /// Get the non-const attribute, whether it's shared or not.
   ATTR &Get() {
      if (IsShared())
         return *fPtr;
      return fAttr;
   }

   /// Convert to an ATTR (const).
   explicit operator const ATTR& () const{ return Get(); }
   /// Convert to an ATTR (non-const).
   explicit operator ATTR& () { return Get(); }

   /// Assign an ATTR.
   TDrawingAttrValOrRef& operator=(const ATTR& attr) { Get() = attr; return *this; }
   /// Move-assign an ATTR.
   TDrawingAttrValOrRef& operator=(ATTR&& attr) { Get() = std::move(attr); return *this; }
};

class TDrawingAttrBase {
protected:
   void Register(TDrawingOptsBase& owner, const char *name);

public:
   virtual void SyncFromShared() = 0;
   virtual ~TDrawingAttrBase();
};

template <class ATTR>
class TDrawingAttr: public TDrawingAttrBase {
private:
   TDrawingAttrValOrRef<ATTR> fAttr;

public:
   /// Initialize TDrawingAttr from a string literal.
   template <int N>
   TDrawingAttr(TDrawingOptsBase& owner, const char (&name)[N]): fAttr(name)
   {
      Register(owner, name);
   }

   template <int N, class...ARGS>
   TDrawingAttr(TDrawingOptsBase& owner, const char (&name)[N], ARGS... args): fAttr(name, args...)
   {
      Register(owner, name);
   }

   TDrawingAttr(const TDrawingAttr&) = default;
   TDrawingAttr(TDrawingAttr&&) = default;
   TDrawingAttr& operator=(const TDrawingAttr&) = default;
   TDrawingAttr& operator=(TDrawingAttr&&) = default;

   /// Create a shared attribute.
   TDrawingAttrValOrRef<ATTR> Share() { return fAttr.GetSharedPtr(); }

   /// Update fAttr from the value of the shared state
   void SyncFromShared() final { fAttr.SyncFromShared(); }

   /// Get the const attribute, whether it's shared or not.
   const ATTR &Get() const { return fAttr.Get(); }

   /// Get the non-const attribute, whether it's shared or not.
   ATTR &Get() { return fAttr.Get(); }

   /// Convert to an ATTR (const).
   explicit operator const ATTR& () const{ return fAttr.Get(); }
   /// Convert to an ATTR (non-const).
   explicit operator ATTR& () { return fAttr.Get(); }

   /// Assign an ATTR.
   TDrawingAttr& operator=(const ATTR& attr) { Get() = attr; return *this; }
   /// Move-assign an ATTR.
   TDrawingAttr& operator=(ATTR&& attr) { Get() = std::move(attr); return *this; }

   /// Assign an ATTR.
   TDrawingAttr& operator=(const TDrawingAttrValOrRef<ATTR>& attr) { fAttr = attr; return *this; }
   /// Move-assign an ATTR.
   TDrawingAttr& operator=(TDrawingAttrValOrRef<ATTR>&& attr) { fAttr = std::move(attr); return *this; }
};


} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_TDrawingAttr
