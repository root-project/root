/// \file ROOT/RDrawingAttr.hxx
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

#ifndef ROOT7_RDrawingAttr
#define ROOT7_RDrawingAttr

#include <ROOT/RStyle.hxx>

#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

class RDrawingOptsBase;

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

class RDrawingAttrBase {
   /// The attribute name, as used in style files.
   std::string fName;

protected:
   const std::string &GetStyleClass(const RDrawingOptsBase& opts) const;

public:
   RDrawingAttrBase() = default;
   RDrawingAttrBase(const char* name): fName(name) {}

   const std::string& GetName() const { return fName; }
   virtual void Snapshot() = 0;
   virtual ~RDrawingAttrBase();
};

/** \class ROOT::Experimental::RDrawingAttrOrRef
 A wrapper for a graphics attribute, for instance a `RColor`.
 The `TTopmostPad` keeps track of shared attributes used by multiple drawing options by means of
 `weak_ptr`s; `RDrawingAttrOrRef`s hold `shared_ptr`s to these.
 The reference into the table of the shared attributes is wrapped into the reference of the `RDrawingAttrOrRef`
 to make them type-safe (i.e. distinct for `RColor`, `long long` and `double`).
 */

template <class ATTR>
class RDrawingAttr: public RDrawingAttrBase {
private:
   /// The shared_ptr, shared with the relevant attribute table of `TTopmostPad`.
   std::shared_ptr<ATTR> fPtr; //!

   /// The attribute value. It is authoritative if `!fPtr && !fIsDefault`, otherwise it gets
   /// updated by `Snapshot()`.
   ATTR fAttr;

   /// Whether this attribute is shared (through `TTopmostPad`'s attribute table) with other `RDrawingAttrOrRef`
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
   RDrawingAttr(RDrawingOptsBase& opts, const char *name): RDrawingAttrBase(name) {
      InitializeAttrFromString(name, RStyle::GetCurrent().GetAttribute(name, GetStyleClass(opts)), fAttr);
   }

   /// Construct a default, non-shared attribute. The default value gets read from the default style,
   /// given the attribute's name and arguments for the default attribute constructor, should no
   /// style entry be found.
   template <class...ARGS>
   RDrawingAttr(RDrawingOptsBase& opts, const char *name, ARGS... args): RDrawingAttrBase(name), fAttr(args...) {
      InitializeAttrFromString(name, RStyle::GetCurrent().GetAttribute(name, GetStyleClass(opts)), fAttr);
   }

   /// Construct a *non-shared* attribute, copying the attribute's value.
   RDrawingAttr(const RDrawingAttr &other): RDrawingAttrBase(other), fAttr(other.Get()) {}

   /// Move an attribute.
   RDrawingAttr(RDrawingAttr &&other) = default;

   /// Create a shared attribute.
   RDrawingAttr Share() {
      return GetSharedPtr();
   }

   /// Update fAttr from the value of the shared state
   void Snapshot() override {
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
   RDrawingAttr& operator=(const ATTR& attr) {
      fPtr.reset();
      fAttr = attr;
      return *this;
   }
   /// Move-assign an ATTR.
   RDrawingAttr& operator=(ATTR&& attr) {
      fPtr.reset();
      fAttr = attr;
      return *this;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RDrawingAttr
