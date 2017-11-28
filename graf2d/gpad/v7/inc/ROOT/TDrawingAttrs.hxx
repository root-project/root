/// \file ROOT/TDrawingOptsBase.hxx
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

#ifndef ROOT7_TDrawingAttrs
#define ROOT7_TDrawingAttrs

#include <ROOT/TColor.hxx>

#include <RStringView.h>

#include <map>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {
class TCanvas;
class TPadBase;
class TDrawingOptsBaseNoDefault;

namespace Internal {

template <class PRIMITIVE>
class TDrawingAttrTable;
}

/** \class ROOT::Experimental::TDrawingAttrRef
 The `TCanvas` keep track of `TColor`s, integer and floating point attributes used by the drawing options,
 making them accessible from other drawing options. The index into the table of the active attributes is
 wrapped into `TDrawingAttrRef` to make them type-safe (i.e. distinct for `TColor`, `long long` and `double`).
 */

template <class PRIMITIVE>
class TDrawingAttrRef {
private:
   size_t fIdx = (size_t)-1; ///< The index in the relevant attribute table of `TCanvas`.

   /// Construct a reference given the index.
   explicit TDrawingAttrRef(size_t idx): fIdx(idx) {}

   friend class Internal::TDrawingAttrTable<PRIMITIVE>;

public:
   /// Construct an invalid reference.
   TDrawingAttrRef() = default;

   /// Construct a reference from its options object, name, default value and set of string options.
   ///
   /// Initializes the PRIMITIVE to the default value, as available in TDrawingOptsBase::GetDefaultCanvas(),
   /// or to `deflt` if no entry exists under the attribute name. The attribute name is `opts.GetName() + "." + attrName`.
   /// `optStrings` is only be used if `PRIMITIVE` is `long long`; the style setting is expected to be one of
   /// the strings, with the attribute's value the index of the string.
   TDrawingAttrRef(TDrawingOptsBaseNoDefault &opts, const std::string &attrName, const PRIMITIVE &deflt,
                   const std::vector<std::string_view> &optStrings = {});

   /// Get the underlying index.
   operator size_t() const { return fIdx; }

   /// Whether the reference is valid.
   explicit operator bool() const { return fIdx != (size_t)-1; }
};

extern template class TDrawingAttrRef<TColor>;
extern template class TDrawingAttrRef<long long>;
extern template class TDrawingAttrRef<double>;

namespace Internal {

template <class PRIMITIVE>
class TDrawingAttrAndUseCount {
   /// The value.
   PRIMITIVE fVal;
   /// The value's use count.
   size_t fUseCount = 1;

   /// Clear the value; use count must be 0.
   void Clear();

public:
   /// Default constructor: a default-constructed value that is unused.
   TDrawingAttrAndUseCount(): fVal(), fUseCount(0) {}

   /// Initialize with a value, setting use count to 1.
   explicit TDrawingAttrAndUseCount(const PRIMITIVE &val): fVal(val) {}

   /// Create a value, initializing use count to 1.
   void Create(const PRIMITIVE &val);

   /// Increase the use count; use count must be >= 1 before (i.e. does not create or "resurrect" values).
   void IncrUse();

   /// Decrease the use count; use count must be >= 1 before the call. Calls Clear() if use count drops to 0.
   void DecrUse();

   /// Whether the use count is 0 and this object has space for a new value.
   bool IsFree() const { return fUseCount == 0; }

   /// Value access (non-const).
   PRIMITIVE &Get() { return fVal; }

   /// Value access (const).
   const PRIMITIVE &Get() const { return fVal; }
};

// Only these specializations are used and provided:
extern template class TDrawingAttrAndUseCount<TColor>;
extern template class TDrawingAttrAndUseCount<long long>;
extern template class TDrawingAttrAndUseCount<double>;

template <class PRIMITIVE>
class TDrawingAttrTable {
public:
   using value_type = Internal::TDrawingAttrAndUseCount<PRIMITIVE>;

private:
   /// Table of attribute primitives. Slots can be freed and re-used.
   /// Drawing options will reference slots in here through their index.
   std::vector<value_type> fTable;

public:
   /// Register an attribute with the table.
   /// \returns the index in the table.
   TDrawingAttrRef<PRIMITIVE> Register(const PRIMITIVE &val);

   /// Add a use of the attribute at table index idx.
   void IncrUse(TDrawingAttrRef<PRIMITIVE> idx) { fTable[idx].IncrUse(); }

   /// Remove a use of the attribute at table index idx.
   void DecrUse(TDrawingAttrRef<PRIMITIVE> idx) { fTable[idx].DecrUse(); }

   /// Update an existing attribute entry in the table.
   void Update(TDrawingAttrRef<PRIMITIVE> idx, const PRIMITIVE &val) { fTable[idx] = value_type(val); }

   /// Get the value at index `idx` (const version).
   const PRIMITIVE &Get(TDrawingAttrRef<PRIMITIVE> idx) const { return fTable[idx].Get(); }

   /// Get the value at index `idx` (non-const version).
   PRIMITIVE &Get(TDrawingAttrRef<PRIMITIVE> idx) { return fTable[idx].Get(); }

   /// Find the index belonging to the attribute at the given address and add a use.
   /// \returns the reference to `val`, which might be `IsInvalid()` if `val` is not part of this table.
   TDrawingAttrRef<PRIMITIVE> SameAs(const PRIMITIVE &val);

   /// Access to the underlying attribute table (non-const version).
   std::vector<value_type> &GetTable() { return fTable; }

   /// Access to the underlying attribute table (const version).
   const std::vector<value_type> &GetTable() const { return fTable; }
};

extern template class TDrawingAttrTable<TColor>;
extern template class TDrawingAttrTable<long long>;
extern template class TDrawingAttrTable<double>;
} // namespace Internal

struct TLineAttrs {
   TDrawingAttrRef<TColor> fColor;    ///< Line color
   TDrawingAttrRef<long long> fWidth; ///< Line width

   struct Width {
      long long fVal;
      explicit operator long long() const { return fVal; }
   };

   TLineAttrs() = default;
   /// Construct from the pad that holds our `TDrawable`, passing the configuration name of this line attribute.
   TLineAttrs(TDrawingOptsBaseNoDefault &opts, const std::string &name, const TColor &col, Width width)
      : fColor(opts, name + ".Color", col), fWidth(opts, name + ".Width", (long long)width)
   {}
};

struct TFillAttrs {
   TDrawingAttrRef<TColor> fColor; ///< Fill color

   TFillAttrs() = default;

   /// Construct from the pad that holds our `TDrawable`, passing the configuration name of this line attribute.
   TFillAttrs(TDrawingOptsBaseNoDefault &opts, const std::string &name, const TColor &col)
      : fColor(opts, name + ".Color", col)
   {}
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_TDrawingAttrs
