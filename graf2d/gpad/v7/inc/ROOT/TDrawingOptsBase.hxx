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

#ifndef ROOT7_TDrawingOptsBase
#define ROOT7_TDrawingOptsBase

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
class TOptsAttrTable;
}

/** class ROOT::Experimental::TOptsAttrRef
 The `TCanvas` keep track of `TColor`s, integer and floating point attributes used by the drawing options,
 making them accessible from other drawing options. The index into the table of the active attributes is
 wrapped into `TOptsAttrRef` to make them type-safe (i.e. distinct for `TColor`, `long long` and `double`).
 */

template <class PRIMITIVE>
class TOptsAttrRef {
private:
   size_t fIdx = (size_t)-1; ///<! The index in the relevant attribute table of `TCanvas`.

   /// Construct a reference given the index.
   explicit TOptsAttrRef(size_t idx): fIdx(idx) {}

   friend class Internal::TOptsAttrTable<PRIMITIVE>;

public:
   /// Construct an invalid reference.
   TOptsAttrRef() = default;

   /// Construct a reference from its options object, name, and set of string options.
   /// Initialized the PRIMITIVE to the default value, as available in TDrawingOptsBase::GetDefaultCanvas().
   /// The value of this attribute will be the index in the vector of strings; the default value is parsed by
   /// finding the configured string in the vector of strings. `optStrings` will only be used if
   /// `PRIMITIVE` is `long long`.
   TOptsAttrRef(TDrawingOptsBaseNoDefault &opts, std::string_view name,
                const std::vector<std::string_view> &optStrings = {});

   /// Get the underlying index.
   operator size_t() const { return fIdx; }

   /// Whether the reference is valid.
   explicit operator bool() const { return fIdx != (size_t)-1; }
};

extern template class TOptsAttrRef<TColor>;
extern template class TOptsAttrRef<long long>;
extern template class TOptsAttrRef<double>;

namespace Internal {

template <class PRIMITIVE>
class TOptsAttrAndUseCount {
   /// The value.
   PRIMITIVE fVal;
   /// The value's use count.
   size_t fUseCount = 1;

   /// Clear the value; use count must be 0.
   void Clear();

public:
   /// Initialize with a value, setting use count to 1.
   TOptsAttrAndUseCount(const PRIMITIVE &val): fVal(val) {}

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
extern template class TOptsAttrAndUseCount<TColor>;
extern template class TOptsAttrAndUseCount<long long>;
extern template class TOptsAttrAndUseCount<double>;

template <class PRIMITIVE>
class TOptsAttrTable {
public:
   using value_type = Internal::TOptsAttrAndUseCount<PRIMITIVE>;

private:
   /// Table of attribute primitives. Slots can be freed and re-used.
   /// Drawing options will reference slots in here through their index.
   std::vector<value_type> fTable;

public:
   /// Register an attribute with the table.
   /// \returns the index in the table.
   TOptsAttrRef<PRIMITIVE> Register(const PRIMITIVE &val);

   /// Add a use of the attribute at table index idx.
   void IncrUse(TOptsAttrRef<PRIMITIVE> idx) { fTable[idx].IncrUse(); }

   /// Remove a use of the attribute at table index idx.
   void DecrUse(TOptsAttrRef<PRIMITIVE> idx) { fTable[idx].DecrUse(); }

   /// Update an existing attribute entry in the table.
   void Update(TOptsAttrRef<PRIMITIVE> idx, const PRIMITIVE &val) { fTable[idx] = val; }

   /// Get the value at index `idx` (const version).
   const PRIMITIVE &Get(TOptsAttrRef<PRIMITIVE> idx) const { return fTable[idx].Get(); }

   /// Get the value at index `idx` (non-const version).
   PRIMITIVE &Get(TOptsAttrRef<PRIMITIVE> idx) { return fTable[idx].Get(); }

   /// Find the index belonging to the attribute at the given address and add a use.
   /// \returns the reference to `val`, which might be `IsInvalid()` if `val` is not part of this table.
   TOptsAttrRef<PRIMITIVE> SameAs(const PRIMITIVE &val);

   /// Access to the underlying attribute table (non-const version).
   std::vector<value_type>& GetTable() { return fTable; }

   /// Access to the underlying attribute table (const version).
   const std::vector<value_type>& GetTable() const { return fTable; }
};

extern template class TOptsAttrTable<TColor>;
extern template class TOptsAttrTable<long long>;
extern template class TOptsAttrTable<double>;
} // namespace Internal

/** \class ROOT::Experimental::TDrawingOptsBase
  Base class for drawing options. Implements access to the default and the default's initialization
  from a config file.

  Drawing options are made up of three kinds of primitives:
    - TColor
    - integer (long long)
    - floating point (double)
  They register the primitives with their `TCanvas`, and then refer to the registered primitives.
  Upon destruction, the `TDrawingOptsBase` deregisters the use of its primitives with the `TCanvas`,
  which empties the respective slots in the tables of the `TCanvas`, unless other options reference
  the same primitives (through `SameAs()`), and until the last use has deregistered.

  In derived classes (e.g. drawing options for the class `MyFancyBox`), declare attribute members as
     TOptsAttrRef<TColor> fLineColor{*this, "MyFancyBox.LineColor"};
  The attribute's value will be taken from the  will be initialized
  */

class TDrawingOptsBaseNoDefault {
public:
   template <class PRIMITIVE>
   class OptsAttrRefArr {
      /// Indexes of the `TCanvas`'s attribute table entries used by the options object.
      std::vector<TOptsAttrRef<PRIMITIVE>> fRefArray;

   public:
      ~OptsAttrRefArr();
      /// Register an attribute.
      ///\returns the index of the new attribute.
      TOptsAttrRef<PRIMITIVE> Register(TCanvas &canv, const PRIMITIVE &val);

      /// Re-use an existing attribute.
      ///\returns the index of the attribute (i.e. valRef).
      TOptsAttrRef<PRIMITIVE> SameAs(TCanvas &canv, TOptsAttrRef<PRIMITIVE> idx);

      /// Re-use an existing attribute.
      ///\returns the index of the attribute, might be `IsInvalid()` if `val` could not be found.
      TOptsAttrRef<PRIMITIVE> SameAs(TCanvas &canv, const PRIMITIVE &val);

      /// Update the attribute at index `idx` to the value `val`.
      void Update(TCanvas &canv, TOptsAttrRef<PRIMITIVE> idx, const PRIMITIVE &val);

      /// Clear all attribute references, removing their uses in `TCanvas`.
      void Release(TCanvas &canv);

      /// Once copied, elements of a OptsAttrRefArr need to increase their use count.
      void RegisterCopy(TCanvas &canv);
   };

private:
   /// The `TCanvas` holding the `TDrawable` (or its `TPad`).
   TCanvas *fCanvas; //! always != nullptr

   /// Indexes of the `TCanvas`'s color table entries used by this options object.
   OptsAttrRefArr<TColor> fColorIdx;

   /// Indexes of the `TCanvas`'s integer table entries used by this options object.
   OptsAttrRefArr<long long> fIntIdx;

   /// Indexes of the `TCanvas`'s floating point table entries used by this options object.
   OptsAttrRefArr<double> fFPIdx;

   /// Access to the attribute tables (non-const version).
   OptsAttrRefArr<TColor> &GetAttrsRefArr(TColor*) { return fColorIdx; }
   OptsAttrRefArr<long long> &GetAttrsRefArr(long long*) { return fIntIdx; }
   OptsAttrRefArr<double> &GetAttrsRefArr(double*) { return fFPIdx; }

   /// Access to the attribute tables (const version).
   const OptsAttrRefArr<TColor> &GetAttrsRefArr(TColor*) const  { return fColorIdx; }
   const OptsAttrRefArr<long long> &GetAttrsRefArr(long long*) const { return fIntIdx; }
   const OptsAttrRefArr<double> &GetAttrsRefArr(double*) const { return fFPIdx; }

protected:
   /// Construct from the pad that holds our `TDrawable`.
   TDrawingOptsBaseNoDefault(TPadBase &pad);

   /// Default attributes need to register their values in a pad - they will take this pad!
   static TPadBase &GetDefaultCanvas();

   /// The `TCanvas` holding the `TDrawable` (or its `TPad`).
   TCanvas &GetCanvas() { return *fCanvas; }

   template <class PRIMITIVE>
   TOptsAttrRef<PRIMITIVE> Register(const PRIMITIVE &val)
   {
      return GetAttrsRefArr((PRIMITIVE*)nullptr).Register(GetCanvas(), val);
   }

   template <class PRIMITIVE>
   void Update(TOptsAttrRef<PRIMITIVE> idx, const PRIMITIVE &val)
   {
      GetAttrsRefArr((PRIMITIVE*)nullptr).Update(GetCanvas(), idx, val);
   }

   template <class PRIMITIVE>
   TOptsAttrRef<PRIMITIVE> SameAs(TOptsAttrRef<PRIMITIVE> idx)
   {
      return GetAttrsRefArr((PRIMITIVE*)nullptr).SameAs(GetCanvas(), idx);
   }

   template <class PRIMITIVE>
   TOptsAttrRef<PRIMITIVE> SameAs(const PRIMITIVE &val)
   {
      return GetAttrsRefArr((PRIMITIVE*)nullptr).SameAs(GetCanvas(), val);
   }

   template <class PRIMITIVE> friend class TOptsAttrRef;

public:
   ~TDrawingOptsBaseNoDefault();
   TDrawingOptsBaseNoDefault(const TDrawingOptsBaseNoDefault &other);
   TDrawingOptsBaseNoDefault(TDrawingOptsBaseNoDefault &&other) = default;
};

extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<TColor>;
extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<long long>;
extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<double>;

template <class DERIVED>
class TDrawingOptsBase: public TDrawingOptsBaseNoDefault {
public:
   /// Construct from the pad that holds our `TDrawable`.
   TDrawingOptsBase(TPadBase &pad): TDrawingOptsBaseNoDefault(pad)
   {
      if (&pad != &GetDefaultCanvas())
         Default();
   }
      
   /// Retrieve the default drawing options for `DERIVED`. Can be used to query and adjust the
   /// default options.
   static DERIVED &Default()
   {
      static DERIVED defaultOpts(GetDefaultCanvas());
      return defaultOpts;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
