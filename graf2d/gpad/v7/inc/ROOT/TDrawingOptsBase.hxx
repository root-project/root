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

/** class ROOT::Experimental::TOptsAttrIdx
 The `TCanvas` keep track of `TColor`s, integer and floating point attributes used by the drawing options,
 making them accessible from other drawing options. The index into the table of the active attributes is
 wrapped into `TOptsAttrIdx` to make them type-safe (i.e. distinct for `TColor`, `long long` and `double`).
 */

template <class PRIMITIVE>
struct TOptsAttrIdx {
   size_t fIdx = (size_t)-1; ///<! The index in the relevant attribute table of `TCanvas`.
   operator size_t() const { return fIdx; }
};

namespace Internal {
/// Implementation detail - reads the config values from (system).rootstylerc.
std::map<std::string, std::string> ReadDrawingOptsDefaultConfig(std::string_view section);
class TDrawingOptsBase;

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
   bool IsFree() const { fUseCount == 0; }
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
   TOptsAttrIdx<PRIMITIVE> Register(const PRIMITIVE &val);

   /// Add a use of the attribute at table index idx.
   void IncrUse(TOptsAttrIdx<PRIMITIVE> idx) { fTable[idx].IncrUse(); }

   /// Remove a use of the attribute at table index idx.
   void DecrUse(TOptsAttrIdx<PRIMITIVE> idx) { fTable[idx].DeclUse(); }

   /// Update an existing attribute entry in the table.
   void Update(TOptsAttrIdx<PRIMITIVE> idx, const PRIMITIVE &val) { fTable[idx] = val; }

   /// Get the value at index `idx` (const version).
   const value_type &Get(TOptsAttrIdx<PRIMITIVE> idx) const { return fTable[idx]; }

   /// Get the value at index `idx` (non-const version).
   value_type &Get(TOptsAttrIdx<PRIMITIVE> idx) { return fTable[idx]; }
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

  Derived classes must implement `InitializeDefaultFromFile()`.
  */

class TDrawingOptsBase {
private:
   /// `TCanvas` that holds our `TDrawable` (or its TPad).
   TCanvas &fCanvas;

   /// Indexes of the `TCanvas`'s color table entries used by this options object.
   std::vector<TOptsAttrIdx<TColor>> fColorIdx;

   /// Indexes of the `TCanvas`'s integer table entries used by this options object.
   std::vector<TOptsAttrIdx<long long>> fIntIdx;

   /// Indexes of the `TCanvas`'s floating point table entries used by this options object.
   std::vector<TOptsAttrIdx<double>> fFPIdx;

   /// Read the configuration for section from the config file. Used by derived classes to
   /// initialize their default, in `InitializeDefaultFromFile()`.
   static std::map<std::string, std::string> ReadConfig(std::string_view section)
   {
      return Internal::ReadDrawingOptsDefaultConfig(section);
   }

   template <class PRIMITIVE>
   std::vector<TOptsAttrIdx<PRIMITIVE>> &GetIndexVec();

   template <class PRIMITIVE>
   const std::vector<TOptsAttrIdx<PRIMITIVE>> &GetIndexVec() const;

protected:
   /// Default attributes need to register their values in a pad - they will take this pad!
   static TCanvas &GetDefaultCanvas();

   /// Register an attribute.
   ///\returns the index of the new attribute.
   template <class PRIMITIVE>
   TOptsAttrIdx<PRIMITIVE> Register(const PRIMITIVE &col);

   /// Update the attribute at index `idx` to the value `val`.
   template <class PRIMITIVE>
   void Update(TOptsAttrIdx<PRIMITIVE> idx, const PRIMITIVE &val);

public:
   ~TDrawingOptsBase();

   template <class PRIMITIVE>
   struct AttrIdxAndDefault {
      TOptsAttrIdx<PRIMITIVE> &fIdxMemRef;
      TOptsAttrIdx<PRIMITIVE> fDefaultIdx;
      void Init(TDrawingOptsBase &opts);
   };

   class Attrs {
      std::vector<AttrIdxAndDefault<TColor>> fCols;
      std::vector<AttrIdxAndDefault<long long>> fInts;
      std::vector<AttrIdxAndDefault<double>> fFPs;

      Attrs& Add(const AttrIdxAndDefault<TColor>& c) { fCols.push_back(c); return *this; }
      Attrs& Add(const AttrIdxAndDefault<long long>& c) { fInts.push_back(c); return *this; }
      Attrs& Add(const AttrIdxAndDefault<double>& c) { fFPs.push_back(c); return *this; }
   };

   /// Construct from the pad that holds our `TDrawable` and the index reference data mambers,
   /// registering our colors, integer attributes and floating point attributes. The pair consists of
   /// an integer reference to our member (e.g. `fLineColorIndex`) and the default value as stored in the
   /// canvas `GetDefaultCanvas()`.
   TDrawingOptsBase(TPadBase &pad, const Attrs& attrs);

   template <class PRIMITIVE> friend class AttrIdxAndDefault;
};
extern template std::vector<TOptsAttrIdx<TColor>> &TDrawingOptsBase::GetIndexVec<TColor>();
extern template std::vector<TOptsAttrIdx<long long>> &TDrawingOptsBase::GetIndexVec<long long>();
extern template std::vector<TOptsAttrIdx<double>> &TDrawingOptsBase::GetIndexVec<double>();
extern template const std::vector<TOptsAttrIdx<TColor>> &TDrawingOptsBase::GetIndexVec<TColor>() const;
extern template const std::vector<TOptsAttrIdx<long long>> &TDrawingOptsBase::GetIndexVec<long long>() const;
extern template const std::vector<TOptsAttrIdx<double>> &TDrawingOptsBase::GetIndexVec<double>() const;
extern template class TDrawingOptsBase::AttrIdxAndDefault<TColor>;
extern template class TDrawingOptsBase::AttrIdxAndDefault<long long>;
extern template class TDrawingOptsBase::AttrIdxAndDefault<double>;

/** \class ROOT::Experimental::TDrawingOptsBaseT
 Templated layer on top of TDrawingOptBase, providing a `Default()` initialized from a config file.
 */

template <class DERIVED>
class TDrawingOptsBaseT: public TDrawingOptsBase {
protected:
   const DERIVED &ToDerived() const { return static_cast<DERIVED &>(*this); }
   DERIVED &ToDerived() { return static_cast<DERIVED &>(*this); }

   /// Default implementation: no configuration variables in config file, simply default-initialize
   /// the drawing options.
   static DERIVED InitializeDefaultFromFile() { return DERIVED(GetDefaultCanvas()); }

public:
   /// Retrieve the default drawing options for `DERIVED`. Can be used to query and adjust the
   /// default options.
   static DERIVED &Default()
   {
      static DERIVED defaultOpts = DERIVED::InitializeDefaultFromFile();
      return defaultOpts;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
