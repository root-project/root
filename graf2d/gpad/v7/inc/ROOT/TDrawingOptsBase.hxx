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
#include <ROOT/TDrawingAttrs.hxx>

#include <RStringView.h>

#include <map>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {
class TCanvas;
class TPadBase;

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

      /// Access to attribute (const version).
      const PRIMITIVE &Get(TCanvas &canv, TOptsAttrRef<PRIMITIVE> idx) const;

      /// Access to attribute (non-const version).
      PRIMITIVE &Get(TCanvas &canv, TOptsAttrRef<PRIMITIVE> idx);
   };

private:
   /// The `TCanvas` holding the `TDrawable` (or its `TPad`).
   TCanvas *fCanvas = nullptr;

   /// String vector specifying the prefixes used to select the right setting from the style file,
   /// e.g. `{"1D", "Hist", "Line"}`.
   /// \note Requires external string storage; usually, this parameter is passed as
   /// ```
   ///     TXYZDrawingOpts fOpts{*this, "Hist.Foo", {12, red}};
   /// ```
   std::vector<std::string_view> fConfigPrefix;

   /// Indexes of the `TCanvas`'s color table entries used by this options object.
   OptsAttrRefArr<TColor> fColorIdx;

   /// Indexes of the `TCanvas`'s integer table entries used by this options object.
   OptsAttrRefArr<long long> fIntIdx;

   /// Indexes of the `TCanvas`'s floating point table entries used by this options object.
   OptsAttrRefArr<double> fFPIdx;

   /// Access to the attribute tables (non-const version).
   OptsAttrRefArr<TColor> &GetAttrsRefArr(TColor *) { return fColorIdx; }
   OptsAttrRefArr<long long> &GetAttrsRefArr(long long *) { return fIntIdx; }
   OptsAttrRefArr<double> &GetAttrsRefArr(double *) { return fFPIdx; }

   /// Access to the attribute tables (const version).
   const OptsAttrRefArr<TColor> &GetAttrsRefArr(TColor *) const { return fColorIdx; }
   const OptsAttrRefArr<long long> &GetAttrsRefArr(long long *) const { return fIntIdx; }
   const OptsAttrRefArr<double> &GetAttrsRefArr(double *) const { return fFPIdx; }

protected:
   /// Construct from the pad that holds our `TDrawable`.
   TDrawingOptsBaseNoDefault(TPadBase &pad, const std::vector<string_view> &configPrefix);

   /// Default attributes need to register their values in a pad - they will take this pad!
   static TPadBase &GetDefaultCanvas();

   /// The `TCanvas` holding the `TDrawable` (or its `TPad`) (non-const version).
   TCanvas &GetCanvas() { return *fCanvas; }

   /// The `TCanvas` holding the `TDrawable` (or its `TPad`) (const version).
   const TCanvas &GetCanvas() const { return *fCanvas; }

   template <class PRIMITIVE>
   TOptsAttrRef<PRIMITIVE> Register(const PRIMITIVE &val)
   {
      return GetAttrsRefArr((PRIMITIVE *)nullptr).Register(GetCanvas(), val);
   }

   template <class PRIMITIVE>
   void Update(TOptsAttrRef<PRIMITIVE> idx, const PRIMITIVE &val)
   {
      GetAttrsRefArr((PRIMITIVE *)nullptr).Update(GetCanvas(), idx, val);
   }

   template <class PRIMITIVE>
   TOptsAttrRef<PRIMITIVE> SameAs(TOptsAttrRef<PRIMITIVE> idx)
   {
      return GetAttrsRefArr((PRIMITIVE *)nullptr).SameAs(GetCanvas(), idx);
   }

   template <class PRIMITIVE>
   TOptsAttrRef<PRIMITIVE> SameAs(const PRIMITIVE &val)
   {
      return GetAttrsRefArr((PRIMITIVE *)nullptr).SameAs(GetCanvas(), val);
   }

   template <class PRIMITIVE>
   friend class TOptsAttrRef;

public:
   TDrawingOptsBaseNoDefault() = default;
   ~TDrawingOptsBaseNoDefault();
   TDrawingOptsBaseNoDefault(const TDrawingOptsBaseNoDefault &other);
   TDrawingOptsBaseNoDefault(TDrawingOptsBaseNoDefault &&other) = default;

   std::string GetConfigPrefix() const {
      std::string ret;
      for (auto el: fConfigPrefix) {
         ret += el;
         ret += '.';
      }
      ret.erase(ret.end() - 1);
   }

   /// Access to the attribute (non-const version).
   template <class PRIMITIVE>
   PRIMITIVE &Get(TOptsAttrRef<PRIMITIVE> ref) { return GetAttrsRefArr((PRIMITIVE*)nullptr).Get(GetCanvas(), ref); }
   /// Access to the attribute (const version).
   template <class PRIMITIVE>
   const PRIMITIVE &Get(TOptsAttrRef<PRIMITIVE> ref) const { return GetAttrsRefArr((PRIMITIVE*)nullptr).Get(GetCanvas(), ref); }
};

extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<TColor>;
extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<long long>;
extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<double>;

template <class DERIVED>
class TDrawingOptsBase: public TDrawingOptsBaseNoDefault {
public:
   TDrawingOptsBase() = default;
   /// Construct from the pad that holds our `TDrawable`.
   TDrawingOptsBase(TPadBase &pad, const std::vector<string_view> &configPrefix):
   TDrawingOptsBaseNoDefault(pad, []&configPrefix(){configPrefix.push_back("Line"); return configPrefix;})
   {
      if (&pad != &GetDefaultCanvas())
         Default(configPrefix);
   }

   /// Retrieve the default drawing options for `DERIVED`. Can be used to query and adjust the
   /// default options.
   static DERIVED &Default(string_view configPrefix = {})
   {
      static DERIVED defaultOpts(GetDefaultCanvas(), configPrefix);
      return defaultOpts;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
