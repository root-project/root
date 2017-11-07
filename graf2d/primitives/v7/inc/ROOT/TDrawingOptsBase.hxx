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
#include <ROOT/TStyle.hxx>

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
     TDrawingAttrRef<TColor> fLineColor{*this, "MyFancyBoxLineColor", kRed};
  The attribute's initial value will be taken from the current style or, if that has no setting for
  the attribute, from the argument passed to the constructor (`kRed` in the example above).
  */

class TDrawingOptsBaseNoDefault {
public:
   template <class PRIMITIVE>
   class OptsAttrRefArr {
      /// Indexes of the `TCanvas`'s attribute table entries used by the options object.
      std::vector<TDrawingAttrRef<PRIMITIVE>> fRefArray;

   public:
      ~OptsAttrRefArr();
      /// Register an attribute.
      ///\returns the index of the new attribute.
      TDrawingAttrRef<PRIMITIVE> Register(TCanvas &canv, const PRIMITIVE &val);

      /// Re-use an existing attribute.
      ///\returns the index of the attribute (i.e. valRef).
      TDrawingAttrRef<PRIMITIVE> SameAs(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx);

      /// Re-use an existing attribute.
      ///\returns the index of the attribute, might be `IsInvalid()` if `val` could not be found.
      TDrawingAttrRef<PRIMITIVE> SameAs(TCanvas &canv, const PRIMITIVE &val);

      /// Update the attribute at index `idx` to the value `val`.
      void Update(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx, const PRIMITIVE &val);

      /// Clear all attribute references, removing their uses in `TCanvas`.
      void Release(TCanvas &canv);

      /// Once copied, elements of a OptsAttrRefArr need to increase their use count.
      void RegisterCopy(TCanvas &canv);

      /// Access to attribute (const version).
      const PRIMITIVE &Get(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx) const;

      /// Access to attribute (non-const version).
      PRIMITIVE &Get(TCanvas &canv, TDrawingAttrRef<PRIMITIVE> idx);
   };

private:
   /// The `TCanvas` holding the `TDrawable` (or its `TPad`).
   TCanvas *fCanvas = nullptr;

   /// Name of these drawing options, e.g. "1D"; will cause a member `TLineAttr{*this, "Line"}` to
   /// look for a style setting called "1D.Line.Color".
   std::string fName;

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
   TDrawingOptsBaseNoDefault(TPadBase &pad, std::string_view name);

   /// Default attributes need to register their values in a pad - they will take this pad
   /// for default attributes of a style, as identified by the style's name.
   static TPadBase &GetDefaultCanvas(const TStyle &style);

   /// Whether the canvas is one of the canvases used to store attribute defaults.
   static bool IsDefaultCanvas(const TPadBase &canv);

   /// Get the (style config) name of this option set.
   const std::string &GetName() const { return fName; }

   /// The `TCanvas` holding the `TDrawable` (or its `TPad`) (non-const version).
   TCanvas &GetCanvas() { return *fCanvas; }

   /// The `TCanvas` holding the `TDrawable` (or its `TPad`) (const version).
   const TCanvas &GetCanvas() const { return *fCanvas; }

   template <class PRIMITIVE>
   TDrawingAttrRef<PRIMITIVE> Register(const PRIMITIVE &val)
   {
      return GetAttrsRefArr((PRIMITIVE *)nullptr).Register(GetCanvas(), val);
   }

   template <class PRIMITIVE>
   void Update(TDrawingAttrRef<PRIMITIVE> idx, const PRIMITIVE &val)
   {
      GetAttrsRefArr((PRIMITIVE *)nullptr).Update(GetCanvas(), idx, val);
   }

   template <class PRIMITIVE>
   TDrawingAttrRef<PRIMITIVE> SameAs(TDrawingAttrRef<PRIMITIVE> idx)
   {
      return GetAttrsRefArr((PRIMITIVE *)nullptr).SameAs(GetCanvas(), idx);
   }

   template <class PRIMITIVE>
   TDrawingAttrRef<PRIMITIVE> SameAs(const PRIMITIVE &val)
   {
      return GetAttrsRefArr((PRIMITIVE *)nullptr).SameAs(GetCanvas(), val);
   }

   template <class PRIMITIVE>
   friend class TDrawingAttrRef;

public:
   TDrawingOptsBaseNoDefault() = default;
   ~TDrawingOptsBaseNoDefault();
   TDrawingOptsBaseNoDefault(const TDrawingOptsBaseNoDefault &other);
   TDrawingOptsBaseNoDefault(TDrawingOptsBaseNoDefault &&other) = default;

   /// Access to the attribute (non-const version).
   template <class PRIMITIVE>
   PRIMITIVE &Get(TDrawingAttrRef<PRIMITIVE> ref) { return GetAttrsRefArr((PRIMITIVE*)nullptr).Get(GetCanvas(), ref); }
   /// Access to the attribute (const version).
   template <class PRIMITIVE>
   const PRIMITIVE &Get(TDrawingAttrRef<PRIMITIVE> ref) const { return GetAttrsRefArr((PRIMITIVE*)nullptr).Get(GetCanvas(), ref); }
};

extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<TColor>;
extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<long long>;
extern template class TDrawingOptsBaseNoDefault::OptsAttrRefArr<double>;

template <class DERIVED>
class TDrawingOptsBase: public TDrawingOptsBaseNoDefault {
public:
   TDrawingOptsBase() = default;
   /// Construct from the pad that holds our `TDrawable`.
   TDrawingOptsBase(TPadBase &pad, std::string_view name):
   TDrawingOptsBaseNoDefault(pad, name)
   {
      if (!IsDefaultCanvas(pad))
         Default();
   }

   /// Apply the given options to this option set.
   void Apply(const DERIVED& other) {
      static_cast<DERIVED&>(*this) = other;
   }

   /// Retrieve the default drawing options for the given style.
   static DERIVED GetDefaultForStyle(const TStyle& style)
   {
      return DERIVED(GetDefaultCanvas(style));
   }

   /// Retrieve the default drawing options for `DERIVED`. Can be used to query and adjust the
   /// default options.
   static DERIVED &Default()
   {
      static DERIVED defaultOpts = GetDefaultForStyle(TStyle::GetCurrent());
      return defaultOpts;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
