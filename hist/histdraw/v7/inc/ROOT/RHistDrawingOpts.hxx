/// \file ROOT/RHistDrawingOpts.h
/// \ingroup HistDraw ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHistDrawingOpts
#define ROOT7_RHistDrawingOpts

#include <ROOT/RAttrLine.hxx>
#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RDrawingOptsBase.hxx>

namespace ROOT {
namespace Experimental {

template <int DIMENSION>
class RHistDrawingOpts {
   static_assert(DIMENSION != 0, "Cannot draw 0-dimensional histograms!");
   static_assert(DIMENSION > 3, "Cannot draw histograms with more than 3 dimensions!");
   static_assert(DIMENSION < 3, "This should have been handled by the specializations below?!");
};


/** \class RHistDrawingOpts<1>
 Drawing options for a 1D histogram.
 */
template <>
class RHistDrawingOpts<1>: public RDrawingOptsBase, public RDrawingAttrBase {
public:
   enum class EStyle { kHist, kBar, kText };

private:
   static const std::array<std::string, 3> &Styles() {
      static std::array<std::string, 3> styles{"hist", "bar", "text"};
      return styles;
   }

public:
   RHistDrawingOpts():
      RDrawingAttrBase("hist1D", this, nullptr, {"style"})
   {}

   /// The drawing style.
   void SetStyle(EStyle style) { Set(0, Styles()[static_cast<std::size_t>(style)]); }
   std::pair<EStyle, bool> GetStyle() const;

   RAttrLine contentLine{"contentLine", this};
   RAttrLine barLine{"barLine", this};
   RAttrLine uncertaintyLine{"uncertaintyLine", this};
   RAttrLine borderLine{"borderLine", this};
};

/** \class RHistDrawingOpts<2>
 Drawing options for a 2D histogram.
 */
template <>
class RHistDrawingOpts<2>: public RDrawingOptsBase, public RDrawingAttrBase {
public:
   enum class EStyle { kBox, kSurf, kText };

private:
   static const std::array<std::string, 3> &Styles() {
      static std::array<std::string, 3> styles{"box", "surf", "text"};
      return styles;
   }

public:
   RHistDrawingOpts():
      RDrawingAttrBase("hist2D", this, nullptr, {"style"})
   {}

   /// The drawing style.
   void SetStyle(EStyle style) { Set(0, Styles()[static_cast<std::size_t>(style)]); }
   std::pair<EStyle, bool> GetStyle() const;

   RAttrLine boxLine{"boxLine", this};
};

/** \class RHistDrawingOpts<3>
 Drawing options for a 3D histogram.
 */
template <>
class RHistDrawingOpts<3>: public RDrawingOptsBase, public RDrawingAttrBase {
public:
   enum class EStyle { kBox, kIso };

private:
   static const std::array<std::string, 2> &Styles() {
      static std::array<std::string, 2> styles{"box", "iso"};
      return styles;
   }

public:
   RHistDrawingOpts():
      RDrawingAttrBase("hist3D", this, nullptr, {"style"})
   {}

   /// The drawing style.
   void SetStyle(EStyle style) { Set(0, Styles()[static_cast<std::size_t>(style)]); }
   std::pair<EStyle, bool> GetStyle() const;

   RAttrLine boxLine{"boxLine", this};
   RAttrLine isoLine{"isoLine", this};
};

} // namespace Experimental
} // namespace ROOT

#endif
