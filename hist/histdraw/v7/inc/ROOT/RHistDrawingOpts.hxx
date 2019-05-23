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
   /// The drawing style.
   EStyle fStyle;

   std::vector<MemberAssociation> GetMembers() final {
      return {
         Associate("style", fStyle)
      };
   };

protected:
   Name_t GetName() const final { return "hist1D"; }

public:
   // Not needed; attributes always part of options (not in holder).
   std::unique_ptr<RDrawingAttrBase> Clone() const { return {}; }

   /// The drawing style.
   void SetStyle(EStyle style) { fStyle = style; }
   EStyle GetStyle() const { return fStyle; }

   RAttrLine &Line() { return Get<RAttrLine>("contentLine"); }
   RAttrLine &BarLine() { return Get<RAttrLine>("barLine"); }
   RAttrLine &UncertaintyLine() { return Get<RAttrLine>("uncertaintyLine"); }
};

/** \class RHistDrawingOpts<2>
 Drawing options for a 2D histogram.
 */
template <>
class RHistDrawingOpts<2>: public RDrawingOptsBase, public RDrawingAttrBase {
public:
   enum class EStyle { kBox, kSurf, kText };

private:
   /// The drawing style.
   EStyle fStyle;

   std::vector<MemberAssociation> GetMembers() final {
      return {
         Associate("style", fStyle)
      };
   };

protected:
   Name_t GetName() const final { return "hist2D"; }

public:
   // Not needed; attributes always part of options (not in holder).
   std::unique_ptr<RDrawingAttrBase> Clone() const { return {}; }

   /// The drawing style.
   void SetStyle(EStyle style) { fStyle = style; }
   EStyle GetStyle() const { return fStyle; }

   RAttrLine &BoxLine() { return Get<RAttrLine>("boxLine"); }
};

/** \class RHistDrawingOpts<3>
 Drawing options for a 3D histogram.
 */
template <>
class RHistDrawingOpts<3>: public RDrawingOptsBase, public RDrawingAttrBase {
public:
   enum class EStyle { kBox, kIso };

private:
   /// The drawing style.
   EStyle fStyle;

   std::vector<MemberAssociation> GetMembers() final {
      return {
         Associate("style", fStyle)
      };
   };

protected:
   Name_t GetName() const final { return "hist3D"; }

public:
   // Not needed; attributes always part of options (not in holder).
   std::unique_ptr<RDrawingAttrBase> Clone() const { return {}; }

   /// The drawing style.
   void SetStyle(EStyle style) { fStyle = style; }
   EStyle GetStyle() const { return fStyle; }


   RAttrLine &BoxLine() { return Get<RAttrLine>("boxLine"); }
   RAttrLine &IsoLine() { return Get<RAttrLine>("isoLine"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
