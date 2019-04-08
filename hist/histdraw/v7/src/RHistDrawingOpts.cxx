/// \file RHistDrawingOpts.cxx
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawingOpts.hxx"

#include "ROOT/TLogger.hxx"

#include <algorithm>

std::pair<ROOT::Experimental::RHistDrawingOpts<1>::EStyle, bool>
ROOT::Experimental::RHistDrawingOpts<1>::GetStyle() const {
   auto ret = Get(0);
   const auto &styles = Styles();
   auto it = std::find(styles.begin(), styles.end(), ret.first);
   if (it == styles.end()) {
      R__ERROR_HERE("HistDraw") << "Unknown 1D style " << ret.first;
      return {EStyle::kHist, ret.second};
   }
   return {static_cast<EStyle>(it - styles.begin()), ret.second};
}

std::pair<ROOT::Experimental::RHistDrawingOpts<2>::EStyle, bool>
ROOT::Experimental::RHistDrawingOpts<2>::GetStyle() const {
   auto ret = Get(0);
   const auto &styles = Styles();
   auto it = std::find(styles.begin(), styles.end(), ret.first);
   if (it == styles.end()) {
      R__ERROR_HERE("HistDraw") << "Unknown 2D style " << ret.first;
      return {EStyle::kBox, ret.second};
   }
   return {static_cast<EStyle>(it - styles.begin()), ret.second};
}

std::pair<ROOT::Experimental::RHistDrawingOpts<3>::EStyle, bool>
ROOT::Experimental::RHistDrawingOpts<3>::GetStyle() const {
   auto ret = Get(0);
   const auto &styles = Styles();
   auto it = std::find(styles.begin(), styles.end(), ret.first);
   if (it == styles.end()) {
      R__ERROR_HERE("HistDraw") << "Unknown 3D style " << ret.first;
      return {EStyle::kIso, ret.second};
   }
   return {static_cast<EStyle>(it - styles.begin()), ret.second};
}
