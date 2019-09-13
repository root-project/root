/// \file RHistPainter.cxx
/// \ingroup HistPainter ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#include "ROOT/RHistPainter.hxx" see ROOT/RHistDrawable.h

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RPadPainter.hxx"
#include "ROOT/RDisplayItem.hxx"

#include <iostream>
#include <cassert>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

namespace {
class RHistPainter1D : public RHistPainterBase<1> {
public:
   void Paint(RHistDrawable<1> &drw, RPadPainter &pad) final
   {
      pad.AddDisplayItem(std::make_unique<RDrawableDisplayItem>(drw));
   }
   virtual ~RHistPainter1D() final {}
};

class RHistPainter2D : public RHistPainterBase<2> {
public:
   void Paint(RHistDrawable<2> &drw, RPadPainter &pad) final
   {
      pad.AddDisplayItem(std::make_unique<RDrawableDisplayItem>(drw));
   }
   virtual ~RHistPainter2D() final {}
};

class RHistPainter3D : public RHistPainterBase<3> {
public:
   void Paint(RHistDrawable<3> &drw, RPadPainter &pad) final
   {
      pad.AddDisplayItem(std::make_unique<RDrawableDisplayItem>(drw));
   }
   virtual ~RHistPainter3D() final {}
};

struct HistPainterReg {
   RHistPainter1D fPainter1D;
   RHistPainter2D fPainter2D;
   RHistPainter3D fPainter3D;
} histPainterReg;
} // unnamed namespace
