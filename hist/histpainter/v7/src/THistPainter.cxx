/// \file THistPainter.cxx
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

//#include "ROOT/THistPainter.hxx" see ROOT/THistDrawable.h

#include "ROOT/THistDrawable.hxx"
#include "ROOT/TVirtualCanvasPainter.hxx"
#include "ROOT/TDisplayItem.hxx"
#include "TH1.h"

#include <iostream>
#include <cassert>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

namespace {
class THistPainter1D: public THistPainterBase<1> {
public:
   void Paint(TDrawable &drw, const THistDrawingOpts<1> & /*opts*/, TVirtualCanvasPainter &canv) final
   {
      // TODO: paint!
      std::cout << "Painting 1D histogram @" << &drw << '\n';

      assert(dynamic_cast<THistDrawable<1> *>(&drw) && "Wrong drawable type");
      THistDrawable<1> &hd = static_cast<THistDrawable<1> &>(drw);

      ROOT::Experimental::TDisplayItem *res = new ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::THistDrawable<1>>(&hd);
      // res->SetOption("col");

      canv.AddDisplayItem(res);
   }
   virtual ~THistPainter1D() final {}
};

class THistPainter2D: public THistPainterBase<2> {
public:
   void Paint(TDrawable &drw, const THistDrawingOpts<2> & /*opts*/, TVirtualCanvasPainter &canv) final
   {
      std::cout << "Painting 2D histogram @" << &drw << '\n';
      assert(dynamic_cast<THistDrawable<2> *>(&drw) && "Wrong drawable type");
      THistDrawable<2> &hd = static_cast<THistDrawable<2> &>(drw);

      ROOT::Experimental::TDisplayItem *res = new ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::THistDrawable<2>>(&hd);
      // res->SetOption("col");

      canv.AddDisplayItem(res);

      // hd.GetOldHist()->Paint("BOX");
   }
   virtual ~THistPainter2D() final {}
};

class THistPainter3D: public THistPainterBase<3> {
public:
   void Paint(TDrawable &hist, const THistDrawingOpts<3> & /*opts*/, TVirtualCanvasPainter & /*canv*/) final
   {
      // TODO: paint!
      std::cout << "Painting 3D histogram (to be done) @" << &hist << '\n';
   }
   virtual ~THistPainter3D() final {}
};

struct HistPainterReg {
   THistPainter1D fPainter1D;
   THistPainter2D fPainter2D;
   THistPainter3D fPainter3D;
} histPainterReg;
} // unnamed namespace
