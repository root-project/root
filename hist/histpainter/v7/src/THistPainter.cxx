/// \file ROOT/THistPainter.cxx
/// \ingroup HistPainter
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-09

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "ROOT/THistImpl.h"
#include "ROOT/THistDrawable.h"

using namespace ROOT;
using namespace ROOT::Internal;

namespace {
class THistPainter1D: public THistPainterBase<1> {
public:
  void Paint(TDrawable& hist, THistDrawOptions<1> /*opts*/) final {
    // TODO: paint!
    std::cout << "Painting histogram @" << &hist << '\n';
  }
  virtual ~THistPainter1D() final {}
};

class THistPainter2D: public THistPainterBase<2> {
public:
  void Paint(TDrawable& hist, THistDrawOptions<2> /*opts*/) final {
    // TODO: paint!
    std::cout << "Painting histogram @" << &hist << '\n';
  }
  virtual ~THistPainter2D() final {}
};

class THistPainter3D: public THistPainterBase<3> {
public:
  void Paint(TDrawable& hist, THistDrawOptions<3> /*opts*/) final {
    // TODO: paint!
    std::cout << "Painting histogram @" << &hist << '\n';
  }
  virtual ~THistPainter3D() final {}
};


struct HistPainterReg {
  THistPainter1D fPainter1D;
  THistPainter2D fPainter2D;
  THistPainter3D fPainter3D;
} histPainterReg;
} // unnamed namespace
