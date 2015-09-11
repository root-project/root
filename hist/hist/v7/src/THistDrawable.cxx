/// \file ROOT/THistDrawable.cxx
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/THistDrawable.h"

using namespace ROOT;
using namespace ROOT::Internal;

template <int DIMENSION>
THistPainterBase<DIMENSION>::~THistPainterBase() { fgPainter = nullptr; }

template <int DIMENSION>
THistPainterBase<DIMENSION>* THistPainterBase<DIMENSION>::fgPainter = nullptr;

template class THistPainterBase<1>;
template class THistPainterBase<2>;
template class THistPainterBase<3>;

