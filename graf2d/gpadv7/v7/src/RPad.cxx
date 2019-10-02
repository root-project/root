/// \file RPad.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RPad.hxx"

#include "ROOT/RLogger.hxx"
#include <ROOT/RPadDisplayItem.hxx>
#include <ROOT/RPadPainter.hxx>
#include <ROOT/RCanvas.hxx>

#include <cassert>
#include <limits>

/////////////////////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RPad::~RPad() = default;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Paint the pad.
void ROOT::Experimental::RPad::Paint(Internal::RPadPainter &toppad)
{
   Internal::RPadPainter painter;

   painter.PaintDrawables(*this);

   painter.fPadDisplayItem->SetPadPosSize(&fPos, &fSize);

   painter.fPadDisplayItem->SetAttributes(&GetAttrMap());

   toppad.AddDisplayItem(std::move(painter.fPadDisplayItem));
}
