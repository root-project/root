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
#include <ROOT/RCanvas.hxx>

#include <cassert>
#include <limits>

using namespace ROOT::Experimental;

/////////////////////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RPad::~RPad() = default;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Create pad display item

std::unique_ptr<RDisplayItem> RPad::Display(const RPadBase &, Version_t vers) const
{
   auto paditem = std::make_unique<RPadDisplayItem>();

   DisplayPrimitives(*paditem.get(), vers);

   paditem->SetPadPosSize(&fPos, &fSize);

   return paditem;
}
