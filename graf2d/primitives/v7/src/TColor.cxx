/// \file TColor.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-27
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TColor.hxx"

#include <exception>

using namespace ROOT::Experimental;

// TColor constexpr values:
constexpr TColor::Alpha TColor::kOpaque;
constexpr TColor::Alpha TColor::kTransparent;
constexpr TColor::RGBA TColor::kRed;
constexpr TColor::RGBA TColor::kGreen;
constexpr TColor::RGBA TColor::kBlue;
constexpr TColor::RGBA TColor::kWhite;
constexpr TColor::RGBA TColor::kBlack;
constexpr TColor::RGBA TColor::kInvisible;
constexpr TColor::AutoTag TColor::kAuto;


float TColor::GetPaletteOrdinal() const
{
   if (fKind != EKind::kPalettePos)
      throw std::runtime_error("This color does not represent a palette ordinal!");
   return fRedOrPalettePos;
}

bool TColor::AssertNotPalettePos() const
{
   if (fKind == EKind::kPalettePos) {
      throw std::runtime_error("This color does not represent a palette ordinal!");
      return false;
   }
   return true;
}
