/// \file RPadPos.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RPadPos.hxx"

#include <ROOT/RPadExtent.hxx>
#include <ROOT/TLogger.hxx>

////////////////////////////////////////////////////////////////////////////////
/// Initialize a RPadPos from a style string.
/// Syntax: X, Y
/// where X and Y are a series of numbers separated by "+", where each number is
/// followed by one of `px`, `user`, `normal` to specify an extent in pixel,
/// user or normal coordinates. Spaces between any part is allowed.
/// Example: `100 px + 0.1 user, 0.5 normal` is a `RPadPos{100_px + 0.1_user, 0.5_normal}`.

ROOT::Experimental::RPadPos ROOT::Experimental::PosFromString(const std::string &name, const std::string &attrStrVal)
{
   RPadPos ret;
   ret.SetFromAttrString(name, attrStrVal);
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a RPadPos to a style string, matching what PosFromString can parse.

std::string ROOT::Experimental::PosToString(const RPadPos &pos)
{
   std::string ret = PadLengthToString(pos.fHoriz);
   ret += ", ";
   ret += PadLengthToString(pos.fVert);
   return ret;
}

