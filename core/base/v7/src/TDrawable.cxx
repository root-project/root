/// \file TDrawable.cxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TDrawable.hxx"

#include <cassert>

// pin vtable
ROOT::Experimental::TDrawable::~TDrawable() {}

void ROOT::Experimental::TDrawable::Execute(const std::string &)
{
   assert(false && "Did not expect a menu item to be invoked!");
}
