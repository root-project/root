/// \file RTreeStorage.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RTreeStorage.hxx"

#include <ROOT/RStringView.hxx>


ROOT::Experimental::Detail::RTreeSource::RTreeSource(std::string_view /*treeName*/)
{
}

ROOT::Experimental::Detail::RTreeSource::~RTreeSource()
{
}

ROOT::Experimental::Detail::RTreeSink::RTreeSink(std::string_view /*treeName*/)
{
}

ROOT::Experimental::Detail::RTreeSink::~RTreeSink()
{
}
