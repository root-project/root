/// \file RPageStorage.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RPagePool.hxx>

#include <ROOT/RStringView.hxx>

namespace ROOT {
namespace Experimental {
namespace Detail {

RPageStorage::RPageStorage(std::string_view name) : fNTupleName(name)
{
}

RPageStorage::~RPageStorage()
{
}

RPageSource::RPageSource(std::string_view name) : RPageStorage(name)
{
}

RPageSource::~RPageSource()
{
}

RPageSink::RPageSink(std::string_view name) : RPageStorage(name)
{
}

RPageSink::~RPageSink()
{
}

} // namespace Detail
} // namespace Experimental
} // namespace ROOT
