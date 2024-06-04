/// \file RNTupleDSHelpers.hxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-06-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleDSHelpers
#define ROOT_RNTupleDSHelpers

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleDS.hxx>

namespace ROOT {
namespace RDF {
namespace Experimental {
////////////////////////////////////////////////////////////////////////////////
/// \brief Create an RDataFrame from an RNTuple.
///
/// \param[in] ntupleName The ntuple name
/// \param[in] fileName The file name
///
/// \return An RDataFrame based on the provided RNTuple
///
/// \note It is possible to create RNTuple-based dataframes with the standard RDataFrame constructor!
///
ROOT::RDataFrame FromRNTuple(std::string_view ntupleName, std::string_view fileName);
ROOT::RDataFrame FromRNTuple(std::string_view ntupleName, const std::vector<std::string> &fileNames);
ROOT::RDataFrame FromRNTuple(ROOT::Experimental::RNTuple *ntuple);
} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RNTupleDSHelpers
