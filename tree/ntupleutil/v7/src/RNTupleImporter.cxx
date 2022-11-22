/// \file RNTupleImporter.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2022-11-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleImporter.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RStringView.hxx>

#include <memory>

std::unique_ptr<ROOT::Experimental::RNTupleImporter>
ROOT::Experimental::RNTupleImporter::Create(std::string_view /* sourceFile */, std::string_view /* treeName */,
                                            std::string_view /* destFile */)
{
   R__LOG_WARNING(NTupleLog()) << "not yet implemented";
   return nullptr;
}
