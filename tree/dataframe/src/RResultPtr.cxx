// Author: Vincenzo Eduardo Padulano, Enrico Guiraud CERN 2023/02

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string>

#include <ROOT/RResultPtr.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDF/RLoopManager.hxx>

ROOT::Internal::RDF::SnapshotPtr_t
ROOT::Internal::RDF::CloneResultAndAction(const ROOT::Internal::RDF::SnapshotPtr_t &inptr,
                                          const std::string &outputFileName)
{
   return ROOT::Internal::RDF::SnapshotPtr_t{
      inptr.fObjPtr, inptr.fLoopManager,
      inptr.fActionPtr->CloneAction(reinterpret_cast<void *>(const_cast<std::string *>(&outputFileName)))};
}
