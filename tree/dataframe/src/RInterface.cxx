// Author: Vincenzo Eduardo Padulano, Axel Naumann, Enrico Guiraud CERN 02/2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RInterface.hxx"

void ROOT::Internal::RDF::ChangeEmptyEntryRange(const ROOT::RDF::RNode &node,
                                                std::pair<ULong64_t, ULong64_t> &&newRange)
{
   R__ASSERT(newRange.second >= newRange.first && "end is less than begin in the passed entry range!");
   node.GetLoopManager()->SetEmptyEntryRange(std::move(newRange));
}
