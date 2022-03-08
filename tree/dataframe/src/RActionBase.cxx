// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RActionBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx"

using namespace ROOT::Internal::RDF;

RActionBase::RActionBase(RLoopManager *lm, const ColumnNames_t &colNames, const RColumnRegister &colRegister,
                         const std::vector<std::string> &prevVariations)
   : fLoopManager(lm), fNSlots(lm->GetNSlots()), fColumnNames(colNames),
     fVariations(Union(prevVariations, colRegister.GetVariationDeps(fColumnNames))), fColRegister(colRegister)
{
}

// outlined to pin virtual table
RActionBase::~RActionBase()
{
   // The RLoopManager is kept alive via fColRegister.
   fLoopManager->Deregister(this);
}
