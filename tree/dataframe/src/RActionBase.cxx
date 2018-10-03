// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RActionBase.hxx"
#include "ROOT/RLoopManager.hxx"

using namespace ROOT::Internal::RDF;

RActionBase::RActionBase(RLoopManager *implPtr, const unsigned int nSlots, const ColumnNames_t &colNames,
                         const RBookedCustomColumns &customColumns)
   : fLoopManager(implPtr), fNSlots(nSlots), fColumnNames(colNames), fCustomColumns(customColumns) {}

RActionBase::~RActionBase()
{
   fLoopManager->Deregister(this);
}
