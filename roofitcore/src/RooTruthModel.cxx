/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// 

#include <iostream.h>
#include "RooFitCore/RooTruthModel.hh"

ClassImp(RooTruthModel) 
;


RooTruthModel::RooTruthModel(const char *name, const char *title) : 
  RooResolutionModel(name,title)
{  
}


RooTruthModel::RooTruthModel(const RooTruthModel& other, const char* name) : 
  RooResolutionModel(other,name)
{
}


RooTruthModel::~RooTruthModel()
{
  // Destructor
}


Bool_t RooTruthModel::isBasisSupported(const char* name) const 
{
  // Truth model is delta function, i.e. convolution integral
  // is basis function, therefore we can handle any basis function
  return kTRUE ;
}


Double_t RooTruthModel::evaluate(const RooDataSet* dset) const 
{
  return basis().getVal() ;
}
