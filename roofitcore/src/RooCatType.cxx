/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [CAT] --
// RooCatType is an auxilary class for RooAbsCategory and defines a 
// a single category state. The class holds a string label and an integer 
// index value which define the state

#include "RooFit.h"

#include <stdlib.h>
#include <stdlib.h>
#include "RooCatType.h"

#include "Riostream.h"


ClassImp(RooCatType)
;


void RooCatType::SetName(const Text_t* name) 
{ 
  if (strlen(name)>255) {
    std::cout << "RooCatType::SetName warning: label '" << name << "' truncated at 255 chars" << std::endl ;
    _label[255]=0 ;
  }
  strncpy(_label,name,255) ;
}

void RooCatType::printToStream(ostream& os, PrintOption /*opt*/, TString /*indent*/) const {
  // Print info about this category type to the specified stream. We only have a
  // OneLine output format.

  // we don't use oneLinePrint() since GetTitle() is empty.
  os << ClassName() << "::" << GetName() << ": Value = " << getVal() << endl;
}

