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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooCatType is an auxilary class for RooAbsCategory and defines a 
// a single category state. The class holds a string label and an integer 
// index value which define the state
// END_HTML
//

#include "RooFit.h"

#include <stdlib.h>
#include <stdlib.h>
#include "TClass.h"
#include "RooCatType.h"

#include "Riostream.h"


ClassImp(RooCatType)
;



//_____________________________________________________________________________
void RooCatType::SetName(const Text_t* name) 
{ 
  // Constructor with name argument
  if (strlen(name)>255) {
    std::cout << "RooCatType::SetName warning: label '" << name << "' truncated at 255 chars" << std::endl ;
    _label[255]=0 ;
  }
  strncpy(_label,name,255) ;
}



//_____________________________________________________________________________
void RooCatType::printName(ostream& os) const 
{
  // Print the name of the state
  os << GetName() ;
}



//_____________________________________________________________________________
void RooCatType::printTitle(ostream& os) const 
{
  // Print the title of the state
  os << GetTitle() ;
}



//_____________________________________________________________________________
void RooCatType::printClassName(ostream& os) const 
{
  // Print the class name of the state
  os << IsA()->GetName() ;
}



//_____________________________________________________________________________
void RooCatType::printValue(ostream& os) const
{
  // Print the value (index integer) of the state
  os << getVal() ;
}

