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

// -- CLASS DESCRIPTION [MISC] --
// Classes inheriting from this class can be plotted and printed.

#include "RooFit.h"

#include "RooPrintable.h"

#include "Riostream.h"
#include <iomanip>
#include "TNamed.h"

ClassImp(RooPrintable)
;

Int_t  RooPrintable::_nameLength(0) ;

// Implement ostream operator on RooPrintable in terms of printStream(InLine)  
namespace RooFit {
  ostream& operator<<(ostream& os, const RooPrintable& rp) { 
    rp.printStream(os,rp.defaultPrintContents("I"),RooPrintable::kInline) ; return os ; 
  }
}

void RooPrintable::nameFieldLength(Int_t newLen)
{
  _nameLength = newLen>0 ? newLen : 0 ;
}


void RooPrintable::printStream(ostream& os, Int_t contents, StyleOption style, TString indent) const 
{
  // enum ContentsOption { Name=1, Value=2, ValueName=3, ValueNameExtra=4, Structure=5 } ;
  //   enum StyleOption { Inline=1, SingleLine=2, MultiLine=3 } ;

  // Handling of 'verbose' and 'treestructure' is delegated to dedicated implementation functions
  if (style==kVerbose||style==kStandard) {
    printMultiline(os,contents,style==kVerbose,indent) ;
    return ;
  } else if (style==kTreeStructure) {
    printTree(os,indent) ;
    return ;
  }

  // Handle here Inline and SingleLine styles
  if (style!=kInline) os << indent ;

  // Print class name if requested
  if (contents&kAddress) {
    os << this ; 
    if (contents!=kAddress) {
      os << " " ;
    }
  }

  // Print class name if requested
  if (contents&kClassName) {
    printClassName(os) ;
    if (contents!=kClassName) {
      os << "::" ;
    }
  }

  // Print object name if requested
  if (contents&kName) {
    if (_nameLength>0) {
      os << setw(_nameLength) ;
    }
    printName(os) ;
  }

  // Print input argument structure from proxies if requested
  if (contents&kArgs) {
    printArgs(os) ;
  }
  
  // Print value if requested
  if (contents&kValue) {
    if (contents&kName) {
      os << " = " ;
    }
    printValue(os) ;
  }

  // Print extras if required
  if (contents&kExtras) {
    if (contents!=kExtras) {
      os << " " ;
    }
    printExtras(os) ;
  }

  // Print title if required
  if (contents&kTitle) {
    if (contents==kTitle) {
      printTitle(os) ;
    } else {
      os << " \"" ;
      printTitle(os) ;
      os << "\"" ;
    }
  }

  if (style!=kInline) os << endl ;
  
}


// Virtual hook function for class-specific content implementation
void RooPrintable::printValue(ostream& /*os*/) const
{
}

void RooPrintable::printExtras(ostream& /*os*/) const
{
}

void RooPrintable::printMultiline(ostream& /*os*/, Int_t /*contents*/, Bool_t /*verbose*/, TString /*indent*/) const
{
}

void RooPrintable::printTree(ostream& /*os*/, TString /*indent*/) const
{
}

void RooPrintable::printArgs(ostream& /*os*/) const 
{
}

void RooPrintable::printName(ostream& /*os*/) const 
{
}

void RooPrintable::printTitle(ostream& /*os*/) const 
{
}

void RooPrintable::printClassName(ostream& /*os*/) const 
{
}


Int_t RooPrintable::defaultPrintContents(Option_t* /*opt*/) const
{ 
  return kName|kValue ; 
}

RooPrintable::StyleOption RooPrintable::defaultPrintStyle(Option_t* /*opt*/) const
{ 
  return kSingleLine ; 
}


ostream &RooPrintable::defaultPrintStream(ostream *os) {
  // Return a reference to the current default stream to use in
  // Print(). Use the optional parameter to specify a new default
  // stream (a reference to the old one is still returned). This
  // method allows subclasses to provide an inline implementation of
  // Print() without pulling in iostream.h.

  static ostream *_defaultPrintStream = &cout;

  ostream& _oldDefault= *_defaultPrintStream;
  if(0 != os) _defaultPrintStream= os;
  return _oldDefault;
}
