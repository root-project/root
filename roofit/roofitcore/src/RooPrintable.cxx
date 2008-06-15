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
// RooPlotable is a 'mix-in' base class that define the standard RooFit plotting and
// printing methods. Each RooPlotable implementation must define methods that
// print the objects name, class name, title, value, arguments and extras
// to a provided stream. The definition of value is class dependent. The definition
// of arguments is also class dependent, but should always be interpreted as
// the names (and properties) of any (RooAbsArg) external inputs of a given object.
// The extras method can be used to print any properties that does not fit in any
// of the other classes. Each object an also override the definitions made
// in defaultPrintStyle and defaultPrintContents to determine what is printed
// (in terms of contents) and how it is printed (inline,single-line or multiline)
// given a Print() option string. 
// END_HTML
//

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


//_____________________________________________________________________________
void RooPrintable::nameFieldLength(Int_t newLen)
{
  _nameLength = newLen>0 ? newLen : 0 ;
}



//_____________________________________________________________________________
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

//_____________________________________________________________________________
void RooPrintable::printValue(ostream& /*os*/) const
{
}


//_____________________________________________________________________________
void RooPrintable::printExtras(ostream& /*os*/) const
{
}


//_____________________________________________________________________________
void RooPrintable::printMultiline(ostream& /*os*/, Int_t /*contents*/, Bool_t /*verbose*/, TString /*indent*/) const
{
}


//_____________________________________________________________________________
void RooPrintable::printTree(ostream& /*os*/, TString /*indent*/) const
{
}


//_____________________________________________________________________________
void RooPrintable::printArgs(ostream& /*os*/) const 
{
}


//_____________________________________________________________________________
void RooPrintable::printName(ostream& /*os*/) const 
{
}


//_____________________________________________________________________________
void RooPrintable::printTitle(ostream& /*os*/) const 
{
}


//_____________________________________________________________________________
void RooPrintable::printClassName(ostream& /*os*/) const 
{
}



//_____________________________________________________________________________
Int_t RooPrintable::defaultPrintContents(Option_t* /*opt*/) const
{ 
  return kName|kValue ; 
}


//_____________________________________________________________________________
RooPrintable::StyleOption RooPrintable::defaultPrintStyle(Option_t* /*opt*/) const
{ 
  return kSingleLine ; 
}



//_____________________________________________________________________________
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
