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
#include "TClass.h"

using namespace std;

ClassImp(RooPrintable)
;

Int_t  RooPrintable::_nameLength(0) ;

namespace RooFit {
  ostream& operator<<(ostream& os, const RooPrintable& rp) { 
    // Implement ostream operator on RooPrintable in terms of printStream(InLine)  
    rp.printStream(os,rp.defaultPrintContents("I"),RooPrintable::kInline) ; return os ; 
  }
}


//_____________________________________________________________________________
void RooPrintable::nameFieldLength(Int_t newLen)
{
  // Set length of field reserved from printing name of RooAbsArgs in 
  // multi-line collection printing to given amount. 
  _nameLength = newLen>0 ? newLen : 0 ;
}



//_____________________________________________________________________________
void RooPrintable::printStream(ostream& os, Int_t contents, StyleOption style, TString indent) const 
{
  // Print description of object on ostream, printing contents set by contents integer,
  // which is interpreted as an OR of 'enum ContentsOptions' values and in the style
  // given by 'enum StyleOption'. Each message is prefixed by string 'indent' when printed

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
    printAddress(os) ;
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
  // Interface to print value of object
}


//_____________________________________________________________________________
void RooPrintable::printExtras(ostream& /*os*/) const
{
  // Interface to print extras of object
}


//_____________________________________________________________________________
void RooPrintable::printMultiline(ostream& /*os*/, Int_t /*contents*/, Bool_t /*verbose*/, TString /*indent*/) const
{
  // Interface for detailed printing of object
}


//_____________________________________________________________________________
void RooPrintable::printTree(ostream& /*os*/, TString /*indent*/) const
{
  // Interface for tree structure printing of object
  cout << "Tree structure printing not implement for class " << IsA()->GetName() << endl ;
}


//_____________________________________________________________________________
void RooPrintable::printArgs(ostream& /*os*/) const 
{
  // Interface for printing of object arguments. Arguments
  // are loosely defined as external server objects
  // in this context
}


//_____________________________________________________________________________
void RooPrintable::printName(ostream& /*os*/) const 
{
  // Print name of object
}


//_____________________________________________________________________________
void RooPrintable::printTitle(ostream& /*os*/) const 
{
  // Print title of object
}


//_____________________________________________________________________________
void RooPrintable::printClassName(ostream& /*os*/) const 
{
  // Print class name of object
}



//_____________________________________________________________________________
void RooPrintable::printAddress(ostream& os) const 
{
  // Print class name of object
  os << this ;
}



//_____________________________________________________________________________
Int_t RooPrintable::defaultPrintContents(Option_t* /*opt*/) const
{ 
  // Default choice of contents to be printed (name and value)
  return kName|kValue ; 
}


//_____________________________________________________________________________
RooPrintable::StyleOption RooPrintable::defaultPrintStyle(Option_t* opt) const
{ 
  if (!opt) {
    return kSingleLine ;
  }

  TString o(opt) ;
  o.ToLower() ;

  if (o.Contains("v")) {
    return kVerbose ;
  } else if (o.Contains("s")) {
    return kStandard ;
  } else if (o.Contains("i")) {
    return kInline ;
  } else if (o.Contains("t")) {
    return kTreeStructure ;
  }

  return kSingleLine ;
}



//_____________________________________________________________________________
ostream &RooPrintable::defaultPrintStream(ostream *os) 
{
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
