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
// Class RooBinningCategory provides a real-to-category mapping defined
// by a series of thresholds.
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooBinningCategory.h"
#include "RooStreamParser.h"
#include "RooThreshEntry.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooBinningCategory)



//_____________________________________________________________________________
RooBinningCategory::RooBinningCategory(const char *name, const char *title, RooAbsRealLValue& inputVar, 
					   const char* binningName, const char* catTypeName) :
  RooAbsCategory(name, title), _inputVar("inputVar","Input category",this,inputVar), _bname(binningName)
{
  // Constructor with input function to be mapped and name and index of default
  // output state of unmapped values

  initialize(catTypeName) ;

}



//_____________________________________________________________________________
RooBinningCategory::RooBinningCategory(const RooBinningCategory& other, const char *name) :
  RooAbsCategory(other,name), _inputVar("inputVar",this,other._inputVar), _bname(other._bname)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooBinningCategory::~RooBinningCategory() 
{
  // Destructor
}




//_____________________________________________________________________________
void RooBinningCategory::initialize(const char* catTypeName)
{
  // Iterator over all bins in input variable and define corresponding state labels

  Int_t nbins = ((RooAbsRealLValue&)_inputVar.arg()).getBinning(_bname.Length()>0?_bname.Data():0).numBins() ;
  for (Int_t i=0 ; i<nbins ; i++) {
    string name = catTypeName!=0 ? Form("%s%d",catTypeName,i)
            : (_bname.Length()>0 ? Form("%s_%s_bin%d",_inputVar.arg().GetName(),_bname.Data(),i) 
            : Form("%s_bin%d",_inputVar.arg().GetName(),i)) ;
    defineType(name.c_str(),i) ;
  }
}




//_____________________________________________________________________________
RooCatType RooBinningCategory::evaluate() const
{
  // Calculate and return the value of the mapping function
  Int_t ibin = ((RooAbsRealLValue&)_inputVar.arg()).getBin(_bname.Length()>0?_bname.Data():0) ;
  const RooCatType* cat = lookupType(ibin) ;
  if (!cat) {

    string name = (_bname.Length()>0) ? Form("%s_%s_bin%d",_inputVar.arg().GetName(),_bname.Data(),ibin) 
	                              : Form("%s_bin%d",_inputVar.arg().GetName(),ibin) ;
    cat = const_cast<RooBinningCategory*>(this)->defineType(name.c_str(),ibin) ;     
  }

  return *cat ;
}




//_____________________________________________________________________________
void RooBinningCategory::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
  // Print info about this threshold category to the specified stream. In addition to the info
  // from RooAbsCategory::printStream() we add:
  //
  //  Standard : input category
  //     Shape : default value
  //   Verbose : list of thresholds

   RooAbsCategory::printMultiline(os,content,verbose,indent);

   if (verbose) {
     os << indent << "--- RooBinningCategory ---" << endl
	<< indent << "  Maps from " ;
     _inputVar.arg().printStream(os,kName|kValue,kSingleLine);
   }
}


