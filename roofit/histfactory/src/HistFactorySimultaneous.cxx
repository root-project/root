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
// RooSimultaneous facilitates simultaneous fitting of multiple PDFs
// to subsets of a given dataset.
// <p>
// The class takes an index category, which is interpreted as
// the data subset indicator, and a list of PDFs, each associated
// with a state of the index category. RooSimultaneous always returns
// the value of the PDF that is associated with the current value
// of the index category
// <p>
// Extended likelihood fitting is supported if all components support
// extended likelihood mode. The expected number of events by a RooSimultaneous
// is that of the component p.d.f. selected by the index category
// END_HTML
//


#include "RooNLLVar.h"

#include "RooStats/HistFactory/RooBarlowBeestonLL.h"
#include "RooStats/HistFactory/HistFactorySimultaneous.h"

using namespace std ;

ClassImp(RooStats::HistFactory::HistFactorySimultaneous)
;


//_____________________________________________________________________________
RooStats::HistFactory::HistFactorySimultaneous::HistFactorySimultaneous(const char *name, const char *title, 
						 RooAbsCategoryLValue& inIndexCat) : 
  RooSimultaneous(name, title, inIndexCat ) {}


//_____________________________________________________________________________
RooStats::HistFactory::HistFactorySimultaneous::HistFactorySimultaneous(const char *name, const char *title, 
				 const RooArgList& inPdfList, RooAbsCategoryLValue& inIndexCat) :
  RooSimultaneous(name, title, inPdfList, inIndexCat) {}


//_____________________________________________________________________________
RooStats::HistFactory::HistFactorySimultaneous::HistFactorySimultaneous(const char *name, const char *title, 
				 map<string,RooAbsPdf*> pdfMap, RooAbsCategoryLValue& inIndexCat) :
  RooSimultaneous(name, title, pdfMap, inIndexCat) {}


//_____________________________________________________________________________
RooStats::HistFactory::HistFactorySimultaneous::HistFactorySimultaneous(const HistFactorySimultaneous& other, const char* name) : 
  RooSimultaneous(other, name) {}

//_____________________________________________________________________________
RooStats::HistFactory::HistFactorySimultaneous::HistFactorySimultaneous(const RooSimultaneous& other, const char* name) : 
  RooSimultaneous(other, name) {}

//_____________________________________________________________________________
RooStats::HistFactory::HistFactorySimultaneous::~HistFactorySimultaneous() 
{
  // Destructor
}


//_____________________________________________________________________________
RooAbsReal* RooStats::HistFactory::HistFactorySimultaneous::createNLL(RooAbsData& data, 
					       const RooCmdArg& arg1, const RooCmdArg& arg2, 
					       const RooCmdArg& arg3, const RooCmdArg& arg4, 
					       const RooCmdArg& arg5, const RooCmdArg& arg6, 
					       const RooCmdArg& arg7, const RooCmdArg& arg8) {
  
  // Probably not necessary because createNLL is virtual...

  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return createNLL(data,l) ;
  
}


//_____________________________________________________________________________

RooAbsReal* RooStats::HistFactory::HistFactorySimultaneous::createNLL(RooAbsData& data, const RooLinkedList& cmdList) {
  
  // We want to overload the method createNLL so it return
  // a RooBarlow-Beeston NLL function, which can be used
  // in HistFactory to minimize statistical uncertainty analytically
  //
  // The only problem is one of ownership
  // This HistFactorySimultaneous and the RooAbsData& data must
  // exist for as long as the RooBarlowBeestonLL does
  //
  // This could be solved if we instead refer to the cloned
  // pdf's and data set in the nll that we create here, but
  // it's unclear how to do so
  //
  // Also, check for ownership/memory issue with the newly created nll
  // and whether RooBarlowBeestonLL owns it, etc

  // Create a standard nll
  RooNLLVar* nll = (RooNLLVar*) RooSimultaneous::createNLL( data, cmdList );

  RooBarlowBeestonLL* bbnll = new RooBarlowBeestonLL("bbnll", "bbnll", *nll); //, *observables);
  bbnll->setPdf( this );
  bbnll->setDataset( &data );
  bbnll->initializeBarlowCache(); 
  
  return bbnll;

}
