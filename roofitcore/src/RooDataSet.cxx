/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.cc,v 1.46 2001/09/11 00:30:31 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 * History:
 *   01-Nov-1999 DK Created initial version
 *   30-Nov-2000 WV Add support for multiple file reading with optional common path
 *   09-Mar-2001 WV Migrate from RooFitTools and adapt to RooFitCore
 *   24-Aug-2001 AB Added TH2F * createHistogram method
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

#include <iostream.h>
#include <fstream.h>
#include "TH2.h"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooHist.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooFormulaVar.hh"

ClassImp(RooDataSet)
;



RooDataSet::RooDataSet() {}

RooDataSet::RooDataSet(const char *name, const char *title, const RooArgSet& vars) :
  RooTreeData(name,title,vars)
{
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
		       const RooArgSet& vars, const char *cuts) :
  RooTreeData(name,title,ntuple,vars,cuts)
{
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *t, 
		       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooTreeData(name,title,t,vars,cutVar)
{
}


RooDataSet::RooDataSet(const char *name, const char *title, TTree *t, 
		       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooTreeData(name,title,t,vars,cutVar)
{
}


RooDataSet::RooDataSet(const char *name, const char *title, TTree *ntuple, 
		       const RooArgSet& vars, const char *cuts) :
  RooTreeData(name,title,ntuple,vars,cuts)
{
}


RooDataSet::RooDataSet(const char *name, const char *filename, const char *treename, 
		       const RooArgSet& vars, const char *cuts) :
  RooTreeData(name,filename,treename,vars,cuts)
{
}


RooDataSet::RooDataSet(RooDataSet const & other, const char* newname=0) :
  RooTreeData(other,newname)
{
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
		       const RooArgSet& vars, const RooFormulaVar* cutVar, Bool_t copyCache) :
  RooTreeData(name,title,ntuple,vars,cutVar, copyCache)
{
}


RooAbsData* RooDataSet::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache)
{
  return new RooDataSet(GetName(), GetTitle(), this, varSubset, cutVar, copyCache) ;
}



RooDataSet::~RooDataSet()
{
}


void RooDataSet::add(const RooArgSet& data, Double_t weight) {
  // Add a row of data elements  
  _vars= data;
  Fill();
}



void RooDataSet::append(RooDataSet& data) {
  // Append given data set to this data set
  loadValues(data._tree,(RooFormulaVar*)0) ;
}



TH2F* RooDataSet::createHistogram(const RooAbsReal& var1, const RooAbsReal& var2, const char* cuts, const char *name) const
{
  return createHistogram(var1, var2, var1.getPlotBins(), var2.getPlotBins(), cuts, name);
}



TH2F* RooDataSet::createHistogram(const RooAbsReal& var1, const RooAbsReal& var2, Int_t nx, Int_t ny, const char* cuts, const char *name) const
{
  // Create a TH2F histogram of the distribution of the specified variable
  // using this dataset. Apply any cuts to select which events are used.
  // The variable being plotted can either be contained directly in this
  // dataset, or else be a function of the variables in this dataset.
  // The histogram will be created using RooAbsReal::createHistogram() with
  // the name provided (with our dataset name prepended).

  Bool_t ownPlotVarX(kFALSE) ;
  // Is this variable in our dataset?
  RooAbsReal* plotVarX= (RooAbsReal*)_vars.find(var1.GetName());
  if(0 == plotVarX) {
    // Is this variable a client of our dataset?
    if (!var1.dependsOn(_vars)) {
      cout << GetName() << "::createHistogram: Argument " << var1.GetName() 
           << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    plotVarX = (RooAbsReal*) var1.Clone()  ;
    ownPlotVarX = kTRUE ;

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    plotVarX->redirectServers(const_cast<RooArgSet&>(_vars)) ;
  }

  Bool_t ownPlotVarY(kFALSE) ;
  // Is this variable in our dataset?
  RooAbsReal* plotVarY= (RooAbsReal*)_vars.find(var2.GetName());
  if(0 == plotVarY) {
    // Is this variable a client of our dataset?
    if (!var2.dependsOn(_vars)) {
      cout << GetName() << "::createHistogram: Argument " << var2.GetName() 
           << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    plotVarY = (RooAbsReal*) var2.Clone()  ;
    ownPlotVarY = kTRUE ;

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    plotVarY->redirectServers(const_cast<RooArgSet&>(_vars)) ;
  }

  // Create selection formula if selection cuts are specified
  RooFormula* select(0) ;
  if(0 != cuts && strlen(cuts)) {
    select=new RooFormula(cuts,cuts,_vars);
    if (!select || !select->ok()) {
      delete select;
      return 0 ;
    }
  }
  
  TString histName(name);
  histName.Prepend("_");
  histName.Prepend(fName);

  // create the histogram
  TH2F* histogram=new TH2F(histName.Data(), "Events", nx, var1.getPlotMin(), var1.getPlotMax(), 
                                                      ny, var2.getPlotMin(), var2.getPlotMax());
  if(!histogram) {
    cout << fName << "::createHistogram: unable to create a new histogram" << endl;
    return 0;
  }

  // Dump contents  
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) 
  {
    Int_t entryNumber=_tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);

    if (select && select->eval()==0) continue ;
    histogram->Fill(plotVarX->getVal(), plotVarY->getVal()) ;
  }

  if (ownPlotVarX) delete plotVarX ;
  if (ownPlotVarY) delete plotVarY ;
  if (select) delete select ;

  return histogram ;
}


RooDataSet *RooDataSet::read(const char *fileList, RooArgSet &variables,
			     const char *options, const char* commonPath, 
			     const char* indexCatName) {
  //Read given ascii file(s), and construct a data set, using the given
  //ArgSet as structure definition. The possible options are:
  //  Q : be quiet about non-fatal parsing errors
  //  D : print extra debugging info
  // Need to document arguments and "blindState" more here...

  // Append blinding state category to variable list if not already there
  Bool_t ownIsBlind(kTRUE) ;
  RooAbsArg* blindState = variables.find("blindState") ;
  if (!blindState) {
    blindState = new RooCategory("blindState","Blinding State") ;
    variables.add(*blindState) ;
  } else {
    ownIsBlind = kFALSE ;    
    if (blindState->IsA()!=RooCategory::Class()) {
      cout << "RooDataSet::read: ERROR: variable list already contains" 
	   << "a non-RooCategory blindState member" << endl ;
      return 0 ;
    }
    cout << "RooDataSet::read: WARNING: recycling existing "
         << "blindState category in variable list" << endl ;
  }
  RooCategory* blindCat = (RooCategory*) blindState ;

  // Configure blinding state category
  blindCat->setAttribute("Dynamic") ;
  blindCat->defineType("Normal",0) ;
  blindCat->defineType("Blind",1) ;

  // parse the option string
  TString opts= options;
  opts.ToLower();
  Bool_t verbose= !opts.Contains("q");
  Bool_t debug= opts.Contains("d");
  Bool_t haveRefBlindString(false), haveUnblindFile(false) ;

  RooDataSet *data= new RooDataSet("dataset", fileList, variables);
  if (ownIsBlind) { variables.remove(*blindState) ; delete blindState ; }
  if(!data) {
    cout << "RooDataSet::read: unable to create a new dataset"
	 << endl;
    return 0;
  }

  // Redirect blindCat to point to the copy stored in the data set
  blindCat = (RooCategory*) data->_vars.find("blindState") ;

  // Find index category, if requested
  RooCategory *indexCat(0), *indexCatOrig(0) ;
  if (indexCatName) { 
    RooAbsArg* tmp(0) ;
    tmp = data->_vars.find(indexCatName) ;
    if (!tmp) {
      cout << "RooDataSet::read: no index category named " 
	   << indexCatName << " in supplied variable list" << endl ;
      return 0 ;
    }
    if (tmp->IsA()!=RooCategory::Class()) {
      cout << "RooDataSet::read: variable " << indexCatName 
	   << " is not a RooCategory" << endl ;
      return 0 ;
    }
    indexCat = (RooCategory*)tmp ;
    
    // Prevent RooArgSet from attempting to read in indexCat
    indexCat->setAttribute("Dynamic") ;
  }


  Int_t outOfRange(0) ;

  // Make local copy of file list for tokenizing 
  char fileList2[10240] ;
  strcpy(fileList2,fileList) ;
  
  // Loop over all names in comma separated list
  char *filename = strtok(fileList2,",") ;
  Int_t fileSeqNum(0) ;
  while (filename) {

    // Determine index category number, if this option is active
    if (indexCat) {

      // Find and detach optional file category name 
      char *catname = strchr(filename,':') ;

      if (catname) {
	// Use user category name if provided
	*catname=0 ;
	catname++ ;

	const RooCatType* type = indexCat->lookupType(catname,kFALSE) ;
	if (type) {
	  // Use existing category index
	  indexCat->setIndex(type->getVal()) ;
	} else {
	  // Register cat name
	  indexCat->defineType(catname,fileSeqNum) ;
	  indexCat->setIndex(fileSeqNum) ;
	}
      } else {
	// Assign autogenerated name
	char newLabel[128] ;
	sprintf(newLabel,"file%03d",fileSeqNum) ;
	if (indexCat->defineType(newLabel,fileSeqNum)) {
	  cout << "RooDataSet::read: Error, cannot register automatic type name " << newLabel 
	       << " in index category " << indexCat->GetName() << endl ;
	  return 0 ;
	}	
	// Assign new category number
	indexCat->setIndex(fileSeqNum) ;
      }
    }

    cout << "RooDataSet::read: reading file " << filename << endl ;

    // Prefix common path 
    TString fullName(commonPath) ;
    fullName.Append(filename) ;
    ifstream file(fullName) ;

    if(!file.good()) {
      cout << "RooDataSet::read: unable to open "
	   << filename << ", skipping" << endl;
    }
    
    Double_t value;
    Int_t line(0) ;
    Bool_t haveBlindString(false) ;

    while(file.good() && !file.eof()) {
      line++;
      if(debug) cout << "reading line " << line << endl;

      // process comment lines
      if (file.peek() == '#')
	{
	  if(debug) cout << "skipping comment on line " << line << endl;
	    
	  TString line ;
	  line.ReadLine(file) ;
	  if (line.Contains("#BLIND#")) {	  
	    haveBlindString = true ;
	    if (haveRefBlindString) {
	      
	      // compare to ref blind string 
	      TString curBlindString(line(7,line.Length()-7)) ;
	      if (debug) cout << "Found blind string " << curBlindString << endl ;
	      if (curBlindString != data->_blindString) {
		  cout << "RooDataSet::read: ERROR blinding string mismatch, abort" << endl ;
		  return 0 ;
		}
	      
	    } else {
	      // store ref blind string 
	      data->_blindString=TString(line(7,line.Length()-7)) ;
	      if (debug) cout << "Storing ref blind string " << data->_blindString << endl ;
	      haveRefBlindString = true ;
	    }	    
	  }     
	}
      else {	

	// Skip empty lines 
	// if(file.peek() == '\n') { file.get(); }

	// Read single line
	Bool_t readError = data->_vars.readFromStream(file,kTRUE,verbose) ;

	// Stop at end of file or on read error
	if(file.eof()) break ;	
	if(!file.good()) {
	  cout << "RooDataSet::read(static): read error at line " << line << endl ;
	  break;
	}	

	if (readError) {
	  outOfRange++ ;
	  continue ;
	}
	blindCat->setIndex(haveBlindString) ;
	data->Fill(); // store this event
      }
    }

    file.close();

    // get next file name 
    filename = strtok(0,",") ;
    fileSeqNum++ ;
  }

  if (indexCat) {
    // Copy dynamically defined types from new data set to indexCat in original list
    RooCategory* origIndexCat = (RooCategory*) variables.find(indexCatName) ;
    TIterator* tIter = indexCat->typeIterator() ;
    RooCatType* type(0) ;
      while (type=(RooCatType*)tIter->Next()) {
	origIndexCat->defineType(type->GetName(),type->getVal()) ;
      }
  }
  cout << "RooDataSet::read: read " << data->GetEntries()
       << " events (ignored " << outOfRange << " out of range events)" << endl;
  return data;
}
