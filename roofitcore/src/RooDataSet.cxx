/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.cc,v 1.28 2001/06/16 20:28:20 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu 
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   01-Nov-1999 DK Created initial version
 *   19-Apr-2000 DK Bug fix to read() method which caused an error on EOF
 *   21-Jun-2000 DK Change allocation unit in loadValues() from Float_t
 *                  to Double_t (this is still not quite right...)
 *   19-Oct-2000 DK Add constructor and read method for 6 variables.
 *   08-Nov-2000 DK Overhaul of loadValues() to support different leaf types.
 *   29-Nov-2000 WV Add support for reading hex numbers from ascii files
 *   30-Nov-2000 WV Add support for multiple file reading with optional common path
 *   09-Mar-2001 WV Migrate from RooFitTools and adapt to RooFitCore
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooDataSet is an extension of the ROOT TTree object and designated
// RFC object to hold unbinned fit data. Data sets can be created empty, 
// from a TTree or from an ASCII file. In all cases a RooArgSet serves 
// as column definition.
//
// A data set can hold RooRealVar, RooCategory and RooStringVar data types.
// Derived types can be added as a fundamental type, with the conversion
// done at the time of addition.

#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include <stdlib.h>
#include <math.h>

#include "TRegexp.h"
#include "TTreeFormula.h"
#include "TIterator.h"
#include "TObjArray.h"
#include "TTreeFormula.h"
#include "TIterator.h"
#include "TPaveText.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TROOT.h"

#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooFormula.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooHist.hh"

ClassImp(RooDataSet)

RooDataSet::RooDataSet(const char *name, const char *title, const RooArgSet& vars) :
  TTree(name, title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), _truth(), _branch(0)
{
  // Constructor with list of variables
  initialize(vars);
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *t, 
                       const RooArgSet& vars, const char *cuts) :
  TTree(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), _truth(), _branch(0), 
  _blindString(t->_blindString)
{
  // Constructor from existing data set with list of variables and cut expression
  initialize(vars);
  loadValues(t,cuts);
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *t, 
                       const RooArgSet& vars, Bool_t copyCache) :
  TTree(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), _truth(), _branch(0), 
  _blindString(t->_blindString)
{
  // Constructor from existing data set with list of variables that preserves the cache
  initialize(vars);
  initCache(t->_cachedVars) ;
  loadValues(t,"");
}

RooDataSet::RooDataSet(const char *name, const char *title, TTree *t, 
                       const RooArgSet& vars, const char *cuts) :
  TTree(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), _truth(), _branch(0)
{
  // Constructor from existing TTree with list of variables and cut expression
  initialize(vars);
  loadValues(t,cuts);
}

RooDataSet::RooDataSet(const char *name, const char *filename,
		       const char *treename,
                       const RooArgSet& vars, const char *cuts) :
  TTree(name,name), _vars("Dataset Variables"), _cachedVars("Cached Variables"), _truth(), _branch(0)
{
  // Constructor from TTree file with list of variables and cut expression
  initialize(vars);
  loadValues(filename,treename,cuts);
}


RooDataSet::RooDataSet(RooDataSet const & other) : 
  TTree(other.GetName(),other.GetTitle()), _vars("Dataset Variables"), 
  _cachedVars("Cached Variables"), _truth(), _branch(0)
{
  // Copy constructor
  initialize(other._vars) ;
  loadValues(&other,"") ;
}

RooDataSet::~RooDataSet()
{
  // Destructor

  // we cloned the initial AbsArgs ourselves and own them
  _vars.Delete() ;
}


void RooDataSet::initialize(const RooArgSet& vars) {
  // Initialize dataset: attach variables of internal ArgSet 
  // to the corresponding TTree branches

  // iterate over the variables for this dataset
  TIterator* iter = vars.MakeIterator() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)iter->Next())) {
    if (!var->isFundamental()) {
      cout << "RooDataSet::initialize(" << GetName() 
	   << "): Data set cannot contain non-fundamental types, ignoring " 
	   << var->GetName() << endl ;
    } else {
      RooAbsArg* varClone = (RooAbsArg*) var->Clone() ;
      varClone->attachToTree(*this) ;
      _vars.add(*varClone) ;
    }
  }
  delete iter ;

  _iterator= _vars.MakeIterator();
  _cacheIter = _cachedVars.MakeIterator() ;
}


void RooDataSet::initCache(const RooArgSet& cachedVars) 
{
  // Initialize cache of dataset: attach variables of ArgSet cache
  // to the corresponding TTree branches

  // iterate over the cache variables for this dataset
  TIterator* iter = cachedVars.MakeIterator() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)iter->Next())) {
    var->attachToTree(*this) ;
    _cachedVars.add(*var) ;
  }
  delete iter ;
}


void RooDataSet::loadValues(const char *filename, const char *treename,
			    const char *cuts) {
  // Load the value of a TTree stored in given file
  TFile *file= (TFile*)gROOT->GetListOfFiles()->FindObject(filename);
  if(!file) file= new TFile(filename);
  if(!file) {
    cout << "RooDataSet::loadValues: unable to open " << filename << endl;
  }
  else {
    TTree* tree= (TTree*)gDirectory->Get(treename);
    loadValues(tree,cuts);
  }
}

void RooDataSet::loadValues(const TTree *t, const char *cuts) 
{
  // Load values of given ttree

  // Clone source tree
  //TTree* tClone = (TTree*) t->Clone() ;
  TTree* tClone = t->CloneTree() ;
  
  // Clone list of variables
  RooArgSet sourceArgSet(_vars,"sourceArgSet") ;
  
  // Attach args in cloned list to cloned source tree
  TIterator* sourceIter =  sourceArgSet.MakeIterator() ;
  RooAbsArg* sourceArg(0) ;
  while (sourceArg=(RooAbsArg*)sourceIter->Next()) {
    sourceArg->attachToTree(*tClone) ;
  }

  // Create an event selector using the cuts provided, if any.
  RooFormula* select(0) ;
   if(0 != cuts && strlen(cuts)) {
     select=new RooFormula(cuts,cuts,_vars);
     if (!select || !select->ok()) {
       delete select;
       return ;
     }
   }

  // Loop over events in source tree   
  RooAbsArg* destArg(0) ;
  Int_t nevent= (Int_t)tClone->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=tClone->GetEntryNumber(i);
    if (entryNumber<0) break;
    tClone->GetEntry(entryNumber,1);
    
    // Does this event pass the cuts?
    if (select && select->eval()==0) continue ; 
        
    // Copy from source to destination
     _iterator->Reset() ;
     sourceIter->Reset() ;
     while (destArg = (RooAbsArg*)_iterator->Next()) {       
       sourceArg = (RooAbsArg*) sourceIter->Next() ;
       if (!sourceArg->isValid()) {
	 continue ;
       }
       destArg->copyCache(sourceArg) ;
     }   

     Fill() ;
   }

   SetTitle(t->GetTitle());
   if (select) delete select;
   delete tClone ;
}


void RooDataSet::append(RooDataSet& data) {
  // Append given data set to this data set
  loadValues(&data,0) ;
}


void RooDataSet::dump() {
  // DEBUG: Dump contents

  RooAbsArg* arg ;

  // Header line
  _iterator->Reset() ;
  while (arg = (RooAbsArg*)_iterator->Next()) {       
    cout << arg->GetName() << "  " ;
  }  
  cout << endl ;
     
  // Dump contents 
  Int_t nevent= (Int_t)GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);
     
    _iterator->Reset() ;
    // Copy from source to destination
    while (arg = (RooAbsArg*)_iterator->Next()) {
     arg->writeToStream(cout,kTRUE) ; 
     cout << " " ;
     }  
    cout << endl ;
  }
}


void RooDataSet::addColumn(RooAbsArg& newVar)
{
  // Add and precalculate new column, using given valHolder to store precalculate value

  // Create a fundamental object of the right type to hold newVar values
  RooAbsArg* valHolder= newVar.createFundamental();
  // Sanity check that the holder really is fundamental
  if(!valHolder->isFundamental()) {
    cout << GetName() << "::addColumn: holder argument is not fundamental: \""
	 << valHolder->GetName() << "\"" << endl;
    return;
  }

  // Clone current tree
  RooDataSet* cloneData = new RooDataSet(*this) ;

  // Clone variable and attach to cloned tree 
  RooAbsArg* newVarClone = (RooAbsArg*) newVar.Clone() ;
  newVarClone->redirectServers(cloneData->_vars,kFALSE) ;

  // Attach value place holder to this tree
  ((RooAbsArg*)valHolder)->attachToTree(*this) ;
  _vars.add(*valHolder) ;

  // Fill values of of placeholder
  Reset() ;
  for (int i=0 ; i<cloneData->GetEntries() ; i++) {
    cloneData->get(i) ;

    _vars = cloneData->_vars ;
    newVarClone->syncCache(this) ;
    valHolder->copyCache(newVarClone) ;

    Fill() ;
  }
  
  delete newVarClone;
  delete cloneData ;
}


void RooDataSet::cacheArg(RooAbsArg& newVar) 
{
  // Precalculate the values of given variable for this data set and allow
  // the data set to directly write the internal cache of given variable

  newVar.attachToTree(*this) ;
  _cachedVars.add(newVar) ;

  fillCacheArgs() ;
}


void RooDataSet::cacheArgs(RooArgSet& newVarSet) 
{
  // Call cacheArg for each argument in given list

  TIterator *iter = newVarSet.MakeIterator() ;
  RooAbsArg* arg ;

  while (arg=(RooAbsArg*)iter->Next()) {
    // Attach newVar to this tree
    arg->attachToTree(*this) ;
    _cachedVars.add(*arg) ;
  }

  // Recalculate the cached variables
  fillCacheArgs() ;
}


void RooDataSet::fillCacheArgs()
{
  // Recalculate contents of cached variables

  // Clone current tree
  RooDataSet* cloneData = new RooDataSet(*this) ;
  
  // Refill regular and cached variables of current tree from clone
  Reset() ;
  for (int i=0 ; i<cloneData->GetEntries() ; i++) {
    cloneData->get(i) ;

    // Copy the regular variables
    _vars = cloneData->_vars ;

    // Recalculate the cached variables
    RooAbsArg* cacheVar ;
    _cacheIter->Reset() ;
    while (cacheVar=(RooAbsArg*)_cacheIter->Next()) {
      cacheVar->syncCache(this) ;
    }

    Fill() ;
  }

  delete cloneData ;
}


void RooDataSet::add(const RooArgSet& data) {
  // Add a row of data elements  
  _vars= data;
  Fill();
}

RooPlot *RooDataSet::plotOn(RooPlot *frame, const char* cuts, Option_t* drawOptions) const 
{
  // Fill a histogram of values calculated from events in our dataset.

  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plot: frame is null" << endl;
    return 0;
  }
  RooAbsReal *var= frame->getPlotVar();
  if(0 == var) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // create a temporary histogram of this variable
  TH1F *hist= createHistogram(*var, cuts, "plot");
  if(0 == hist) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: createHistogram() failed" << endl;
    return 0;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist);
  if(0 == graph) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: unable to create a RooHist object" << endl;
    delete hist;
    return 0;
  }

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // add the RooHist to the specified plot
  frame->addPlotable(graph,drawOptions);

  // cleanup
  delete hist;

  return frame;  
}

TH1F* RooDataSet::createHistogram(const RooAbsReal& var, const char* cuts, const char *name) const
{
  // Create a TH1F histogram of the distribution of the specified variable
  // using this dataset. Apply any cuts to select which events are used.
  // The variable being plotted can either be contained directly in this
  // dataset, or else be a function of the variables in this dataset.
  // The histogram will be created using RooAbsReal::createHistogram() with
  // the name provided (with our dataset name prepended).

  Bool_t ownPlotVar(kFALSE) ;
  // Is this variable in our dataset?
  RooAbsReal* plotVar= (RooAbsReal*)_vars.find(var.GetName());
  if(0 == plotVar) {
    // Is this variable a client of our dataset?
    if (!var.dependsOn(_vars)) {
      cout << GetName() << "::createHistogram: Argument " << var.GetName() 
	   << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    plotVar = (RooAbsReal*) var.Clone()  ;
    ownPlotVar = kTRUE ;

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    plotVar->redirectServers(const_cast<RooArgSet&>(_vars)) ;
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

  // WVE use var instead of plotVar, otherwise binning properties
  // of data set copy of plot var are always used.
  TH1F *histo= var.createHistogram(histName.Data(), "Events");

  // Dump contents   
  Int_t nevent= (Int_t)GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);

    if (select && select->eval()==0) continue ;
    histo->Fill(plotVar->getVal()) ;
  }

  if (ownPlotVar) delete plotVar ;
  if (select) delete select ;

  return histo ;
}


Roo1DTable* RooDataSet::Table(RooAbsCategory& cat, const char* cuts, const char* opts)
{
  // Create and fill a 1-dimensional table for given category column

  // First see if var is in data set 
  RooAbsCategory* tableVar = (RooAbsCategory*) _vars.find(cat.GetName()) ;
  Bool_t ownPlotVar(kFALSE) ;
  if (!tableVar) {
    if (!cat.dependsOn(_vars)) {
      cout << "RooDataSet::Table(" << GetName() << "): Argument " << cat.GetName() 
	   << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    tableVar = (RooAbsCategory*) cat.Clone()  ;
    ownPlotVar = kTRUE ;    

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    tableVar->redirectServers(_vars) ;
  }

  Roo1DTable* table = tableVar->createTable("dataset") ;
  
  // Dump contents   
  Int_t nevent= (Int_t)GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);
    table->fill(*tableVar) ;
  }

  if (ownPlotVar) delete tableVar ;

  return table ;
}

void RooDataSet::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this dataset to the specified output stream.
  //
  //   Standard: number of entries
  //      Shape: list of variables we define & were generated with

  oneLinePrint(os,*this);
  if(opt >= Standard) {
    os << indent << "  Contains " << GetEntries() << " entries" << endl;
    if(opt >= Shape) {
      os << indent << "  Defines ";
      TString deeper(indent);
      deeper.Append("  ");
      _vars.printToStream(os,Standard,deeper);
      os << indent << "  Caches ";
      _cachedVars.printToStream(os,Standard,deeper);
      
      if(_truth.GetSize() > 0) {
	os << indent << "  Generated with ";
	_truth.printToStream(os,Shape,deeper);
      }
    }
  }
}

const RooArgSet* RooDataSet::get(Int_t index) const {

  // Return ArgSet containing given row of data

  Int_t ret = ((RooDataSet*)this)->GetEntry(index, 1) ;
  if(!ret) return 0;

  // Raise all dirty flags 
  _iterator->Reset() ;
  RooAbsArg* var(0) ;
  while (var=(RooAbsArg*)_iterator->Next()) {
    var->postTreeLoadHook() ;
    var->setValueDirty(kTRUE) ; // This triggers recalculation of all clients
  } 

  _cacheIter->Reset() ;
  while (var=(RooAbsArg*)_cacheIter->Next()) {
    var->postTreeLoadHook() ;
    var->setValueDirty(kTRUE)  ; // This triggers recalculation of all clients
    var->setValueDirty(kFALSE) ; // This prevent recalculation of self
  } 
  
  return &_vars;
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
    ifstream file(TString(commonPath)+TString(filename));

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
