/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTreeData.cc,v 1.5 2001/09/24 23:06:01 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu 
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
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
 *   24-Aug-2001 AB Added TH2F * createHistogram method
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooTreeData is an extension of the ROOT TTree object and designated
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

#include "RooFitCore/RooTreeData.hh"
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
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooTrace.hh"

ClassImp(RooTreeData)
;


RooTreeData::RooTreeData() 
{
  RooTrace::create(this) ; 
}


RooTreeData::RooTreeData(const char *name, const char *title, const RooArgSet& vars) :
  RooAbsData(name,title,vars), _truth("Truth")
{
  RooTrace::create(this) ;
  _tree = new TTree(name, title) ;

  // Constructor with list of variables
  initialize(vars);
}


RooTreeData::RooTreeData(const char *name, const char *title, RooTreeData *t, 
                       const RooArgSet& vars, const char *cuts) :
 RooAbsData(name,title,vars), _truth("Truth"), 
  _blindString(t->_blindString)
{
  RooTrace::create(this) ;
  _tree = new TTree(name, title) ;

  // Constructor from existing data set with list of variables and cut expression
  initialize(vars);

  // Create a RooFormulaVar cut from given cut expression
  // Attach formula to original data set
  RooFormulaVar cutVar(cuts,cuts,t->_vars) ;

  loadValues(t->_tree,&cutVar);
}


RooTreeData::RooTreeData(const char *name, const char *title, RooTreeData *t, 
                       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooAbsData(name,title,vars),_truth("Truth"), 
  _blindString(t->_blindString)
{
  RooTrace::create(this) ;
  _tree = new TTree(name, title) ;

  // Constructor from existing data set with list of variables and cut expression
  initialize(vars);

  // Deep clone cutVar and attach clone to this dataset
  RooArgSet* tmp = (RooArgSet*) RooArgSet(cutVar).snapshot() ;
  RooFormulaVar* cloneVar = (RooFormulaVar*) tmp->find(cutVar.GetName()) ;
  cloneVar->attachDataSet(*this) ;

  loadValues(t->_tree,cloneVar);

  delete tmp ;
}


RooTreeData::RooTreeData(const char *name, const char *title, TTree *t, 
                       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooAbsData(name,title,vars), _truth("Truth")
{
  RooTrace::create(this) ;
  _tree = new TTree(name, title) ;

  // Constructor from existing data set with list of variables and cut expression
  initialize(vars);

  // Deep clone cutVar and attach clone to this dataset
  RooArgSet* tmp = (RooArgSet*) RooArgSet(cutVar).snapshot() ;
  RooFormulaVar* cloneVar = (RooFormulaVar*) tmp->find(cutVar.GetName()) ;
  cloneVar->attachDataSet(*this) ;

  loadValues(t,cloneVar);

  delete tmp ;
}


RooTreeData::RooTreeData(const char *name, const char *title, RooTreeData *t, 
                       const RooArgSet& vars, const RooFormulaVar* cutVar, Bool_t copyCache) :
  RooAbsData(name,title,vars), _truth("Truth"), 
  _blindString(t->_blindString)
{
  RooTrace::create(this) ;
  _tree = new TTree(name, title) ;

  // Deep clone cutVar and attach clone to this dataset
  RooFormulaVar* cloneVar(0) ;
  if (cutVar) {
    RooArgSet* tmp = (RooArgSet*) RooArgSet(*cutVar).snapshot() ;
    cloneVar = (RooFormulaVar*) tmp->find(cutVar->GetName()) ;
    cloneVar->attachDataSet(*t) ;
  }

  // Constructor from existing data set with list of variables that preserves the cache
  initialize(vars);
  initCache(t->_cachedVars) ;
  
  loadValues(t->_tree,cloneVar);

  if (cloneVar) delete cloneVar ;
}



RooTreeData::RooTreeData(const char *name, const char *title, TTree *t, 
                       const RooArgSet& vars, const char *cuts) :
  RooAbsData(name,title,vars), _truth("Truth")
{
  RooTrace::create(this) ;
  _tree = new TTree(name, title) ;

  // Constructor from existing TTree with list of variables and cut expression
  initialize(vars);

  // Create a RooFormulaVar cut from given cut expression
  RooFormulaVar cutVar(cuts,cuts,_vars) ;

  loadValues(t,&cutVar);
}



RooTreeData::RooTreeData(const char *name, const char *filename,
		       const char *treename,
                       const RooArgSet& vars, const char *cuts) :
  RooAbsData(name,name,vars), _truth("Truth")
{
  RooTrace::create(this) ;
  _tree = new TTree(name, name) ;

  // Constructor from TTree file with list of variables and cut expression
  initialize(vars);

  // Create a RooFormulaVar cut from given cut expression
  RooFormulaVar cutVar(cuts,cuts,_vars) ;

  loadValues(filename,treename,&cutVar);
}


RooTreeData::RooTreeData(RooTreeData const & other, const char* newName) : 
  RooAbsData(other,newName), _truth("Truth")
{
  // Copy constructor
  RooTrace::create(this) ;
  _tree = new TTree(newName, other.GetTitle()) ;

  initialize(other._vars) ;
  loadValues(other._tree,0) ;
}


RooTreeData::~RooTreeData()
{
  // Destructor
  RooTrace::destroy(this) ;

  delete _tree ;
}


void RooTreeData::initialize(const RooArgSet& vars) {
  // Initialize dataset: attach variables of internal ArgSet 
  // to the corresponding TTree branches

  // Attach each variable to the dataset
  _iterator->Reset() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)_iterator->Next())) {
    var->attachToTree(*_tree) ;
  }
}


void RooTreeData::initCache(const RooArgSet& cachedVars) 
{
  // Initialize cache of dataset: attach variables of ArgSet cache
  // to the corresponding TTree branches

  // iterate over the cache variables for this dataset
  TIterator* iter = cachedVars.createIterator() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)iter->Next())) {
    var->attachToTree(*_tree) ;
    _cachedVars.add(*var) ;
  }
  delete iter ;
}


void RooTreeData::loadValues(const char *filename, const char *treename,
			    RooFormulaVar* cutVar) {
  // Load the value of a TTree stored in given file
  TFile *file= (TFile*)gROOT->GetListOfFiles()->FindObject(filename);
  if(!file) file= new TFile(filename);
  if(!file) {
    cout << "RooTreeData::loadValues: unable to open " << filename << endl;
  }
  else {
    TTree* tree= (TTree*)gDirectory->Get(treename);
    loadValues(tree,cutVar);
  }
}


void RooTreeData::loadValues(const TTree *t, RooFormulaVar* select) 
{
  // Load values of given ttree

  // Clone source tree
  TTree* tClone = ((TTree*)t)->CloneTree() ;
  
  // Clone list of variables  
  RooArgSet *sourceArgSet = (RooArgSet*) _vars.snapshot(kFALSE) ;
  
  // Attach args in cloned list to cloned source tree
  TIterator* sourceIter =  sourceArgSet->createIterator() ;
  RooAbsArg* sourceArg(0) ;
  while (sourceArg=(RooAbsArg*)sourceIter->Next()) {
    sourceArg->attachToTree(*tClone) ;
  }

  // Redirect formula servers to sourceArgSet
  if (select) {
    select->recursiveRedirectServers(*sourceArgSet) ;

    RooArgSet branchList ;
    select->branchNodeServerList(&branchList) ;
    TIterator* iter = branchList.createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      arg->setOperMode(RooAbsArg::ADirty) ;
    }
    delete iter ;
  }

  // Loop over events in source tree   
  RooAbsArg* destArg(0) ;
  Int_t nevent= (Int_t)tClone->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=tClone->GetEntryNumber(i);
    if (entryNumber<0) break;
    tClone->GetEntry(entryNumber,1);
 
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

     // Does this event pass the cuts?
     if (select && select->getVal()==0) {
       continue ; 
     }

     Fill() ;
   }
  
  SetTitle(t->GetTitle());

  delete sourceIter ;
  delete sourceArgSet ;
  delete tClone ;
}


void RooTreeData::dump() {
  // DEBUG: Dump contents

  RooAbsArg* arg ;

  // Header line
  _iterator->Reset() ;
  while (arg = (RooAbsArg*)_iterator->Next()) {       
    cout << arg->GetName() << "  " ;
  }  
  cout << endl ;
     
  // Dump contents 
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=_tree->GetEntryNumber(i);
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


void RooTreeData::cacheArg(RooAbsArg& newVar) 
{
  // Precalculate the values of given variable for this data set and allow
  // the data set to directly write the internal cache of given variable

  newVar.attachToTree(*_tree) ;
  _cachedVars.add(newVar) ;

  fillCacheArgs() ;
}


void RooTreeData::cacheArgs(RooArgSet& newVarSet) 
{
  // Call cacheArg for each argument in given list

  TIterator *iter = newVarSet.createIterator() ;
  RooAbsArg* arg ;

  while (arg=(RooAbsArg*)iter->Next()) {
    // Attach newVar to this tree
    arg->attachToTree(*_tree) ;

    // Remove all server links 
//     TIterator* sIter = arg->serverIterator() ;
//     RooAbsArg* server ;
//     while(server=(RooAbsArg*)sIter->Next()) {
//       arg->removeServer(*server) ;
//     }
//     delete sIter ;

    _cachedVars.add(*arg) ;
  }
  delete iter ;
  
  // Recalculate the cached variables
  fillCacheArgs() ;
}


void RooTreeData::fillCacheArgs()
{
  // Recalculate contents of cached variables

  // Clone current tree
  RooTreeData* cloneData = (RooTreeData*) Clone() ; //new RooTreeData(*this) ;
  
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
      cacheVar->syncCache(&_vars) ;
    }

    Fill() ;
  }

  delete cloneData ;
}


const RooArgSet* RooTreeData::get(Int_t index) const {

  // Return ArgSet containing given row of data

  Int_t ret = ((RooTreeData*)this)->GetEntry(index, 1) ;
  if(!ret) return 0;

  if (_doDirtyProp) {
    // Raise all dirty flags 
    _iterator->Reset() ;
    RooAbsArg* var(0) ;
    while (var=(RooAbsArg*)_iterator->Next()) {
      var->setValueDirty() ; // This triggers recalculation of all clients
    } 
    
    _cacheIter->Reset() ;
    while (var=(RooAbsArg*)_cacheIter->Next()) {
      var->setValueDirty()  ; // This triggers recalculation of all clients, but doesn't recalculate self
      var->clearValueDirty() ; 
    } 
  }

  return &_vars;
}


RooAbsArg* RooTreeData::addColumn(RooAbsArg& newVar)
{
  // Add and precalculate new column, using given valHolder to store precalculate value

  // Create a fundamental object of the right type to hold newVar values
  RooAbsArg* valHolder= newVar.createFundamental();
  // Sanity check that the holder really is fundamental
  if(!valHolder->isFundamental()) {
    cout << GetName() << "::addColumn: holder argument is not fundamental: \""
	 << valHolder->GetName() << "\"" << endl;
    return 0;
  }

  // Clone current tree
  RooTreeData* cloneData = (RooTreeData*) Clone() ; //new RooTreeData(*this) ;  

  // Clone variable and attach to cloned tree 
  RooArgSet* newVarCloneList = (RooArgSet*) RooArgSet(newVar).snapshot() ;  
  RooAbsArg* newVarClone = newVarCloneList->find(newVar.GetName()) ;
  newVarClone->recursiveRedirectServers(cloneData->_vars,kFALSE) ;

//   RooAbsArg* newVarClone = (RooAbsArg*) newVar.Clone() ;
//   newVarClone->redirectServers(cloneData->_vars,kFALSE) ;

  // Attach value place holder to this tree
  ((RooAbsArg*)valHolder)->attachToTree(*_tree) ;
  _vars.addOwned(*valHolder) ;

  // Fill values of of placeholder
  Reset() ;
  for (int i=0 ; i<cloneData->GetEntries() ; i++) {
    cloneData->get(i) ;

    _vars = cloneData->_vars ;

    newVarClone->syncCache(&_vars) ;
    valHolder->copyCache(newVarClone) ;

    Fill() ;
  }
  
  delete newVarCloneList;
//   delete newVarClone;
  delete cloneData ;
  
  return valHolder ;
}


RooPlot *RooTreeData::plotOn(RooPlot *frame, const RooFormulaVar* cutVar, Option_t* drawOptions) const 
{
  return 0 ;
}



RooPlot *RooTreeData::plotOn(RooPlot *frame, const char* cuts, Option_t* drawOptions) const 
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




RooPlot* RooTreeData::plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, 
				 const char* cut, Option_t* drawOptions) const 
{
  cout << "RooTreeData::plotAsymOn(" << GetName() << ") not implemented." << endl ;
  return frame ;
}




TH1F* RooTreeData::createHistogram(const RooAbsReal& var, const char* cuts, const char *name) const
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
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=_tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);

    if (select && select->eval()==0) continue ;
    histo->Fill(plotVar->getVal(),weight()) ;
  }

  if (ownPlotVar) delete plotVar ;
  if (select) delete select ;

  return histo ;
}



Roo1DTable* RooTreeData::table(const RooAbsCategory& cat, const char* cuts, const char* opts) const
{
  // Create and fill a 1-dimensional table for given category column

  // First see if var is in data set 
  RooAbsCategory* tableVar = (RooAbsCategory*) _vars.find(cat.GetName()) ;
  Bool_t ownPlotVar(kFALSE) ;
  if (!tableVar) {
    if (!cat.dependsOn(_vars)) {
      cout << "RooTreeData::Table(" << GetName() << "): Argument " << cat.GetName() 
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
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=_tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);
    table->fill(*tableVar,weight()) ;
  }

  if (ownPlotVar) delete tableVar ;

  return table ;
}


void RooTreeData::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this dataset to the specified output stream.
  //
  //   Standard: number of entries
  //      Shape: list of variables we define & were generated with

  oneLinePrint(os,*this);
  if(opt >= Standard) {
    os << indent << "  Contains " << numEntries(kTRUE) << " entries" << endl;
    if(opt >= Shape) {
      os << indent << "  Defines ";
      TString deeper(indent);
      deeper.Append("  ");
      _vars.printToStream(os,Standard,deeper);
      os << indent << "  Caches ";
      _cachedVars.printToStream(os,Standard,deeper);
      
      if(_truth.getSize() > 0) {
	os << indent << "  Generated with ";
	_truth.printToStream(os,Shape,deeper);
      }
    }
  }
}






