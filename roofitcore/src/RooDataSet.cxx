/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.cc,v 1.4 2001/03/17 00:32:54 verkerke Exp $
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
#include "BaBar/BaBar.hh"

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

ClassImp(RooDataSet)

RooDataSet::RooDataSet(const char *name, const char *title, const RooArgSet& vars) :
  TTree(name, title), _vars("Dataset Variables"), _truth(), _branch(0)
{
  initialize(vars);
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *t, 
                       const RooArgSet& vars, const char *cuts) :
  TTree(name,title), _vars("Dataset Variables"), _truth(), _branch(0), _blindString(t->_blindString)
{
  initialize(vars);
  loadValues(t,cuts);
}

RooDataSet::RooDataSet(const char *name, const char *title, TTree *t, 
                       const RooArgSet& vars, const char *cuts) :
  TTree(name,title), _vars("Dataset Variables"), _truth(), _branch(0)
{
  initialize(vars);
  loadValues(t,cuts);
}

RooDataSet::RooDataSet(const char *name, const char *filename,
		       const char *treename,
                       const RooArgSet& vars, const char *cuts) :
  TTree(name,name), _vars("Dataset Variables"), _truth(), _branch(0)
{
  initialize(vars);
  loadValues(filename,treename,cuts);
}

RooDataSet::~RooDataSet()
{
  // we cloned the initial AbsArgs ourselves and own them
  _vars.Delete() ;
}


void RooDataSet::append(RooDataSet& data) {
  loadValues(&data,0) ;
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

void RooDataSet::loadValues(TTree *t, const char *cuts) 
{
  // Load values of given ttree

  // Clone source tree
  TTree* tClone = (TTree*) t->Clone() ;
  
  // Clone list of variables
  RooArgSet sourceArgSet("sourceArgSet",_vars) ;
  
  // Attach args in cloned list to cloned source tree
  TIterator* sourceIter =  sourceArgSet.MakeIterator() ;
  RooAbsArg* sourceArg(0) ;
  while (sourceArg=(RooAbsArg*)sourceIter->Next()) {
    sourceArg->attachToTree(*tClone) ;
  }

  // Create an event selector using the cuts provided, if any.
  TTreeFormula *select(0);
  if(0 != cuts && strlen(cuts)) {
    select=new TTreeFormula("Selection",cuts,t);
    if (!select || !select->GetNdim()) {
      delete select;
      select=0;
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
    if(0 != select) {
      select->GetNdata();
      if (select->EvalInstance(0)==0) continue;
    }
        
    // Copy from source to destination
     _iterator->Reset() ;
     sourceIter->Reset() ;
     while (destArg = (RooAbsArg*)_iterator->Next()) {       
       sourceArg = (RooAbsArg*) sourceIter->Next() ;
       if (!sourceArg->isValid()) {
	 continue ;
       }
       *destArg = *sourceArg ;
    }   
     Fill() ;
   }

   SetTitle(t->GetTitle());
   delete select;
   delete tClone ;
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
    GetEntry(entryNumber,1);
     
    _iterator->Reset() ;
    // Copy from source to destination
    while (arg = (RooAbsArg*)_iterator->Next()) {
     arg->writeToStream(cout,kTRUE) ; 
     cout << " " ;
     }  
    cout << endl ;
  }
}


void RooDataSet::initialize(const RooArgSet& vars) {
  // Initialize dataset: attach variables of internal ArgSet 
  // to the corresponding TTree branches

  // iterate over the variables for this dataset
  TIterator* iter = vars.MakeIterator() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)iter->Next())) {
    if (var->isDerived()) {
      cout << "RooDataSet::initialize(" << GetName() 
	   << "): Data set cannot contain derived values, ignoring " 
	   << var->GetName() << endl ;
    } else {
      RooAbsArg* varClone = (RooAbsArg*) var->Clone() ;
      varClone->attachToTree(*this) ;
      _vars.add(*varClone) ;
    }
  }

  _iterator= _vars.MakeIterator();
}


void RooDataSet::add(const RooArgSet& data) {
  // Add a row of data elements  virtual void attachToTree(TTree& t, Int_t bufSize=32000) = 0 ;
  _vars= data;
  Fill();
}


TH1F* RooDataSet::Plot(RooAbsReal& var, const char* cuts, const char* opts)
{
  // Plot distribution given variable for this data set 
  
  // First see if var is in data set 
  RooAbsReal* plotVar = (RooAbsReal*) _vars.find(var.GetName()) ;
  Bool_t ownPlotVar(kFALSE) ;
  if (!plotVar) {
    if (!var.dependsOn(_vars)) {
      cout << "RooDataSet::Plot: Argument " << var.GetName() 
	   << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    plotVar = (RooAbsReal*) var.Clone()  ;
    ownPlotVar = kTRUE ;    

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    plotVar->redirectServers(_vars) ;
  }

  TH1F *histo= plotVar->createHistogram("dataset", "Events");

  // Dump contents   
  Int_t nevent= (Int_t)GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);
    histo->Fill(plotVar->getVal()) ;
  }

  if (ownPlotVar) delete plotVar ;
  return histo ;
}


Roo1DTable* RooDataSet::Table(RooAbsCategory& cat, const char* cuts, const char* opts)
{
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


void RooDataSet::printToStream(ostream& os, PrintOption opt) 
{
  // Print structure of this data set
  cout << "RooDataSet \"" << GetTitle() << "\" contains" << endl
       << GetEntries() << " values for ";
  _vars.print(RooAbsArg::Shape);
  if(_truth.GetSize() > 0) {
    cout << "and was generated with ";
    _truth.print();
  }
}

const RooArgSet* RooDataSet::get(Int_t index) const {
  // Return ArgSet containing given row of data

  if(!((RooDataSet*)this)->GetEntry(index, 1)) return 0;

  // Raise all dirty flags 
  _iterator->Reset() ;
  RooAbsArg* var(0) ;
  while (var=(RooAbsArg*)_iterator->Next()) {
    var->postTreeLoadHook() ;
    var->setValueDirty(kTRUE) ;
  } 
  
  return &_vars;
}


RooDataSet *RooDataSet::read(const char *fileList, RooArgSet &variables,
			     const char *options, const char* commonPath) {
  //Read given ascii file, and construct a data set, using the given
  //ArgSet as structure definition

  // parse the option string
  TString opts= options;
  opts.ToLower();
  Bool_t verbose= !opts.Contains("q");
  Bool_t debug= opts.Contains("d");
  Bool_t haveRefBlindString(false), haveUnblindFile(false) ;

  RooDataSet *data= new RooDataSet("dataset", fileList, variables);
  if(!data) {
    cout << "RooDataSet::read: unable to create a new dataset"
	 << endl;
    return 0;
  }

  Int_t outOfRange(0) ;

  // Make local copy of file list for tokenizing 
  char fileList2[10240] ;
  strcpy(fileList2,fileList) ;
  
  // Loop over all names in comma separated list
  char *filename = strtok(fileList2,",") ;
  while (filename) {
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
	      TString curBlindString(line(7,line.Length()-8)) ;
	      if (verbose) cout << "Found blind string " << curBlindString << endl ;
	      if (curBlindString != data->_blindString) {
		  cout << "RooDataSet::read: ERROR blinding string mismatch, abort" << endl ;
		  return 0 ;
		}
	      
	    } else {
	      // store ref blind string 
	      data->_blindString=TString(line(7,line.Length()-8)) ;
	      if (verbose) cout << "Storing ref blind string " << data->_blindString << endl ;
	      haveRefBlindString = true ;
	    }
	  }     
	}
      else {	
	// read a value for each variable on one line
	Int_t c, nRead(0), size= data->_vars.GetSize();
	for(Int_t index= 0; index < size; index++) {
	  RooAbsArg *var= (RooAbsArg*)data->_vars.At(index);
	  // skip leading white space
	  while((c= file.peek()) == ' ' || c == '\t') file.get();
	  // is this the end of the current line or the file?
	  if(file.peek() == '\n') {
	    file.get();
	    if(index == 0) {
	      if(debug) cout << "skipping blank line " << line << endl;
	      nRead= -1;
	    }
	    else {
	      cout << "RooDataSet::read: found unexpected end of line "
		   << line << endl;
	    }
	    break;
	  }
	  
	  // Read from stream in compact mode
	  Bool_t isValid = !var->readFromStream(file,kTRUE,verbose) ;

	  if(file.eof()) {
	    if(index == 0) {
	      if(debug) cout << "reached normal end-of-file" << endl;
	      nRead= -1;
	    }
	    else {
	      cout << "RooDataSet::read: found unexpected end of file" << endl;
	    }
	    break;
	  }

	  if (!isValid) {
	    while(file.good() && !file.eof() && (c= file.get()) != '\n');
	    outOfRange++;
	    nRead= -1;
	    break;
	  }

	  if(!file.good() || file.eof()) {
	    cout << "RooDataSet::read: error reading line "
		 << line << endl;
	    break;
	  }
	  nRead++;
	}
	if(nRead < 0) continue;
	if(nRead < size) break;
	data->Fill(); // store this event
	// skip over the rest of the line
	Bool_t extra(kFALSE);
	while(file.good() && !file.eof() && (c= file.get()) != '\n') {
	  if(c != ' ' && c != '\t') extra= kTRUE;
	}
	if(extra) cout << "RooDataSet::read: ignoring extra input "
		       << "on the end of line " << line << endl;
      }
    }

    // check for blind string consistency 
    if (!haveBlindString) haveUnblindFile=true ;
    if ((haveRefBlindString && !haveBlindString) ||
	(haveBlindString && haveUnblindFile)) {
      cout << "RooDataSet::read: ERROR: mixing blind and unblind files, abort" << endl ;
      return 0 ;
    }

    file.close();

    // get next file name 
    filename = strtok(0,",") ;
  }
  cout << "RooDataSet::read: read " << data->GetEntries()
       << " events (ignored " << outOfRange << " out of range events)" << endl;
  return data;
}
