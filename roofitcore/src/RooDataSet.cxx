/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.cc,v 1.52 2001/10/11 01:28:49 verkerke Exp $
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

// -- CLASS DESCRIPTION [DATA] --
// RooDataSet is a container class to hold unbinned data. Each data point
// in N-dimensional space is represented by a RooArgSet of RooRealVar, RooCategory 
// or RooStringVar objects 
//

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
#include "RooFitCore/RooArgList.hh"

ClassImp(RooDataSet)
;

RooDataSet::RooDataSet() {}


RooDataSet::RooDataSet(const char *name, const char *title, const RooArgSet& vars) :
  RooTreeData(name,title,vars)
{
  // Constructor of an empty data set from a RooArgSet defining the dimensions
  // of the data space.
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *dset, 
		       const RooArgSet& vars, const char *cuts) :
  RooTreeData(name,title,dset,vars,cuts)
{
  // Constructor of a data set from (part of) an existing data set. The dimensions
  // of the data set are defined by the 'vars' RooArgSet, which can be identical
  // to 'dset' dimensions, or a subset thereof. The 'cuts' string is an optional
  // RooFormula expression and can be used to select the subset of the data points 
  // in 'dset' to be copied. The cut expression can refer to any variable in the
  // source dataset. For cuts involving variables other than those contained in
  // the source data set, such as intermediate formula objects, use the 
  // equivalent constructor accepting RooFormulaVar reference as cut specification
  //
  // For most uses the RooAbsData::reduce() wrapper function, which uses this constructor, 
  // is the most convenient way to create a subset of an existing data
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *t, 
		       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooTreeData(name,title,t,vars,cutVar)
{
  // Constructor of a data set from (part of) an existing data set. The dimensions
  // of the data set are defined by the 'vars' RooArgSet, which can be identical
  // to 'dset' dimensions, or a subset thereof. The 'cutVar' formula variable
  // is used to select the subset of data points to be copied.
  // For subsets without selection on the data points, or involving cuts
  // operating exclusively and directly on the data set dimensions, the equivalent
  // constructor with a string based cut expression is recommended.
  //
  // For most uses the RooAbsData::reduce() wrapper function, which uses this constructor, 
  // is the most convenient way to create a subset of an existing data
}


RooDataSet::RooDataSet(const char *name, const char *title, TTree *t, 
		       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooTreeData(name,title,t,vars,cutVar)
{
  // Constructor of a data set from (part of) an ROOT TTRee. The dimensions
  // of the data set are defined by the 'vars' RooArgSet. For each dimension
  // specified, the TTree must have a branch with the same name. For category
  // branches, this branch should contain the numeric index value. Real dimensions
  // can be constructed from either 'Double_t' or 'Float_t' tree branches. In the
  // latter case, an automatic conversion is applied.
  //
  // The 'cutVar' formula variable
  // is used to select the subset of data points to be copied.
  // For subsets without selection on the data points, or involving cuts
  // operating exclusively and directly on the data set dimensions, the equivalent
  // constructor with a string based cut expression is recommended.
}


RooDataSet::RooDataSet(const char *name, const char *title, TTree *ntuple, 
		       const RooArgSet& vars, const char *cuts) :
  RooTreeData(name,title,ntuple,vars,cuts)
{
  // Constructor of a data set from (part of) an ROOT TTRee. The dimensions
  // of the data set are defined by the 'vars' RooArgSet. For each dimension
  // specified, the TTree must have a branch with the same name. For category
  // branches, this branch should contain the numeric index value. Real dimensions
  // can be constructed from either 'Double_t' or 'Float_t' tree branches. In the
  // latter case, an automatic conversion is applied.
  //
  // The 'cuts' string is an optional
  // RooFormula expression and can be used to select the subset of the data points 
  // in 'dset' to be copied. The cut expression can refer to any variable in the
  // vars argset. For cuts involving variables other than those contained in
  // the vars argset, such as intermediate formula objects, use the 
  // equivalent constructor accepting RooFormulaVar reference as cut specification
  //
}


RooDataSet::RooDataSet(const char *name, const char *filename, const char *treename, 
		       const RooArgSet& vars, const char *cuts) :
  RooTreeData(name,filename,treename,vars,cuts)
{
  // Constructor of a data set from (part of) a named ROOT TTRee in given ROOT file. 
  // The dimensions of the data set are defined by the 'vars' RooArgSet. For each dimension
  // specified, the TTree must have a branch with the same name. For category
  // branches, this branch should contain the numeric index value. Real dimensions
  // can be constructed from either 'Double_t' or 'Float_t' tree branches. In the
  // latter case, an automatic conversion is applied.
  //
  // The 'cuts' string is an optional
  // RooFormula expression and can be used to select the subset of the data points 
  // in 'dset' to be copied. The cut expression can refer to any variable in the
  // vars argset. For cuts involving variables other than those contained in
  // the vars argset, such as intermediate formula objects, use the 
  // equivalent constructor accepting RooFormulaVar reference as cut specification
  //
}


RooDataSet::RooDataSet(RooDataSet const & other, const char* newname) :
  RooTreeData(other,newname)
{
  // Copy constructor
}


RooDataSet::RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
		       const RooArgSet& vars, const RooFormulaVar* cutVar, Bool_t copyCache) :
  RooTreeData(name,title,ntuple,vars,cutVar, copyCache)
{
  // Protected constructor for internal use only
}


RooAbsData* RooDataSet::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache)
{
  // Implementation of RooAbsData virtual method that drives the RooAbsData::reduce() methods
  return new RooDataSet(GetName(), GetTitle(), this, varSubset, cutVar, copyCache) ;
}



RooDataSet::~RooDataSet()
{
  // Destructor
}



void RooDataSet::add(const RooArgSet& data, Double_t weight) 
{
  // Add a data point, with its coordinates specified in the 'data' argset, to the data set. 
  // Any variables present in 'data' but not in the dataset will be silently ignored
  //
  // The weight parameter is presently not supported

  _vars= data;
  Fill();
}


Bool_t RooDataSet::merge(RooDataSet* data1, RooDataSet* data2, RooDataSet* data3, 
			 RooDataSet* data4, RooDataSet* data5, RooDataSet* data6) 
{
  // Merge columns of supplied data set(s) with this data set.
  // All data sets must have equal number of entries.
  // In case of duplicate columns the column of the last dataset in the list prevails

  TList dsetList ;
  dsetList.Add(data1) ;
  if (data2) {
    dsetList.Add(data2) ;
    if (data3) {
      dsetList.Add(data3) ;
      if (data4) {
	dsetList.Add(data4) ;
	if (data5) {
	  dsetList.Add(data5) ;
	  if (data6) {
	    dsetList.Add(data6) ;
	  }
	}
      }
    }
  }

  return merge(dsetList) ;
}



Bool_t RooDataSet::merge(const TList& dsetList) 
{
  // Merge columns of supplied data set(s) with this data set.
  // All data sets must have equal number of entries.
  // In case of duplicate columns the column of the last dataset in the list prevails
  
  TIterator* iter = dsetList.MakeIterator() ;
  RooDataSet* data ;

  // Sanity checks: data sets must have the same size
  while(data=(RooDataSet*)iter->Next()) {
    if (numEntries()!=data->numEntries()) {
      cout << "RooDataSet::merge(" << GetName() << " ERROR: datasets have different size" << endl ;
      delete iter ;
      return kTRUE ;    
    }
  }

  // Clone current tree
  RooTreeData* cloneData = (RooTreeData*) Clone() ; 

  // Extend vars with elements of other dataset
  iter->Reset() ;
  while(data=(RooDataSet*)iter->Next()) {
    data->_iterator->Reset() ;
    RooAbsArg* arg ;
    while (arg=(RooAbsArg*)data->_iterator->Next()) {
      RooAbsArg* clone = _vars.addClone(*arg,kTRUE) ;
      if (clone) clone->attachToTree(*_tree) ;
    }
  }
	   
  // Refill current data set with data of clone and other data set
  Reset() ;
  for (int i=0 ; i<cloneData->numEntries() ; i++) {
    
    // Copy variables from self
    _vars = *cloneData->get(i) ;    

    // Copy variables from merge sets
    iter->Reset() ;
    while(data=(RooDataSet*)iter->Next()) {
      _vars = *data->get(i) ;
    }

    Fill() ;
  }
  
  delete cloneData ;
  delete iter ;
  return kFALSE ;
}




void RooDataSet::append(RooDataSet& data) {
  // Add all data points of given data set to this data set.
  // Eventual extra dimensions of 'data' will be stripped in transfer

  loadValues(data._tree,(RooFormulaVar*)0) ;
}



TH2F* RooDataSet::createHistogram(const RooAbsReal& var1, const RooAbsReal& var2, const char* cuts, const char *name) const
{
  // Create a TH2F histogram of the distribution of the specified variable
  // using this dataset. Apply any cuts to select which events are used.
  // The variable being plotted can either be contained directly in this
  // dataset, or else be a function of the variables in this dataset.
  // The histogram will be created using RooAbsReal::createHistogram() with
  // the name provided (with our dataset name prepended).
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


RooDataSet *RooDataSet::read(const char *fileList, RooArgList &variables,
			     const char *verbOpt, const char* commonPath, 
			     const char* indexCatName) {
  // Read given list of ascii files, and construct a data set, using the given
  // ArgList as structure definition.
  //
  // Multiple file names in fileList should be comma separated. Each
  // file is optionally prefixed with 'commonPath' if such a path is
  // provided
  //
  // The arglist specifies the dimensions of the dataset to be built
  // and describes the order in which these dimensions appear in the
  // ascii files to be read. 
  //
  // Each line in the ascii file should contain N white space separated
  // tokens, with N the number of args in 'variables'. Any text beyond
  // N tokens will be ignored with a warning message.
  // [ NB: This format is written by RooArgList::writeToStream() ]
  // 
  // If the value of any of the variables on a given line exceeds the
  // fit range associated with that dimension, the entire line will be
  // ignored. A warning message is printed in each case, unless the
  // 'Q' verbose option is given. (Option 'D' will provide additional
  // debugging information) The number of events read and skipped
  // is always summarized at the end.
  //
  // When multiple files are read, a RooCategory arg in 'variables' can 
  // optionally be designated to hold information about the source file 
  // of each data point. This feature is enabled by giving the name
  // of the (already existing) category variable in 'indexCatName'
  //
  // If no further information is given a label name 'fileNNN' will
  // be assigned to each event, where NNN is the sequential number of
  // the source file in 'fileList'.
  // 
  // Alternatively it is possible to override the default label names
  // of the index category by specifying them in the fileList string:
  // When instead of "file1.txt,file2.txt" the string 
  // "file1.txt:FOO,file2.txt:BAR" is specified, a state named "FOO"
  // is assigned to the index category for each event originating from
  // file1.txt. The labels FOO,BAR may be predefined in the index 
  // category via defineType(), but don't have to be
  //
  // Finally, one can also assign the same label to multiple files,
  // either by specifying "file1.txt:FOO,file2,txt:FOO,file3.txt:BAR"
  // or "file1.txt,file2.txt:FOO,file3.txt:BAR"
  //

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
  TString opts= verbOpt;
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
	Bool_t readError = variables.readFromStream(file,kTRUE,verbose) ;
	data->_vars = variables ;
// 	Bool_t readError = data->_vars.readFromStream(file,kTRUE,verbose) ;

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




Bool_t RooDataSet::write(const char* filename)
{
  // Open file for writing 
  ofstream ofs(filename) ;
  if (ofs.fail()) {
    cout << "RooDataSet::write(" << GetName() << ") cannot create file " << filename << endl ;
    return kTRUE ;
  }

  // Write all lines as arglist in compact mode
  cout << "RooDataSet::write(" << GetName() << ") writing ASCII file " << filename << endl ;
  Int_t i ;
  for (i=0 ; i<numEntries() ; i++) {
    RooArgList list(*get(i),"line") ;
    list.writeToStream(ofs,kTRUE) ;
  }

  if (ofs.fail()) {
    cout << "RooDataSet::write(" << GetName() << "): WARNING error(s) have occured in writing" << endl ;
  }
  return ofs.fail() ;
}
