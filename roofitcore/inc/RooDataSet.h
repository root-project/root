/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.rdl,v 1.34 2001/10/11 01:28:50 verkerke Exp $
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
#ifndef ROO_DATA_SET
#define ROO_DATA_SET

#include "RooFitCore/RooTreeData.hh"


class RooDataSet : public RooTreeData {
public:

  // Constructors, factory methods etc.
  RooDataSet() ; 
  RooDataSet(const char *name, const char *title, const RooArgSet& vars) ;
  RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooDataSet(const char *name, const char *title, RooDataSet *t, 
	     const RooArgSet& vars, const RooFormulaVar& cutVar) ;
  RooDataSet(const char *name, const char *title, TTree *t, 
	     const RooArgSet& vars, const RooFormulaVar& cutVar) ;
  RooDataSet(const char *name, const char *title, TTree *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooDataSet(const char *name, const char *filename, const char *treename, 
	     const RooArgSet& vars, const char *cuts);  
  RooDataSet(RooDataSet const & other, const char* newname=0) ;
  virtual TObject* Clone(const char* newname=0) const { return new RooDataSet(*this,newname?newname:GetName()) ; }
  virtual ~RooDataSet() ;

  // Read data from a text file and create a dataset from it.
  // The possible options are: (D)ebug, (Q)uiet.
  static RooDataSet *read(const char *filename, RooArgList &variables,
			  const char *opts= "", const char* commonPath="",
			  const char *indexCatName=0) ;
  Bool_t write(const char* filename) ;

  virtual Double_t weight() const { return 1.0 ; } ; 

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1.0);
  void append(RooDataSet& data) ;
  Bool_t merge(RooDataSet* data1, RooDataSet* data2=0, RooDataSet* data3=0, 
	       RooDataSet* data4=0, RooDataSet* data5=0, RooDataSet* data6=0) ;

  // Plot the distribution of a real valued arg
  TH2F* createHistogram(const RooAbsReal& var1, const RooAbsReal& var2, const char* cuts="", 
			const char *name= "hist") const;	 
  TH2F* createHistogram(const RooAbsReal& var1, const RooAbsReal& var2, Int_t nx, Int_t ny,
                        const char* cuts="", const char *name="hist") const;

protected:

  friend class RooProdGenContext ;
  Bool_t merge(const TList& data) ;

  // Cache copy feature is not publicly accessible
  RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache=kTRUE) ;
  RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
	     const RooArgSet& vars, const RooFormulaVar* cutVar, Bool_t copyCache);

  ClassDef(RooDataSet,1) // Unbinned data set
};

#endif
