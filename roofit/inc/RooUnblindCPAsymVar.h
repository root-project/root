/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooUnblindCPAsymVar.rdl,v 1.7 2001/05/03 02:15:56 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, University of California Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_UNBLIND_CPASYM_VAR
#define ROO_UNBLIND_CPASYM_VAR

#include "RooFitCore/RooDerivedReal.hh"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitModels/RooBlindTools.hh"

class RooUnblindCPAsymVar : public RooDerivedReal {
public:
  // Constructors, assignment etc
  inline RooUnblindCPAsymVar() ;
  RooUnblindCPAsymVar(const char *name, const char *title, 
			const char *blindString, RooAbsReal& cpasym);
  RooUnblindCPAsymVar(const RooUnblindCPAsymVar& other, const char* name=0);
  virtual TObject* clone() const { return new RooUnblindCPAsymVar(*this); }  
  virtual ~RooUnblindCPAsymVar();

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent="") const ;

protected:

  RooUnblindCPAsymVar& operator=(const RooUnblindCPAsymVar& other) { return *this ; } ;
  
  // Only PDFs can access the unblinded values
  friend class RooAbsPdf ;
  
  // Function evaluation
  virtual Double_t evaluate() const ;

  virtual Bool_t isValid() const ;
  virtual Bool_t isValid(Double_t value, Bool_t verbose=kFALSE) const ;

  RooRealProxy _asym ;
  RooBlindTools _blindEngine ;

  ClassDef(RooUnblindCPAsymVar,1) // a real-valued variable and its value
};

#endif
