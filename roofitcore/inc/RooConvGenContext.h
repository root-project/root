/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_CONV_GEN_CONTEXT
#define ROO_CONV_GEN_CONTEXT

#include "TList.h"
#include "RooFitCore/RooAbsGenContext.hh"
#include "RooFitCore/RooArgSet.hh"

class RooConvolutedPdf;
class RooDataSet;
class RooRealIntegral;
class RooAcceptReject;
class TRandom;
class TIterator;
class RooRealVar ;

class RooConvGenContext : public RooAbsGenContext {
public:
  RooConvGenContext(const RooConvolutedPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		    Bool_t _verbose= kFALSE);
  virtual ~RooConvGenContext();

protected:

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining);

  RooConvGenContext(const RooConvGenContext& other) ;

  RooAbsGenContext* _pdfGen ;   // Physics model generator context
  RooAbsGenContext* _modelGen ; // Resolution model generator context
  TString _convVarName ;        // Name of convolution variable
  RooArgSet* _pdfVars ;         // Holder of PDF x truth event
  RooArgSet* _modelVars ;       // Holder of resModel event
  RooArgSet* _pdfCloneSet ;     // Owner of PDF clone
  RooArgSet* _modelCloneSet ;   // Owner of resModel clone
  RooRealVar* _cvModel ;         // Convolution variable in resModel event
  RooRealVar* _cvPdf ;           // Convolution variable in PDFxTruth event
  RooRealVar* _cvOut ;           // Convolution variable in output event

  ClassDef(RooConvGenContext,0) // Context for generating a dataset from a PDF
};

#endif
