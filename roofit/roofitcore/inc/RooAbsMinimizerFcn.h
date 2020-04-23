/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef __ROOFIT_NOROOMINIMIZER

#ifndef ROO_ABS_MINIMIZER_FCN
#define ROO_ABS_MINIMIZER_FCN

#include "Math/IFunction.h"
#include "Fit/ParameterSettings.h"
#include "Fit/FitResult.h"

#include "TMatrixDSym.h"

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooRealVar.h"

#include <iostream>
#include <fstream>

// forward declaration
class RooMinimizer;

class RooAbsMinimizerFcn {

public:
   RooAbsMinimizerFcn(RooArgList paramList, RooMinimizer *context, bool verbose = false);
   RooAbsMinimizerFcn(const RooAbsMinimizerFcn &other);
   virtual ~RooAbsMinimizerFcn();

   // inform Minuit through its parameter_settings vector of RooFit parameter properties
   Bool_t synchronize_parameter_settings(std::vector<ROOT::Fit::ParameterSettings> &parameters, Bool_t optConst, Bool_t verbose);
   // same, but can be overridden to e.g. also include gradient strategy synchronization in subclasses:
   virtual Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters, Bool_t optConst, Bool_t verbose);

   // used for export to RooFitResult from Minimizer:
   RooArgList *GetFloatParamList();
   RooArgList *GetConstParamList();
   RooArgList *GetInitFloatParamList();
   RooArgList *GetInitConstParamList();
   Int_t GetNumInvalidNLL();

   // need access from Minimizer:
   void SetEvalErrorWall(Bool_t flag);
   void SetPrintEvalErrors(Int_t numEvalErrors);
   Double_t &GetMaxFCN();
   Int_t evalCounter() const;
   void zeroEvalCount();
   void SetVerbose(Bool_t flag = kTRUE);

   // put Minuit results back into RooFit objects:
   void BackProp(const ROOT::Fit::FitResult &results);

   // set different external covariance matrix
   void ApplyCovarianceMatrix(TMatrixDSym &V);

   Bool_t SetLogFile(const char *inLogfile);
   std::ofstream *GetLogFile() { return _logfile; }

   unsigned int get_nDim() const { return _nDim; }

protected:
   // used in BackProp (Minuit results -> RooFit) and ApplyCovarianceMatrix
   void SetPdfParamErr(Int_t index, Double_t value);
   void ClearPdfParamAsymErr(Int_t index);
   void SetPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal);

   inline Bool_t SetPdfParamVal(const Int_t &index, const Double_t &value) const
   {
      RooRealVar *par = (RooRealVar *)_floatParamVec[index];

      if (par->getVal() != value) {
         if (_verbose)
            std::cout << par->GetName() << "=" << value << ", ";

         par->setVal(value);
         return kTRUE;
      }

      return kFALSE;
   }

   // doesn't seem to be used, TODO: make sure
//   Double_t GetPdfParamVal(Int_t index);
//   Double_t GetPdfParamErr(Int_t index);

   void updateFloatVec();

   virtual void optimizeConstantTerms(bool constStatChange, bool constValChange) = 0;

   // members
   RooMinimizer *_context;

   // the following four are mutable because DoEval is const (in child classes)
   mutable Int_t _evalCounter = 0;
   // Reset the *largest* negative log-likelihood value we have seen so far:
   mutable double _maxFCN = -1e30;
   mutable int _numBadNLL = 0;
   mutable int _printEvalErrors = 10;

   Bool_t _doEvalErrorWall = kTRUE;
   unsigned int _nDim = 0;

   std::ofstream *_logfile = nullptr;
   bool _verbose;

   RooArgList *_floatParamList;
   std::vector<RooAbsArg *> _floatParamVec;
   RooArgList *_constParamList;
   RooArgList *_initFloatParamList;
   RooArgList *_initConstParamList;
};

#endif
#endif
