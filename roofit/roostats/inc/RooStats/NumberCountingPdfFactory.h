// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_NumberCountingPdfFactory
#define RooStats_NumberCountingPdfFactory

#include "Rtypes.h"

class RooWorkspace;
class RooRealVar;

namespace RooStats{

   class  NumberCountingPdfFactory {

   public:
      /// need one for expected and one for observed
      NumberCountingPdfFactory();
      virtual ~NumberCountingPdfFactory();

      void AddModel(double* sigExp, Int_t nchan, RooWorkspace* ws,
                    const char* pdfName = "CombinedPdf", const char* masterSignalName = "masterSignal") ;

      void AddData(double* mainMeas, double* bkgMeas, double* db,
                   Int_t nbins, RooWorkspace* ws, const char* dsName = "NumberCountingData");
      void AddExpData(double* sigExp, double* bkgExp, double* db,
                      Int_t nbins, RooWorkspace* ws, const char* dsName = "ExpectedNumberCountingData");
      void AddExpDataWithSideband(double* sigExp, double* bkgExp, double* tau,
                                  Int_t nbins, RooWorkspace* ws, const char* dsName = "NumberCountingData");
      void AddDataWithSideband(double* mainMeas, double* sideband, double* tau,
                               Int_t nbins, RooWorkspace* ws, const char* dsName = "ExpectedNumberCountingData");

   private:
      RooRealVar* SafeObservableCreation(RooWorkspace* ws, const char* varName, double value) ;
      RooRealVar* SafeObservableCreation(RooWorkspace* ws, const char* varName, double value, double maximum) ;


   protected:
      ClassDef(NumberCountingPdfFactory,1) // A factory specific to common number counting problems.

   };
}

#endif
