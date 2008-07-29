/* @(#)root/roostats:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__ 

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace NumberCountingUtils;
#pragma link C++ namespace Statistics;

// for auto-loading namespaces
#ifdef USE_FOR_AUTLOADING
#pragma link C++ class NumberCountingUtils;
#pragma link C++ class Statistics;
#endif

#pragma link C++ class SPlot+;

#pragma link C++ function NumberCountingUtils::BinomialExpZ(Double_t,Double_t,Double_t);
#pragma link C++ function NumberCountingUtils::BinomialWithTauExpZ(Double_t,Double_t,Double_t);
#pragma link C++ function NumberCountingUtils::BinomialObsZ(Double_t,Double_t,Double_t);
#pragma link C++ function NumberCountingUtils::BinomialWithTauObsZ(Double_t,Double_t,Double_t);
#pragma link C++ function NumberCountingUtils::BinomialExpP(Double_t,Double_t,Double_t);
#pragma link C++ function NumberCountingUtils::BinomialWithTauExpP(Double_t,Double_t,Double_t);
#pragma link C++ function NumberCountingUtils::BinomialObsP(Double_t,Double_t,Double_t);
#pragma link C++ function NumberCountingUtils::BinomialWithTauObsP(Double_t,Double_t,Double_t);

#pragma link C++ function NumberCountingUtils::ProfileCombinationExpZ(Double_t*,Double_t*,Double_t*,Int_t);

#pragma link C++ function Statistics::PValueToSignificance(Double_t);

#endif
