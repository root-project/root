// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
BernsteinCorrection is a utility in RooStats to augment a nominal PDF with a polynomial 
correction term.  This is useful for incorporating systematic variations to the nominal PDF.  
The Bernstein basis polynomails are particularly appropriate because they are positive definite. 
</p>
<p>
This tool was inspired by the work of Glen Cowan together with Stephan Horner, Sascha Caron, 
Eilam Gross, and others.  
The initial implementation is independent work.  The major step forward in the approach was 
to provide a well defined algorithm that specifies the order of polynomial to be included 
in the correction.  This is an emperical algorithm, so in addition to the nominal model it 
needs either a real data set or a simulated one.  In the early work, the nominal model was taken
to be a histogram from Monte Carlo simulations, but in this implementation it is generalized to an
arbitrary PDF (which includes a RooHistPdf).  The algorithm basically consists of a 
hypothesis test of an nth-order correction (null) against a n+1-th order correction (alternate). 
The quantity q = -2 log LR is used to determine whether the n+1-th order correction is a major 
improvement to the n-th order correction.  The distribution of q is expected to be roughly 
\chi^2 with one degree of freedom if the n-th order correction is a good model for the data. 
 Thus, one only moves to the n+1-th order correction of q is relatively large.  The chance that 
one moves from the n-th to the n+1-th order correction when the n-th order correction 
(eg. a type 1 error) is sufficient is given by the Prob(\chi^2_1 > threshold).  The constructor 
of this class allows you to directly set this tolerance (in terms of probability that the n+1-th
 term is added unnecessarily).
</p>
<p>
To do:
Add another method to the utility that will make the sampling distribution for -2 log lambda 
for various m vs. m+1 order corrections using a nominal model and perhaps having two ways of 
generating the toys (either via a histogram or via an independent model that is supposed to
 reflect reality).  That will allow one to make plots like Glen has at the end of his DRAFT
 very easily. 
</p>
END_HTML
*/
//


#ifndef RooStats_BernsteinCorrection
#include "RooStats/BernsteinCorrection.h"
#endif 

#include "RooGlobalFunc.h"
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooAbsPdf.h"
#include "RooFit.h"
#include "RooFitResult.h"
#include "TMath.h"
#include <string>
#include <vector>
#include <stdio.h>
#include <sstream>
#include <iostream>

#include "RooProdPdf.h"
#include "RooNLLVar.h"
#include "RooWorkspace.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

#include "RooBernstein.h"


ClassImp(RooStats::BernsteinCorrection) ;

using namespace RooFit;
using namespace RooStats;

//____________________________________
BernsteinCorrection::BernsteinCorrection(Double_t tolerance):
  fMaxCorrection(100), fTolerance(tolerance){
}


//____________________________________
Int_t BernsteinCorrection::ImportCorrectedPdf(RooWorkspace* wks, 
					      const char* nominalName, 
					      const char* varName, 
					      const char* dataName){
  // Main method for Bernstein correction.

  // get ingredients out of workspace
  RooRealVar* x = wks->var(varName);
  RooAbsPdf* nominal = wks->pdf(nominalName);
  RooAbsData* data = wks->data(dataName);

  // initialize alg, by checking how well nominal model fits
  RooFitResult* nominalResult = nominal->fitTo(*data,Save(),Minos(kFALSE), Hesse(kFALSE),PrintLevel(-1));
  Double_t lastNll= nominalResult->minNll();

  // setup a log
  std::stringstream log;
  log << "------ Begin Bernstein Correction Log --------" << endl;

  // Local variables that we want to keep in scope after loop
  RooArgList coeff;
  vector<RooRealVar*> coefficients;
  Double_t q = 1E6;
  Int_t degree = -1;

  // The while loop
  bool keepGoing = true;
  while( keepGoing) {
    degree++;

    // we need to generate names for vars on the fly
    std::stringstream str;
    str<<"_"<<degree;

    RooRealVar* newCoef = new RooRealVar(("c"+str.str()).c_str(),
			"Bernstein basis poly coefficient", 
			1., 0., fMaxCorrection);
    coeff.add(*newCoef);
    coefficients.push_back(newCoef);

    // make the polynomial correction term
    RooBernstein* poly = new RooBernstein("poly", "Bernstein poly", *x, coeff);

    // make the corrected PDF = nominal * poly
    RooProdPdf* corrected = new RooProdPdf("corrected","",RooArgList(*nominal,*poly));

    // check to see how well this correction fits
    RooFitResult* result = corrected->fitTo(*data,Save(),Minos(kFALSE), Hesse(kFALSE),PrintLevel(-1));


    // Hypothesis test between previous correction (null)
    // and this one (alternate).  Use -2 log LR for test statistic
    q = 2*(lastNll - result->minNll()); // -2 log lambda, goes like significance^2
    // check if we should keep going based on rate of Type I error
    keepGoing = (degree < 1 || TMath::Prob(q,1) < fTolerance);

    if(!keepGoing){
      // terminate loop, import corrected PDF
      RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL) ;
      wks->import(*corrected);
      RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
    } else { 
      // memory management
      delete corrected;
      delete poly;
    }

    // for the log
    if(degree != 0){
      log << "degree = " << degree 
	  << " -log L("<<degree-1<<") = " << lastNll 
	  << " -log L(" << degree <<") = " << result->minNll() 
	  << " q = " << q 
	  << " P(chi^2_1 > q) = " << TMath::Prob(q,1) << endl;;
    }

    // update last result for next iteration in loop
    lastNll = result->minNll();

    delete result;
  }

  log << "------ End Bernstein Correction Log --------" << endl;
  cout << log.str();

  return degree;
}


//____________________________________
void BernsteinCorrection::CreateQSamplingDist(RooWorkspace* wks, 
					       const char* nominalName, 
					       const char* varName, 
					       const char* dataName,
					      TH1F* samplingDist,
					      TH1F* samplingDistExtra,
					       Int_t degree, 
					       Int_t nToys){
  // Create sampling distribution for q given degree-1 vs. degree corrections

  // get ingredients out of workspace
  RooRealVar* x = wks->var(varName);
  RooAbsPdf* nominal = wks->pdf(nominalName);
  RooAbsData* data = wks->data(dataName);

  // setup a log
  std::stringstream log;
  log << "------ Begin Bernstein Correction Log --------" << endl;

  // Local variables that we want to keep in scope after loop
  RooArgList coeff; // n-th degree correction
  RooArgList coeffNull; // n-1 correction
  RooArgList coeffExtra; // n+1 correction
  vector<RooRealVar*> coefficients;

  cout << "make coefs" << endl;
  for(int i = 0; i<=degree+1; ++i) {
    // we need to generate names for vars on the fly
    std::stringstream str;
    str<<"_"<<i;

    RooRealVar* newCoef = new RooRealVar(("c"+str.str()).c_str(),
			"Bernstein basis poly coefficient", 
			1., 0., fMaxCorrection);
    
    // keep three sets of coefficients for n-1, n, n+1 corrections
    if(i<degree)  coeffNull.add(*newCoef); 
    if(i<=degree) coeff.add(*newCoef); 
    coeffExtra.add(*newCoef);
    coefficients.push_back(newCoef);
  }

  coeffNull.Print();
  coeff.Print();
  // make the polynomial correction term
  RooBernstein* poly 
    = new RooBernstein("poly", "Bernstein poly", *x, coeff);

  // make the polynomial correction term
  RooBernstein* polyNull 
    = new RooBernstein("polyNull", "Bernstein poly", *x, coeffNull);

  // make the polynomial correction term
  RooBernstein* polyExtra 
    = new RooBernstein("polyExtra", "Bernstein poly", *x, coeffExtra);

  // make the corrected PDF = nominal * poly
  RooProdPdf* corrected 
    = new RooProdPdf("corrected","",RooArgList(*nominal,*poly));

  RooProdPdf* correctedNull
    = new RooProdPdf("correctedNull","",RooArgList(*nominal,*polyNull));

  RooProdPdf* correctedExtra
    = new RooProdPdf("correctedExtra","",RooArgList(*nominal,*polyExtra));


  cout << "made pdfs, make toy generator" << endl;

  // makde a pDF to generate the toys
  RooDataHist dataHist("dataHist","",*x,*data);
  RooHistPdf toyGen("toyGen","",*x,dataHist);


  //  TH1F* samplingDist = new TH1F("samplingDist","",20,0,10);
  Double_t q = 0, qExtra = 0;
  // do toys
  for(int i=0; i<nToys; ++i){
    cout << "on toy " << i << endl;
    RooDataSet* tmpData = toyGen.generate(*x,data->numEntries());
    // check to see how well this correction fits
    RooFitResult* result 
      = corrected->fitTo(*tmpData,Save(),Minos(kFALSE), 
			 Hesse(kFALSE),PrintLevel(-1));

    RooFitResult* resultNull 
      = correctedNull->fitTo(*tmpData,Save(),Minos(kFALSE), 
			 Hesse(kFALSE),PrintLevel(-1));


    RooFitResult* resultExtra
      = correctedExtra->fitTo(*tmpData,Save(),Minos(kFALSE), 
			 Hesse(kFALSE),PrintLevel(-1));


    // Hypothesis test between previous correction (null)
    // and this one (alternate).  Use -2 log LR for test statistic
    q = 2*(resultNull->minNll() - result->minNll()); 

    qExtra = 2*(result->minNll() - resultExtra->minNll()); 

    samplingDist->Fill(q);
    samplingDistExtra->Fill(qExtra);
    cout << resultNull->minNll() << " " << result->minNll() << " " << q << endl;
    
    delete tmpData;
    delete result;
    delete resultNull;
    delete resultExtra;
  }

  //  return samplingDist;
}
