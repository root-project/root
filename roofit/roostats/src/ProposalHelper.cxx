// @(#)root/roostats:$Id$
// Authors: Kevin Belasco        7/22/2009
// Authors: Kyle Cranmer         7/22/2009
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
END_HTML
*/
//_________________________________________________

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef RooStats_ProposalHelper
#include "RooStats/ProposalHelper.h"
#endif
#ifndef ROOSTATS_PdfProposal
#include "RooStats/PdfProposal.h"
#endif
#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif
#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef ROO_DATA_SET
#include "RooDataSet.h"
#endif
#ifndef ROO_ABS_PDF
#include "RooAbsPdf.h"
#endif
#ifndef ROO_ADD_PDF
#include "RooAddPdf.h"
#endif
#ifndef ROO_KEYS_PDF
#include "RooNDKeysPdf.h"
#endif
#ifndef ROO_UNIFORM
#include "RooUniform.h"
#endif
#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif
#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif
#ifndef ROOT_TIterator
#include "TIterator.h"
#endif
#ifndef ROO_MULTI_VAR_GAUSSIAN
#include "RooMultiVarGaussian.h"
#endif
#ifndef ROO_CONST_VAR
#include "RooConstVar.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

#include <map>

ClassImp(RooStats::ProposalHelper);

using namespace RooFit;
using namespace RooStats;
using namespace std;

//static const Double_t DEFAULT_UNI_FRAC = 0.10;
static const Double_t DEFAULT_CLUES_FRAC = 0.20;
//static const Double_t SIGMA_RANGE_DIVISOR = 6;
static const Double_t SIGMA_RANGE_DIVISOR = 5;
//static const Int_t DEFAULT_CACHE_SIZE = 100;
//static const Option_t* CLUES_OPTIONS = "a";

ProposalHelper::ProposalHelper()
{
   fPdfProp = new PdfProposal();
   fVars = NULL;
   fOwnsPdfProp = kTRUE;
   fOwnsPdf = kFALSE;
   fOwnsCluesPdf = kFALSE;
   fOwnsVars = kFALSE;
   fUseUpdates = kFALSE;
   fPdf = NULL;
   fSigmaRangeDivisor = SIGMA_RANGE_DIVISOR;
   fCluesPdf = NULL;
   fUniformPdf = NULL;
   fClues = NULL;
   fCovMatrix = NULL;
   fCluesFrac = -1;
   fUniFrac = -1;
   fCacheSize = -1;
   fCluesOptions = NULL;
}

ProposalFunction* ProposalHelper::GetProposalFunction()
{
   if (fPdf == NULL)
      CreatePdf();
   // kbelasco: check here for memory leaks: does RooAddPdf make copies or
   // take ownership of components, coeffs
   RooArgList* components = new RooArgList();
   RooArgList* coeffs = new RooArgList();
   if (fCluesPdf == NULL)
      CreateCluesPdf();
   if (fCluesPdf != NULL) {
      if (fCluesFrac < 0)
         fCluesFrac = DEFAULT_CLUES_FRAC;
      printf("added clues from dataset %s with fraction %g\n",
            fClues->GetName(), fCluesFrac);
      components->add(*fCluesPdf);
      coeffs->add(RooConst(fCluesFrac));
   }
   if (fUniFrac > 0.) {
      CreateUniformPdf();
      components->add(*fUniformPdf);
      coeffs->add(RooConst(fUniFrac));
   }
   components->add(*fPdf);
   RooAddPdf* addPdf = new RooAddPdf("proposalFunction", "Proposal Density",
         *components, *coeffs);
   fPdfProp->SetPdf(*addPdf);
   fPdfProp->SetOwnsPdf(kTRUE);
   if (fCacheSize > 0)
      fPdfProp->SetCacheSize(fCacheSize);
   fOwnsPdfProp = kFALSE;
   return fPdfProp;
}

void ProposalHelper::CreatePdf()
{
   // kbelasco: check here for memory leaks:
   // does RooMultiVarGaussian make copies of xVec and muVec?
   // or should we delete them?
   if (fVars == NULL) {
      coutE(InputArguments) << "ProposalHelper::CreatePdf(): " <<
         "Variables to create proposal function for are not set." << endl;
      return;
   }
   RooArgList* xVec = new RooArgList();
   RooArgList* muVec = new RooArgList();
   TIterator* it = fVars->createIterator();
   RooRealVar* r;
   RooRealVar* clone;
   while ((r = (RooRealVar*)it->Next()) != NULL) {
      xVec->add(*r);
      TString cloneName = TString::Format("%s%s", "mu__", r->GetName());
      clone = (RooRealVar*)r->clone(cloneName.Data());
      muVec->add(*clone);
      if (fUseUpdates)
         fPdfProp->AddMapping(*clone, *r);
   }
   if (fCovMatrix == NULL)
      CreateCovMatrix(*xVec);
   fPdf = new RooMultiVarGaussian("mvg", "MVG Proposal", *xVec, *muVec,
                                  *fCovMatrix);
   delete xVec;
   delete muVec;
   delete it;
}

void ProposalHelper::CreateCovMatrix(RooArgList& xVec)
{
   Int_t size = xVec.getSize();
   fCovMatrix = new TMatrixDSym(size);
   RooRealVar* r;
   for (Int_t i = 0; i < size; i++) {
      r = (RooRealVar*)xVec.at(i);
      Double_t range = r->getMax() - r->getMin();
      (*fCovMatrix)(i,i) = range / fSigmaRangeDivisor;
   }
}

void ProposalHelper::CreateCluesPdf()
{
   if (fClues != NULL) {
      if (fCluesOptions == NULL)
         fCluesPdf = new RooNDKeysPdf("cluesPdf", "Clues PDF", *fVars, *fClues);
      else
         fCluesPdf = new RooNDKeysPdf("cluesPdf", "Clues PDF", *fVars, *fClues,
               fCluesOptions);
   }
}

void ProposalHelper::CreateUniformPdf()
{
   fUniformPdf = new RooUniform("uniform", "Uniform Proposal PDF",
         RooArgSet(*fVars));
}
