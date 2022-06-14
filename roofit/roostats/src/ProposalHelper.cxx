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

/** \class RooStats::ProposalHelper
    \ingroup Roostats
*/

#include "Rtypes.h"
#include "RooStats/ProposalHelper.h"
#include "RooStats/PdfProposal.h"
#include "RooStats/RooStatsUtils.h"
#include "RooArgSet.h"
#include "RooDataSet.h"
#include "RooAbsPdf.h"
#include "RooAddPdf.h"
#include "RooNDKeysPdf.h"
#include "RooUniform.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "TIterator.h"
#include "RooMultiVarGaussian.h"
#include "RooConstVar.h"
#include "TString.h"

#include <map>

namespace RooStats {
   class ProposalFunction;
}

ClassImp(RooStats::ProposalHelper);

using namespace RooFit;
using namespace RooStats;
using namespace std;

//static const double DEFAULT_UNI_FRAC = 0.10;
static const double DEFAULT_CLUES_FRAC = 0.20;
//static const double SIGMA_RANGE_DIVISOR = 6;
static const double SIGMA_RANGE_DIVISOR = 5;
//static const Int_t DEFAULT_CACHE_SIZE = 100;
//static const Option_t* CLUES_OPTIONS = "a";

////////////////////////////////////////////////////////////////////////////////

ProposalHelper::ProposalHelper()
{
   fPdfProp = new PdfProposal();
   fVars = nullptr;
   fOwnsPdfProp = true;
   fOwnsPdf = false;
   fOwnsCluesPdf = false;
   fOwnsVars = false;
   fUseUpdates = false;
   fPdf = nullptr;
   fSigmaRangeDivisor = SIGMA_RANGE_DIVISOR;
   fCluesPdf = nullptr;
   fUniformPdf = nullptr;
   fClues = nullptr;
   fCovMatrix = nullptr;
   fCluesFrac = -1;
   fUniFrac = -1;
   fCacheSize = -1;
   fCluesOptions = nullptr;
}

////////////////////////////////////////////////////////////////////////////////

ProposalFunction* ProposalHelper::GetProposalFunction()
{
   if (fPdf == nullptr)
      CreatePdf();
   // kbelasco: check here for memory leaks: does RooAddPdf make copies or
   // take ownership of components, coeffs
   RooArgList* components = new RooArgList();
   RooArgList* coeffs = new RooArgList();
   if (fCluesPdf == nullptr)
      CreateCluesPdf();
   if (fCluesPdf != nullptr) {
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
   fPdfProp->SetOwnsPdf(true);
   if (fCacheSize > 0)
      fPdfProp->SetCacheSize(fCacheSize);
   fOwnsPdfProp = false;
   return fPdfProp;
}

////////////////////////////////////////////////////////////////////////////////

void ProposalHelper::CreatePdf()
{
   // kbelasco: check here for memory leaks:
   // does RooMultiVarGaussian make copies of xVec and muVec?
   // or should we delete them?
   if (fVars == nullptr) {
      coutE(InputArguments) << "ProposalHelper::CreatePdf(): " <<
         "Variables to create proposal function for are not set." << endl;
      return;
   }
   RooArgList* xVec = new RooArgList();
   RooArgList* muVec = new RooArgList();
   RooRealVar* clone; 
   for (auto *r : static_range_cast<RooRealVar *> (*fVars)){
      xVec->add(*r);
      TString cloneName = TString::Format("%s%s", "mu__", r->GetName());
      clone = static_cast<RooRealVar*>(r->clone(cloneName.Data()));
      muVec->add(*clone);
      if (fUseUpdates)
         fPdfProp->AddMapping(*clone, *r);
   }
   if (fCovMatrix == nullptr)
      CreateCovMatrix(*xVec);
   fPdf = new RooMultiVarGaussian("mvg", "MVG Proposal", *xVec, *muVec,
                                  *fCovMatrix);
   delete xVec;
   delete muVec;
}

////////////////////////////////////////////////////////////////////////////////

void ProposalHelper::CreateCovMatrix(RooArgList& xVec)
{
   Int_t size = xVec.getSize();
   fCovMatrix = new TMatrixDSym(size);
   RooRealVar* r;
   for (Int_t i = 0; i < size; i++) {
      r = (RooRealVar*)xVec.at(i);
      double range = r->getMax() - r->getMin();
      (*fCovMatrix)(i,i) = range / fSigmaRangeDivisor;
   }
}

////////////////////////////////////////////////////////////////////////////////

void ProposalHelper::CreateCluesPdf()
{
   if (fClues != nullptr) {
      if (fCluesOptions == nullptr)
         fCluesPdf = new RooNDKeysPdf("cluesPdf", "Clues PDF", *fVars, *fClues);
      else
         fCluesPdf = new RooNDKeysPdf("cluesPdf", "Clues PDF", *fVars, *fClues,
               fCluesOptions);
   }
}

////////////////////////////////////////////////////////////////////////////////

void ProposalHelper::CreateUniformPdf()
{
   fUniformPdf = new RooUniform("uniform", "Uniform Proposal PDF",
         RooArgSet(*fVars));
}
