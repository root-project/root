// @(#)root/tmva $Id$
// Author: Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDecisionTree                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of decision tree like PDEFoam                              *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *      Peter Speckmayer - CERN, Switzerland                                      *
 *                                                                                *
 * Copyright (c) 2010:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_____________________________________________________________________
//
// PDEFoamDecisionTree
//
// This PDEFoam variant acts like a decision tree and stores in every
// cell the discriminant
//
//    D = #events with given class / total number of events
//
// as well as the statistical error on the discriminant.  It therefore
// acts as a discriminant estimator.  The decision tree-like behaviour
// is achieved by overriding PDEFoamDiscriminant::Explore() to use a
// decision tree-like cell splitting algorithm (given a separation
// type).
//
// This PDEFoam variant should be booked together with the
// PDEFoamDecisionTreeDensity density estimator, which returns the
// events in a cell without sampling.
//
//_____________________________________________________________________

#ifndef ROOT_TMVA_PDEFoamDecisionTree
#include "TMVA/PDEFoamDecisionTree.h"
#endif
#ifndef ROOT_TMVA_PDEFoamDecisionTreeDensity
#include "TMVA/PDEFoamDecisionTreeDensity.h"
#endif

ClassImp(TMVA::PDEFoamDecisionTree)

//_____________________________________________________________________
TMVA::PDEFoamDecisionTree::PDEFoamDecisionTree()
   : PDEFoamDiscriminant()
   , fSepType(NULL)
{
   // Default constructor for streamer, user should not use it.
}

//_____________________________________________________________________
TMVA::PDEFoamDecisionTree::PDEFoamDecisionTree(const TString& name, SeparationBase *sepType, UInt_t cls)
   : PDEFoamDiscriminant(name, cls)
   , fSepType(sepType)
{
   // Parameters:
   //
   // - name - name of the foam
   //
   // - sepType - separation type used for the cell splitting (will be
   //   deleted in the destructor)
   //
   // - cls - class to consider as signal when calcualting the purity
}

//_____________________________________________________________________
TMVA::PDEFoamDecisionTree::PDEFoamDecisionTree(const PDEFoamDecisionTree &from)
   : PDEFoamDiscriminant(from)
   , fSepType(from.fSepType)
{
   // Copy Constructor  NOT IMPLEMENTED (NEVER USED)
   Log() << kFATAL << "COPY CONSTRUCTOR NOT IMPLEMENTED" << Endl;
}

//_____________________________________________________________________
TMVA::PDEFoamDecisionTree::~PDEFoamDecisionTree()
{
   // Destructor
   // deletes fSepType
   if (fSepType)
      delete fSepType;
}

//_____________________________________________________________________
void TMVA::PDEFoamDecisionTree::Explore(PDEFoamCell *cell)
{
   // Internal subprogram used by Create.  It explores newly defined
   // cell with according to the decision tree logic.  The separation
   // set via the 'sepType' option in the constructor.
   //
   // The optimal division point for eventual future cell division is
   // determined/recorded.  Note that links to parents and initial
   // volume = 1/2 parent has to be already defined prior to calling
   // this routine.
   //
   // Note, that according to the decision tree logic, a cell is only
   // split, if the number of (unweighted) events in each dautghter
   // cell is greater than fNmin.

   if (!cell)
      Log() << kFATAL << "<DTExplore> Null pointer given!" << Endl;

   // create edge histograms
   std::vector<TH1D*> hsig, hbkg, hsig_unw, hbkg_unw;
   hsig.reserve(fDim);
   hbkg.reserve(fDim);
   hsig_unw.reserve(fDim);
   hbkg_unw.reserve(fDim);
   for (Int_t idim = 0; idim < fDim; idim++) {
      hsig.push_back(new TH1D(Form("hsig_%i", idim),
                              Form("signal[%i]", idim), fNBin, fXmin[idim], fXmax[idim]));
      hbkg.push_back(new TH1D(Form("hbkg_%i", idim),
                              Form("background[%i]", idim), fNBin, fXmin[idim], fXmax[idim]));
      hsig_unw.push_back(new TH1D(Form("hsig_unw_%i", idim),
                                  Form("signal_unw[%i]", idim), fNBin, fXmin[idim], fXmax[idim]));
      hbkg_unw.push_back(new TH1D(Form("hbkg_unw_%i", idim),
                                  Form("background_unw[%i]", idim), fNBin, fXmin[idim], fXmax[idim]));
   }

   // get cell position and size
   PDEFoamVect  cellSize(GetTotDim()), cellPosi(GetTotDim());
   cell->GetHcub(cellPosi, cellSize);

   // determine lower and upper cell bound
   std::vector<Double_t> lb(GetTotDim()); // lower bound
   std::vector<Double_t> ub(GetTotDim()); // upper bound
   for (Int_t idim = 0; idim < GetTotDim(); idim++) {
      lb[idim] = VarTransformInvers(idim, cellPosi[idim] - std::numeric_limits<float>::epsilon());
      ub[idim] = VarTransformInvers(idim, cellPosi[idim] + cellSize[idim] + std::numeric_limits<float>::epsilon());
   }

   // fDistr must be of type PDEFoamDecisionTreeDensity*
   PDEFoamDecisionTreeDensity *distr = dynamic_cast<PDEFoamDecisionTreeDensity*>(fDistr);
   if (distr == NULL)
      Log() << kFATAL << "<PDEFoamDecisionTree::Explore>: cast failed: "
            << "PDEFoamDensityBase* --> PDEFoamDecisionTreeDensity*" << Endl;

   // create TMVA::Volume object needed for searching within the BST
   TMVA::Volume volume(&lb, &ub);

   // fill the signal and background histograms for the given volume
   distr->FillHistograms(volume, hsig, hbkg, hsig_unw, hbkg_unw);

   // ------ determine the best division edge
   Double_t xBest = 0.5;    // best division point
   Int_t    kBest = -1;     // best split dimension
   Double_t maxGain = -1.0; // maximum gain
   Double_t nTotS = hsig.at(0)->Integral(0, hsig.at(0)->GetNbinsX() + 1);
   Double_t nTotB = hbkg.at(0)->Integral(0, hbkg.at(0)->GetNbinsX() + 1);
   Double_t nTotS_unw = hsig_unw.at(0)->Integral(0, hsig_unw.at(0)->GetNbinsX() + 1);
   Double_t nTotB_unw = hbkg_unw.at(0)->Integral(0, hbkg_unw.at(0)->GetNbinsX() + 1);

   for (Int_t idim = 0; idim < fDim; ++idim) {
      Double_t nSelS = hsig.at(idim)->GetBinContent(0);
      Double_t nSelB = hbkg.at(idim)->GetBinContent(0);
      Double_t nSelS_unw = hsig_unw.at(idim)->GetBinContent(0);
      Double_t nSelB_unw = hbkg_unw.at(idim)->GetBinContent(0);
      for (Int_t jLo = 1; jLo < fNBin; jLo++) {
         nSelS += hsig.at(idim)->GetBinContent(jLo);
         nSelB += hbkg.at(idim)->GetBinContent(jLo);
         nSelS_unw += hsig_unw.at(idim)->GetBinContent(jLo);
         nSelB_unw += hbkg_unw.at(idim)->GetBinContent(jLo);

         // proceed if total number of events in left and right cell
         // is greater than fNmin
         if (!((nSelS_unw + nSelB_unw) >= GetNmin() &&
               (nTotS_unw - nSelS_unw + nTotB_unw - nSelB_unw) >= GetNmin()))
            continue;

         Double_t xLo = 1.0 * jLo / fNBin;

         // calculate separation gain
         Double_t gain = fSepType->GetSeparationGain(nSelS, nSelB, nTotS, nTotB);

         if (gain >= maxGain) {
            maxGain = gain;
            xBest   = xLo;
            kBest   = idim;
         }
      } // jLo
   } // idim

   if (kBest >= fDim || kBest < 0) {
      // No best division edge found!  One must ensure, that this cell
      // is not chosen for splitting in PeekMax().  But since in
      // PeekMax() it is ensured that cell->GetDriv() > epsilon, one
      // should set maxGain to -1.0 (or even 0.0?) here.
      maxGain = -1.0;
   }

   // set cell properties
   cell->SetBest(kBest);
   cell->SetXdiv(xBest);
   if (nTotB + nTotS > 0)
      cell->SetIntg(nTotS / (nTotB + nTotS));
   else
      cell->SetIntg(0.0);
   cell->SetDriv(maxGain);
   cell->CalcVolume();

   // set cell element 0 (total number of events in cell) during
   // build-up
   if (GetNmin() > 0)
      SetCellElement(cell, 0, nTotS + nTotB);

   // clean up
   for (UInt_t ih = 0; ih < hsig.size(); ih++)  delete hsig.at(ih);
   for (UInt_t ih = 0; ih < hbkg.size(); ih++)  delete hbkg.at(ih);
   for (UInt_t ih = 0; ih < hsig_unw.size(); ih++)  delete hsig_unw.at(ih);
   for (UInt_t ih = 0; ih < hbkg_unw.size(); ih++)  delete hbkg_unw.at(ih);
}
