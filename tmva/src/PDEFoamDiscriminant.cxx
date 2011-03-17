// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDiscriminant                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation.                                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *      Peter Speckmayer - CERN, Switzerland                                      *
 *                                                                                *
 * Copyright (c) 2008, 2010:                                                      *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_____________________________________________________________________
//
// PDEFoamDiscriminant
//
// This PDEFoam variant stores in every cell the discriminant
//
//    D = #events with given class / total number of events
//
// as well as the statistical error on the discriminant.  It therefore
// acts as a discriminant estimator.  It should be booked together
// with the PDEFoamDiscriminantDensity density estimator, which
// returns the discriminant density at a given phase space point
// during the foam build-up.
//
//_____________________________________________________________________

#include <climits>

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#ifndef ROOT_TMVA_PDEFoamDiscriminant
#include "TMVA/PDEFoamDiscriminant.h"
#endif

ClassImp(TMVA::PDEFoamDiscriminant)

//_____________________________________________________________________
TMVA::PDEFoamDiscriminant::PDEFoamDiscriminant()
   : PDEFoam()
   , fClass(0)
{
   // Default constructor for streamer, user should not use it.
}

//_____________________________________________________________________
TMVA::PDEFoamDiscriminant::PDEFoamDiscriminant(const TString& Name, UInt_t cls)
   : PDEFoam(Name)
   , fClass(cls)
{}

//_____________________________________________________________________
TMVA::PDEFoamDiscriminant::PDEFoamDiscriminant(const PDEFoamDiscriminant &From)
   : PDEFoam(From)
   , fClass(0)
{
   // Copy Constructor  NOT IMPLEMENTED (NEVER USED)
   Log() << kFATAL << "COPY CONSTRUCTOR NOT IMPLEMENTED" << Endl;
}

//_____________________________________________________________________
void TMVA::PDEFoamDiscriminant::FillFoamCells(const Event* ev, Float_t wt)
{
   // This function fills an event into the discriminant PDEFoam.  The
   // event weight 'wt' is filled into cell element 0 if the event is
   // of class fClass, and filled into cell element 1 otherwise.

   // find corresponding foam cell
   std::vector<Float_t> values  = ev->GetValues();
   std::vector<Float_t> tvalues = VarTransform(values);
   PDEFoamCell *cell = FindCell(tvalues);

   // 0. Element: Number of signal events (even class == fClass)
   // 1. Element: Number of background events times normalization
   if (ev->GetClass() == fClass)
      SetCellElement(cell, 0, GetCellElement(cell, 0) + wt);
   else
      SetCellElement(cell, 1, GetCellElement(cell, 1) + wt);
}

//_____________________________________________________________________
void TMVA::PDEFoamDiscriminant::Finalize()
{
   // Calc discriminator and its error for every cell and save it to
   // the cell.

   // loop over cells
   for (Long_t iCell = 0; iCell <= fLastCe; iCell++) {
      if (!(fCells[iCell]->GetStat()))
         continue;

      Double_t N_sig = GetCellElement(fCells[iCell], 0); // get number of signal events
      Double_t N_bg  = GetCellElement(fCells[iCell], 1); // get number of bg events

      if (N_sig < 0.) {
         Log() << kWARNING << "Negative number of signal events in cell " << iCell
               << ": " << N_sig << ". Set to 0." << Endl;
         N_sig = 0.;
      }
      if (N_bg < 0.) {
         Log() << kWARNING << "Negative number of background events in cell " << iCell
               << ": " << N_bg << ". Set to 0." << Endl;
         N_bg = 0.;
      }

      // calculate discriminant
      if (N_sig + N_bg > 0) {
         // discriminant
         SetCellElement(fCells[iCell], 0, N_sig / (N_sig + N_bg));
         // discriminant error
         SetCellElement(fCells[iCell], 1, TMath::Sqrt(Sqr(N_sig / Sqr(N_sig + N_bg))*N_sig +
                                                      Sqr(N_bg / Sqr(N_sig + N_bg))*N_bg));

      } else {
         SetCellElement(fCells[iCell], 0, 0.5); // set discriminator
         SetCellElement(fCells[iCell], 1, 1.);  // set discriminator error
      }
   }
}

//_____________________________________________________________________
TH2D* TMVA::PDEFoamDiscriminant::Project2(Int_t idim1, Int_t idim2, ECellValue cell_value, PDEFoamKernelBase *kernel, UInt_t nbin)
{
   // Project foam variable idim1 and variable idim2 to histogram.
   // The projection algorithm is modified such that the z axis range
   // of the returned histogram is [0, 1], as necessary for the
   // interpretation as a discriminator.  This is done by weighting
   // the cell values (in case of cell_value = kValue) by the cell
   // volume in all dimensions, excluding 'idim1' and 'idim2'.
   //
   // Parameters:
   //
   // - idim1, idim2 - dimensions to project to
   //
   // - cell_value - the cell value to draw
   //
   // - kernel - a PDEFoam kernel (optional).  If NULL is given, the
   //            kernel is ignored and the pure cell values are
   //            plotted.
   //
   // - nbin - number of bins in x and y direction of result histogram
   //          (optional, default is 50).
   //
   // Returns:
   // a 2-dimensional histogram

   // avoid plotting of wrong dimensions
   if ((idim1 >= GetTotDim()) || (idim1 < 0) ||
       (idim2 >= GetTotDim()) || (idim2 < 0) ||
       (idim1 == idim2))
      Log() << kFATAL << "<Project2>: wrong dimensions given: "
            << idim1 << ", " << idim2 << Endl;

   // root can not handle too many bins in one histogram --> catch this
   // Furthermore, to have more than 1000 bins in the histogram doesn't make
   // sense.
   if (nbin > 1000) {
      Log() << kWARNING << "Warning: number of bins too big: " << nbin
            << " Using 1000 bins for each dimension instead." << Endl;
      nbin = 1000;
   } else if (nbin < 1) {
      Log() << kWARNING << "Wrong bin number: " << nbin
            << "; set nbin=50" << Endl;
      nbin = 50;
   }

   // create result histogram
   TString hname(Form("h_%d_vs_%d", idim1, idim2));

   // if histogram with this name already exists, delete it
   TH2D* h1 = (TH2D*)gDirectory->Get(hname.Data());
   if (h1) delete h1;
   h1 = new TH2D(hname.Data(), Form("var%d vs var%d", idim1, idim2), nbin, fXmin[idim1], fXmax[idim1], nbin, fXmin[idim2], fXmax[idim2]);

   if (!h1) Log() << kFATAL << "ERROR: Can not create histo" << hname << Endl;
   if (cell_value == kValue)
      h1->GetZaxis()->SetRangeUser(-std::numeric_limits<float>::epsilon(),
                                   1. + std::numeric_limits<float>::epsilon());

   // ============== start projection algorithm ================
   // loop over all histogram bins (2-dim)
   for (Int_t xbin = 1; xbin <= h1->GetNbinsX(); ++xbin) {
      for (Int_t ybin = 1; ybin <= h1->GetNbinsY(); ++ybin) {
         // calculate the phase space point, which corresponds to this
         // bin combination
         std::map<Int_t, Float_t> txvec;
         txvec[idim1] = VarTransform(idim1, h1->GetXaxis()->GetBinCenter(xbin));
         txvec[idim2] = VarTransform(idim2, h1->GetYaxis()->GetBinCenter(ybin));

         // find the cells, which corresponds to this phase space
         // point
         std::vector<TMVA::PDEFoamCell*> cells = FindCells(txvec);

         // loop over cells and fill the histogram with the cell
         // values
         Float_t sum_cv = 0; // sum of the cell values
         for (std::vector<TMVA::PDEFoamCell*>::const_iterator it = cells.begin();
              it != cells.end(); ++it) {
            // get cell position and size
            PDEFoamVect cellPosi(GetTotDim()), cellSize(GetTotDim());
            (*it)->GetHcub(cellPosi, cellSize);
            // Create complete event vector from txvec.  The missing
            // coordinates of txvec are set to the cell center.
            std::vector<Float_t> tvec;
            for (Int_t i = 0; i < GetTotDim(); ++i) {
               if (i != idim1 && i != idim2)
                  tvec.push_back(cellPosi[i] + 0.5 * cellSize[i]);
               else
                  tvec.push_back(txvec[i]);
            }
            // get the cell value using the kernel
            Float_t cv = 0;
            if (kernel != NULL) {
               cv = kernel->Estimate(this, tvec, cell_value);
            } else {
               cv = GetCellValue(FindCell(tvec), cell_value);
            }
            if (cell_value == kValue) {
               // calculate cell volume in other dimensions (not
               // including idim1 and idim2)
               Float_t area_cell = 1.;
               for (Int_t d1 = 0; d1 < GetTotDim(); ++d1) {
                  if ((d1 != idim1) && (d1 != idim2))
                     area_cell *= cellSize[d1];
               }
               // calc discriminator * (cell area times foam area)
               // foam is normalized -> length of foam = 1.0
               cv *= area_cell;
            }
            sum_cv += cv;
         }

         // fill the bin content
         h1->SetBinContent(xbin, ybin, sum_cv + h1->GetBinContent(xbin, ybin));
      }
   }

   return h1;
}
