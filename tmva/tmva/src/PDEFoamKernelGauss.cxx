// @(#)root/tmva $Id$
// Author: Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamKernelGauss                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of gauss PDEFoam kernel                                    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *                                                                                *
 * Copyright (c) 2008, 2010:                                                      *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::PDEFoamKernelGauss
\ingroup TMVA
This PDEFoam kernel estimates a cell value for a given event by
weighting all cell values with a gauss function.
*/

#include "TMVA/PDEFoamKernelGauss.h"

#include "TMVA/MsgLogger.h"
#include "TMVA/PDEFoam.h"
#include "TMVA/PDEFoamKernelBase.h"
#include "TMVA/Types.h"

#include "TMath.h"

#include "Rtypes.h"

ClassImp(TMVA::PDEFoamKernelGauss);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for streamer

TMVA::PDEFoamKernelGauss::PDEFoamKernelGauss(Float_t sigma)
: PDEFoamKernelBase()
   , fSigma(sigma)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamKernelGauss::PDEFoamKernelGauss(const PDEFoamKernelGauss &other)
   : PDEFoamKernelBase(other)
   , fSigma(other.fSigma)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Gaussian kernel estimator.  It returns the cell value 'cv',
/// corresponding to the event vector 'txvec' (in foam coordinates)
/// weighted by the cell values of all other cells, where the weight
/// is a gaussian function.
///
/// Parameters:
///
/// - foam - the pdefoam to search in
///
/// - txvec - event vector in foam coordinates [0,1]
///
/// - cv - cell value to estimate

Float_t TMVA::PDEFoamKernelGauss::Estimate(PDEFoam *foam, std::vector<Float_t> &txvec, ECellValue cv)
{
   if (foam == NULL)
      Log() << kFATAL << "<PDEFoamKernelGauss::Estimate>: PDEFoam not set!" << Endl;

   Float_t result = 0, norm = 0;

   for (Long_t iCell = 0; iCell <= foam->fLastCe; iCell++) {
      if (!(foam->fCells[iCell]->GetStat())) continue;

      // calc cell density
      Float_t cell_val = 0;
      if (!foam->CellValueIsUndefined(foam->fCells[iCell]))
         // cell is not empty
         cell_val = foam->GetCellValue(foam->fCells[iCell], cv);
      else
         // cell is empty -> calc average target of neighbor cells
         cell_val = GetAverageNeighborsValue(foam, txvec, cv);

      // calculate gaussian weight between txvec and fCells[iCell]
      Float_t gau = WeightGaus(foam, foam->fCells[iCell], txvec);

      result += gau * cell_val;
      norm   += gau;
   }

   return (norm != 0 ? result / norm : 0);
}

////////////////////////////////////////////////////////////////////////////////
/// This function returns the average value 'cv' of only nearest
/// neighbor cells.  It is used in cases when a cell value is
/// undefined and the cell value shall be estimated by the
/// (well-defined) cell values of the neighbor cells.
///
/// Parameters:
/// - foam - the foam to search in
/// - txvec - event vector, transformed into foam coordinates [0, 1]
/// - cv - cell value, see definition of ECellValue

Float_t TMVA::PDEFoamKernelGauss::GetAverageNeighborsValue(PDEFoam *foam,
                                                           std::vector<Float_t> &txvec,
                                                           ECellValue cv)
{
   const Float_t xoffset = 1.e-6;
   Float_t norm   = 0; // normalisation
   Float_t result = 0; // return value

   PDEFoamCell *cell = foam->FindCell(txvec); // find corresponding cell
   PDEFoamVect cellSize(foam->GetTotDim());
   PDEFoamVect cellPosi(foam->GetTotDim());
   cell->GetHcub(cellPosi, cellSize); // get cell coordinates

   // loop over all dimensions and find neighbor cells
   for (Int_t dim = 0; dim < foam->GetTotDim(); dim++) {
      std::vector<Float_t> ntxvec(txvec);
      PDEFoamCell* left_cell  = 0; // left cell
      PDEFoamCell* right_cell = 0; // right cell

      // get left cell
      ntxvec[dim] = cellPosi[dim] - xoffset;
      left_cell   = foam->FindCell(ntxvec);
      if (!foam->CellValueIsUndefined(left_cell)) {
         // if left cell is not empty, take its value
         result += foam->GetCellValue(left_cell, cv);
         norm++;
      }
      // get right cell
      ntxvec[dim] = cellPosi[dim] + cellSize[dim] + xoffset;
      right_cell  = foam->FindCell(ntxvec);
      if (!foam->CellValueIsUndefined(right_cell)) {
         // if right cell is not empty, take its value
         result += foam->GetCellValue(right_cell, cv);
         norm++;
      }
   }
   if (norm > 0)  result /= norm; // calc average target
   else         result = 0;     // return null if all neighbors are empty

   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the gauss weight between the 'cell' and a given coordinate 'txvec'.
///
/// Parameters:
/// - cell - the cell
///
/// - txvec - the transformed event variables (in [0,1]) (coordinates <0 are
///   set to 0, >1 are set to 1)
///
/// Returns:
///
/// \f[
/// e^(\frac{-(\frac{d}{\sigma})^2}{2})
/// \f]
///
/// where:
///  - d - is the euclidean distance between 'txvec' and the point of the 'cell'
///    which is most close to 'txvec' (in order to avoid artefacts because of the
///    form of the cells).
///  - \f$ sigma = \frac{1}{VolFrac} \f$

Float_t TMVA::PDEFoamKernelGauss::WeightGaus(PDEFoam *foam, PDEFoamCell* cell,
                                             std::vector<Float_t> &txvec)
{
   // get cell coordinates
   PDEFoamVect cellSize(foam->GetTotDim());
   PDEFoamVect cellPosi(foam->GetTotDim());
   cell->GetHcub(cellPosi, cellSize);

   // calc position of nearest edge of cell
   std::vector<Float_t> cell_center;
   cell_center.reserve(foam->GetTotDim());
   for (Int_t i = 0; i < foam->GetTotDim(); ++i) {
      if (txvec[i] < 0.) txvec[i] = 0.;
      if (txvec[i] > 1.) txvec[i] = 1.;
      //cell_center.push_back(cellPosi[i] + (0.5*cellSize[i]));
      if (cellPosi[i] > txvec.at(i))
         cell_center.push_back(cellPosi[i]);
      else if (cellPosi[i] + cellSize[i] < txvec.at(i))
         cell_center.push_back(cellPosi[i] + cellSize[i]);
      else
         cell_center.push_back(txvec.at(i));
   }

   Float_t distance = 0; // euclidean distance for weighting
   for (Int_t i = 0; i < foam->GetTotDim(); ++i)
      distance += Sqr(txvec.at(i) - cell_center.at(i));
   distance = TMath::Sqrt(distance);

   // weight with Gaus
   return TMath::Gaus(distance, 0, fSigma, kFALSE);
}
