// @(#)root/tmva $Id$
// Author: Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamKernelLinN                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of linear neighbors PDEFoam kernel                         *
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

/*! \class TMVA::PDEFoamKernelLinN
\ingroup TMVA
This PDEFoam kernel estimates a cell value for a given event by
weighting with cell values of the nearest neighbor cells.
*/

#include "TMVA/PDEFoamKernelLinN.h"

#include "TMVA/PDEFoam.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/PDEFoamKernelBase.h"
#include "TMVA/Types.h"

#include "Rtypes.h"

#include <vector>

ClassImp(TMVA::PDEFoamKernelLinN);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for streamer

TMVA::PDEFoamKernelLinN::PDEFoamKernelLinN()
: PDEFoamKernelBase()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamKernelLinN::PDEFoamKernelLinN(const PDEFoamKernelLinN &other)
   : PDEFoamKernelBase(other)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Linear neighbors kernel estimator.  It returns the cell value
/// 'cv', corresponding to the event vector 'txvec' (in foam
/// coordinates) linear weighted by the cell values of the neighbor
/// cells.
///
/// Parameters:
///
/// - foam - the pdefoam to search in
///
/// - txvec - event vector in foam coordinates [0,1]
///
/// - cv - cell value to estimate

Float_t TMVA::PDEFoamKernelLinN::Estimate(PDEFoam *foam, std::vector<Float_t> &txvec, ECellValue cv)
{
   if (foam == NULL)
      Log() << kFATAL << "<PDEFoamKernelLinN::Estimate>: PDEFoam not set!" << Endl;

   return WeightLinNeighbors(foam, txvec, cv, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the cell value, corresponding to 'txvec' (foam
/// coordinates [0,1]), weighted by the neighbor cells via a linear
/// function.
///
/// Parameters:
///  - foam - the foam to search in
///
///  - txvec - event vector, transformed into foam coordinates [0,1]
///
///  - cv - cell value to be weighted
///
///  - treatEmptyCells - if this option is set to false (default),
///    it is not checked, wether the cell value or neighbor cell
///    values are undefined (using foam->CellValueIsUndefined()).
///    If this option is set to true, than only non-empty neighbor
///    cells are taken into account for weighting.  If the cell
///    value of the cell, which contains txvec, is empty, than its
///    value is estimated by the average value of the non-empty
///    neighbor cells (using GetAverageNeighborsValue()).

Float_t TMVA::PDEFoamKernelLinN::WeightLinNeighbors(PDEFoam *foam, std::vector<Float_t> &txvec, ECellValue cv, Bool_t treatEmptyCells)
{
   Float_t result = 0.;
   UInt_t norm     = 0;
   const Float_t xoffset = 1.e-6;

   if (txvec.size() != UInt_t(foam->GetTotDim()))
      Log() << kFATAL << "Wrong dimension of event variable!" << Endl;

   // find cell, which contains txvec
   PDEFoamCell *cell = foam->FindCell(txvec);
   PDEFoamVect cellSize(foam->GetTotDim());
   PDEFoamVect cellPosi(foam->GetTotDim());
   cell->GetHcub(cellPosi, cellSize);
   // calc value of cell, which contains txvec
   Float_t cellval = 0;
   if (!(treatEmptyCells && foam->CellValueIsUndefined(cell)))
      // cell is not empty -> get cell value
      cellval = foam->GetCellValue(cell, cv);
   else
      // cell is empty -> get average value of non-empty neighbor
      // cells
      cellval = GetAverageNeighborsValue(foam, txvec, cv);

   // loop over all dimensions to find neighbor cells
   for (Int_t dim = 0; dim < foam->GetTotDim(); dim++) {
      std::vector<Float_t> ntxvec(txvec);
      Float_t mindist;
      PDEFoamCell *mindistcell = 0; // cell with minimal distance to txvec
      // calc minimal distance to neighbor cell
      mindist = (txvec[dim] - cellPosi[dim]) / cellSize[dim];
      if (mindist < 0.5) { // left neighbour
         ntxvec[dim] = cellPosi[dim] - xoffset;
         mindistcell = foam->FindCell(ntxvec); // left neighbor cell
      } else { // right neighbour
         mindist = 1 - mindist;
         ntxvec[dim] = cellPosi[dim] + cellSize[dim] + xoffset;
         mindistcell = foam->FindCell(ntxvec); // right neighbor cell
      }
      // get cell value of cell, which contains ntxvec
      Float_t mindistcellval = foam->GetCellValue(mindistcell, cv);
      // if treatment of empty neighbor cells is deactivated, do
      // normal weighting
      if (!(treatEmptyCells && foam->CellValueIsUndefined(mindistcell))) {
         result += cellval        * (0.5 + mindist);
         result += mindistcellval * (0.5 - mindist);
         norm++;
      }
   }
   if (norm == 0) return cellval;   // all nearest neighbors were empty
   else         return result / norm; // normalisation
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

Float_t TMVA::PDEFoamKernelLinN::GetAverageNeighborsValue(PDEFoam *foam,
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
