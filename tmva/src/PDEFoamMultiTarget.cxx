// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamMultiTarget                                                    *
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
// PDEFoamMultiTarget
//
// This PDEFoam variant is used to estimate multiple targets by
// creating an event density foam (PDEFoamEvent), which has dimension:
//
//    dimension = number of variables + number targets
//
// This PDEFoam variant stores in every cell the sum of event weights
// and the sum of the squared event weights.  During evaluation for a
// given event, which has only variables and no targets (number of
// event variables is smaller than the foam dimension), the targets
// are estimated by finding all cells, which correspond to this event
// and calculate the Mean (or Mpv, depending on the ETargetSelection)
// cell center weighted by the event density in the cell.
//
// This PDEFoam variant should be booked together with the
// PDEFoamEventDensity density estimator, which returns the event
// weight density at a given phase space point during the foam
// build-up.
//
//_____________________________________________________________________

#ifndef ROOT_TMVA_PDEFoamMultiTarget
#include "TMVA/PDEFoamMultiTarget.h"
#endif

ClassImp(TMVA::PDEFoamMultiTarget)

//_____________________________________________________________________
TMVA::PDEFoamMultiTarget::PDEFoamMultiTarget()
   : PDEFoamEvent()
   , fTargetSelection(kMean)
{
   // Default constructor for streamer, user should not use it.
}

//_____________________________________________________________________
TMVA::PDEFoamMultiTarget::PDEFoamMultiTarget(const TString& name, ETargetSelection ts)
   : PDEFoamEvent(name)
   , fTargetSelection(ts)
{
   // User constructor
   //
   // Parameters:
   //
   // - name - name of PDEFoam object
   //
   // - ts - target selection method used in
   //   GetCellValue(const std::map<Int_t, Float_t>& xvec, ECellValue)
   //   Cadidates are: TMVA::kMean, TMVA::kMpv
   //
   //   - TMVA::kMean - The function GetCellValue() finds all cells
   //     which contain a given event vector 'xvec' and returns the
   //     mean target (for every target variable in the foam).
   //
   //   - TMVA::kMpv - The function GetCellValue() finds all cells
   //     which contain a given event vector 'xvec' and returns the
   //     most probable target (for every target variable in the
   //     foam), that is the target value which corresponds to the
   //     cell with the largest event density.
}

//_____________________________________________________________________
TMVA::PDEFoamMultiTarget::PDEFoamMultiTarget(const PDEFoamMultiTarget &from)
   : PDEFoamEvent(from)
   , fTargetSelection(from.fTargetSelection)
{
   // Copy Constructor  NOT IMPLEMENTED (NEVER USED)
   Log() << kFATAL << "COPY CONSTRUCTOR NOT IMPLEMENTED" << Endl;
}

//_____________________________________________________________________
std::vector<Float_t> TMVA::PDEFoamMultiTarget::GetCellValue(const std::map<Int_t, Float_t>& xvec, ECellValue /*cv*/)
{
   // This function is overridden from PDFEFoam.  It returns all
   // regression targets (in order), given an untransformed event
   // vector 'xvec'.  The key of 'xvec' is the dimension and the value
   // (Float_t) is the coordinate.
   //
   // Note: number of foam dimensions = number of variables + number
   // of targets
   //
   // Parameters:
   // - xvec - map of event variables (no targets!)
   // - cv - cell value to return (ignored!)
   //
   // Return:
   // Targets, ordered by missing dimensions in 'xvec'.
   // The size of the returned vector = foam dimension - size of xvec.

   // transform event vector
   std::map<Int_t, Float_t> txvec; // transformed event vector
   for (std::map<Int_t, Float_t>::const_iterator it = xvec.begin();
        it != xvec.end(); ++it) {
      Float_t coordinate = it->second; // event coordinate
      Int_t dim = it->first;           // dimension
      // checkt whether coordinate is within foam borders. if not,
      // push event coordinate into foam
      if (coordinate <= fXmin[dim])
         coordinate = fXmin[dim] + std::numeric_limits<float>::epsilon();
      else if (coordinate >= fXmax[dim])
         coordinate = fXmax[dim] - std::numeric_limits<float>::epsilon();
      // transform event
      txvec.insert(std::pair<Int_t, Float_t>(dim, VarTransform(dim, coordinate)));
   }

   // map of targets
   std::map<Int_t, Float_t> target;

   // find cells, which fit txvec
   std::vector<PDEFoamCell*> cells = FindCells(txvec);
   if (cells.empty()) {
      // return empty target vector (size = dimension of foam -
      // number of variables)
      return std::vector<Float_t>(GetTotDim() - xvec.size(), 0);
   }

   // initialize the target map
   for (Int_t idim = 0; idim < GetTotDim(); ++idim) {
      // is idim a target dimension, i.e. is idim missing in txvec?
      if (txvec.find(idim) == txvec.end())
         target.insert(std::pair<Int_t, Float_t>(idim, 0));
   }

   switch (fTargetSelection) {
   case kMean:
      CalculateMean(target, cells);
      break;
   case kMpv:
      CalculateMpv(target, cells);
      break;
   default:
      Log() << "<PDEFoamMultiTarget::GetCellValue>: "
            << "unknown target selection type!" << Endl;
      break;
   }

   // copy targets to result vector
   std::vector<Float_t> result;
   result.reserve(target.size());
   for (std::map<Int_t, Float_t>::const_iterator it = target.begin();
        it != target.end(); ++it)
      result.push_back(it->second);

   return result;
}

void TMVA::PDEFoamMultiTarget::CalculateMpv(std::map<Int_t, Float_t>& target, const std::vector<PDEFoamCell*>& cells)
{
   // This function calculates the most probable target value from a
   // given number of cells.  The most probable target is defined to
   // be the coordinates of the cell which has the biggest event
   // density.
   //
   // Parameters:
   //
   // - target - map of targets, where the key is the dimension and
   //   the value is the target value.  It is assumed that this map is
   //   initialized such that there is a map entry for every target.
   //
   // - cells - vector of PDEFoam cells to pick the most probable
   //   target from

   Double_t max_dens = 0.0;            // maximum cell density

   // loop over all cells and find cell with maximum event density
   for (std::vector<PDEFoamCell*>::const_iterator cell_it = cells.begin();
        cell_it != cells.end(); ++cell_it) {

      // get event density of cell
      const Double_t cell_density = GetCellValue(*cell_it, kValueDensity);

      // has this cell a larger event density?
      if (cell_density > max_dens) {
         // get cell position and size
         PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
         (*cell_it)->GetHcub(cellPosi, cellSize);

         // save new maximum density
         max_dens = cell_density;

         // calculate new target values
         for (std::map<Int_t, Float_t>::iterator target_it = target.begin();
              target_it != target.end(); ++target_it) {
            const Int_t dim = target_it->first; // target dimension
            target_it->second =
               VarTransformInvers(dim, cellPosi[dim] + 0.5 * cellSize[dim]);
         }
      }
   }
}

void TMVA::PDEFoamMultiTarget::CalculateMean(std::map<Int_t, Float_t>& target, const std::vector<PDEFoamCell*>& cells)
{
   // This function calculates the mean target value from a given
   // number of cells.  As weight the event density of the cell is
   // used.
   //
   // Parameters:
   //
   // - target - map of targets, where the key is the dimension and
   //   the value is the target value.  It is assumed that this map is
   //   initialized such that there is a map entry for every target
   //   with all target values set to zero.
   //
   // - cells - vector of PDEFoam cells to pick the most probable
   //   target from

   // normalization
   std::map<Int_t, Float_t> norm;

   // loop over all cells and find cell with maximum event density
   for (std::vector<PDEFoamCell*>::const_iterator cell_it = cells.begin();
        cell_it != cells.end(); ++cell_it) {

      // get event density of cell
      const Double_t cell_density = GetCellValue(*cell_it, kValueDensity);

      // get cell position and size
      PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
      (*cell_it)->GetHcub(cellPosi, cellSize);

      // accumulate weighted target values
      for (std::map<Int_t, Float_t>::iterator target_it = target.begin();
           target_it != target.end(); ++target_it) {
         const Int_t dim = target_it->first; // target dimension
         target_it->second += cell_density *
            VarTransformInvers(dim, cellPosi[dim] + 0.5 * cellSize[dim]);
         norm[dim] += cell_density;
      }
   }

   // normalize the targets
   for (std::map<Int_t, Float_t>::iterator target_it = target.begin();
        target_it != target.end(); ++target_it) {

      // get target dimension
      const Int_t dim = target_it->first;

      // normalize target in dimension 'dim'
      if (norm[dim] > std::numeric_limits<Float_t>::epsilon()) {
         target[dim] /= norm[dim];
      } else {
         // normalisation factor is too small -> return approximate
         // target value
         target[dim] = (fXmax[dim] - fXmin[dim]) / 2.;
      }
   }
}
