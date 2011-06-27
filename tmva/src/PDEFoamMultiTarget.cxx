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
TMVA::PDEFoamMultiTarget::PDEFoamMultiTarget(const TString& Name, ETargetSelection ts)
   : PDEFoamEvent(Name)
   , fTargetSelection(ts)
{
   // User constructor
   //
   // Parameters:
   //
   // - Name - name of PDEFoam object
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
TMVA::PDEFoamMultiTarget::PDEFoamMultiTarget(const PDEFoamMultiTarget &From)
   : PDEFoamEvent(From)
   , fTargetSelection(kMean)
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

   // map of targets and normalization
   std::map<Int_t, Float_t> target, norm;
   Double_t max_dens = 0.;            // maximum cell density

   // find cells, which fit txvec
   std::vector<PDEFoamCell*> cells = FindCells(txvec);
   if (cells.size() < 1) {
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

   // loop over all cells that were found
   for (std::vector<PDEFoamCell*>::const_iterator cell_it = cells.begin();
        cell_it != cells.end(); cell_it++) {

      // get event density in cell
      Double_t cell_density = GetCellValue(*cell_it, kValueDensity);

      // get cell position and size
      PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
      (*cell_it)->GetHcub(cellPosi, cellSize);

      // loop over all target dimensions (= dimensions, that are
      // missing in txvec), in order to calculate target value
      for (Int_t idim = 0; idim < GetTotDim(); ++idim) {
         // is idim a target dimension, i.e. is idim missing in txvec?
         std::map<Int_t, Float_t>::const_iterator txvec_it = txvec.find(idim);
         if (txvec_it == txvec.end()) {
            // idim is missing in txvec --> this is a target
            // dimension!
            switch (fTargetSelection) {
               case kMean:
                  target[idim] += cell_density *
                                  VarTransformInvers(idim, cellPosi[idim] + 0.5 * cellSize[idim]);
                  norm[idim] += cell_density;
                  break;
               case kMpv:
                  if (cell_density > max_dens) {
                     max_dens = cell_density; // save new max density
                     target[idim] =
                        VarTransformInvers(idim, cellPosi[idim] + 0.5 * cellSize[idim]);
                  }
                  break;
               default:
                  Log() << "<PDEFoamMultiTarget::GetCellValue>: "
                        << "unknown target selection type!" << Endl;
                  break;
            }
         }
      } // loop over foam dimensions
   } // loop over cells

   // normalise mean cell density
   if (fTargetSelection == kMean) {
      // loop over all dimensions
      for (Int_t idim = 0; idim < GetTotDim(); ++idim) {
         // is idim in target map?
         std::map<Int_t, Float_t>::const_iterator target_it = target.find(idim);
         if (target_it != target.end()) {
            // idim is in target map! --> Normalize
            if (norm[idim] > std::numeric_limits<float>::epsilon())
               target[idim] /= norm[idim];
            else
               // normalisation factor is too small -> return
               // approximate target value
               target[idim] = (fXmax[idim] - fXmin[idim]) / 2.;
         }
      }
   }

   // copy targets to result vector
   std::vector<Float_t> result;
   result.reserve(target.size());
   for (std::map<Int_t, Float_t>::const_iterator it = target.begin();
        it != target.end(); ++it)
      result.push_back(it->second);

   return result;
}
