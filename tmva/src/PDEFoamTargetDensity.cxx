// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamTargetDensity                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      The TFDSITR class provides an interface between the Binary search tree    *
 *      and the PDEFoam object.  In order to build-up the foam one needs to       *
 *      calculate the density of events at a given point (sampling during         *
 *      Foam build-up).  The function PDEFoamTargetDensity::Density() does this job.  It  *
 *      uses a binary search tree, filled with training events, in order to       *
 *      provide this density.                                                     *
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
// PDEFoamTargetDensity
//
// This is a concrete implementation of PDEFoam.  Density(...)
// estimates the target density (target number: fTarget) at a given
// phase-space point using range-searching.
// _____________________________________________________________________

#include <cmath>

#ifndef ROOT_TMVA_PDEFoamTargetDensity
#include "TMVA/PDEFoamTargetDensity.h"
#endif

ClassImp(TMVA::PDEFoamTargetDensity)

//_____________________________________________________________________
TMVA::PDEFoamTargetDensity::PDEFoamTargetDensity()
   : PDEFoamDensityBase()
   , fTarget(0)
{}

//_____________________________________________________________________
TMVA::PDEFoamTargetDensity::PDEFoamTargetDensity(std::vector<Double_t> box, UInt_t target)
   : PDEFoamDensityBase(box)
   , fTarget(target)
{
   // User construcor
   //
   // Parameters:
   //
   // - box - size of the range-searching box (n-dimensional
   //   std::vector)
   //
   // - target - the target number to calculate the density for
}

//_____________________________________________________________________
TMVA::PDEFoamTargetDensity::PDEFoamTargetDensity(const PDEFoamTargetDensity &distr)
   : PDEFoamDensityBase(distr)
   , fTarget(distr.fTarget)
{
   // Copy constructor
}

//_____________________________________________________________________
Double_t TMVA::PDEFoamTargetDensity::Density(std::vector<Double_t> &Xarg, Double_t &event_density)
{
   // This function is needed during the foam buildup.  It returns the
   // average target value within the range-searching box at point
   // Xarg, divided by volume (specified by fBox).
   //
   // Parameters:
   //
   // - Xarg - event vector (in [fXmin,fXmax]) to place the box at
   //
   // - event_density - here the event density is stored
   //
   // Returns:
   //
   // Average target value in the range-searching volume at point
   // 'Xarg', divided by the box volume.

   if (!fBst)
      Log() << kFATAL << "<PDEFoamTargetDensity::Density()> Binary tree not found!" << Endl;

   //create volume around point to be found
   std::vector<Double_t> lb(GetBox().size());
   std::vector<Double_t> ub(GetBox().size());

   // probevolume relative to hypercube with edge length 1:
   const Double_t probevolume_inv = 1.0 / GetBoxVolume();

   // set upper and lower bound for search volume
   for (UInt_t idim = 0; idim < GetBox().size(); ++idim) {
      lb[idim] = Xarg[idim] - GetBox().at(idim) / 2.0;
      ub[idim] = Xarg[idim] + GetBox().at(idim) / 2.0;
   }

   TMVA::Volume volume(&lb, &ub);                        // volume to search in
   std::vector<const TMVA::BinarySearchTreeNode*> nodes; // BST nodes found

   // do range searching
   Double_t SumOfWeights = fBst->SearchVolume(&volume, &nodes);

   // store density based on total number of events
   event_density = nodes.size() * probevolume_inv;

   Double_t N_tar = 0;           // number of target events found
   // now sum over all nodes->GetTarget(0);
   for (std::vector<const TMVA::BinarySearchTreeNode*>::const_iterator it = nodes.begin();
        it != nodes.end(); ++it) {
      N_tar += ((*it)->GetTargets()).at(fTarget) * ((*it)->GetWeight());
   }

   // return:  (N_tar/N_total) / (cell_volume)
   return (N_tar / (SumOfWeights + 0.1)) * probevolume_inv;
}
