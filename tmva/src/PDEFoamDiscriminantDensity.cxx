// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDiscriminantDensity                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      The TFDSITR class provides an interface between the Binary search tree    *
 *      and the PDEFoam object.  In order to build-up the foam one needs to       *
 *      calculate the density of events at a given point (sampling during         *
 *      Foam build-up).  The function PDEFoamDiscriminantDensity::Density() does this job.  It  *
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
// PDEFoamDiscriminantDensity
//
// This is a concrete implementation of PDEFoam.  Density(...)
// estimates the discriminant density at a given phase-space point
// using range-searching.  The discriminant D is defined as
//
//    D = #events with given class / total number of events
// _____________________________________________________________________

#include <cmath>

#ifndef ROOT_TMVA_PDEFoamDiscriminantDensity
#include "TMVA/PDEFoamDiscriminantDensity.h"
#endif

ClassImp(TMVA::PDEFoamDiscriminantDensity)

//_____________________________________________________________________
TMVA::PDEFoamDiscriminantDensity::PDEFoamDiscriminantDensity()
   : PDEFoamDensityBase()
   , fClass(0)
{}

//_____________________________________________________________________
TMVA::PDEFoamDiscriminantDensity::PDEFoamDiscriminantDensity(std::vector<Double_t> box, UInt_t cls)
   : PDEFoamDensityBase(box)
   , fClass(cls)
{}

//_____________________________________________________________________
TMVA::PDEFoamDiscriminantDensity::PDEFoamDiscriminantDensity(const PDEFoamDiscriminantDensity &distr)
   : PDEFoamDensityBase(distr)
   , fClass(distr.fClass)
{
   // Copy constructor
}

//_____________________________________________________________________
Double_t TMVA::PDEFoamDiscriminantDensity::Density(std::vector<Double_t> &Xarg, Double_t &event_density)
{
   // This function is needed during the foam buildup.  It returns the
   // average target value within volume divided by volume (specified
   // by fVolFrac).

   if (!fBst)
      Log() << kFATAL << "<PDEFoamDiscriminantDensity::Density()> Binary tree not set!" << Endl;

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

   Double_t N_sig = 0;           // number of signal events found
   // calc number of signal events in nodes
   for (std::vector<const TMVA::BinarySearchTreeNode*>::const_iterator it = nodes.begin();
        it != nodes.end(); ++it) {
      if ((*it)->GetClass() == fClass) // signal node
         N_sig += (*it)->GetWeight();
   }

   // return:  (N_sig/N_total) / (cell_volume)
   return (N_sig / (SumOfWeights + 0.1)) * probevolume_inv;
}
