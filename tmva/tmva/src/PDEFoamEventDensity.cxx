// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamEventDensity                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      The TFDSITR class provides an interface between the Binary search tree    *
 *      and the PDEFoam object.  In order to build-up the foam one needs to       *
 *      calculate the density of events at a given point (sampling during         *
 *      Foam build-up).  The function PDEFoamEventDensity::Density() does         *
 *      this job.  It                                                             *
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

/*! \class TMVA::PDEFoamEventDensity
\ingroup TMVA
This is a concrete implementation of PDEFoam.  Density(...)
estimates the event (weight) density at a given phase-space point
using range-searching.
*/

#include "TMVA/PDEFoamEventDensity.h"

#include "TMVA/BinarySearchTree.h"
#include "TMVA/BinarySearchTreeNode.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/PDEFoamDensityBase.h"
#include "TMVA/Types.h"
#include "TMVA/Volume.h"

#include "Rtypes.h"

#include <cmath>
#include <vector>

ClassImp(TMVA::PDEFoamEventDensity);

////////////////////////////////////////////////////////////////////////////////

TMVA::PDEFoamEventDensity::PDEFoamEventDensity()
: PDEFoamDensityBase()
{}

////////////////////////////////////////////////////////////////////////////////
/// User constructor
///
/// Parameters:
///
/// - box - size of sampling box

TMVA::PDEFoamEventDensity::PDEFoamEventDensity(std::vector<Double_t> box)
   : PDEFoamDensityBase(box)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamEventDensity::PDEFoamEventDensity(const PDEFoamEventDensity &distr)
   : PDEFoamDensityBase(distr)
{
}

////////////////////////////////////////////////////////////////////////////////
/// This function is needed during the foam buildup.  It returns the
/// event density within the range-searching volume (specified by
/// fBox).
///
/// Parameters:
///
/// - xev - event vector (in [fXmin,fXmax]) to place the box at
///
/// - event_density - here the event density is stored
///
/// Returns:
///
/// Number of events (event weights), which were found in the
/// range-searching volume at point 'xev', divided by the box
/// volume.

Double_t TMVA::PDEFoamEventDensity::Density(std::vector<Double_t> &xev, Double_t &event_density)
{
   if (!fBst)
      Log() << kFATAL << "<PDEFoamEventDensity::Density()> Binary tree not found!" << Endl;

   //create volume around point to be found
   std::vector<Double_t> lb(GetBox().size());
   std::vector<Double_t> ub(GetBox().size());

   // probevolume relative to hypercube with edge length 1:
   const Double_t probevolume_inv = 1.0 / GetBoxVolume();

   // set upper and lower bound for search volume
   for (UInt_t idim = 0; idim < GetBox().size(); ++idim) {
      lb[idim] = xev[idim] - GetBox().at(idim) / 2.0;
      ub[idim] = xev[idim] + GetBox().at(idim) / 2.0;
   }

   TMVA::Volume volume(&lb, &ub);                        // volume to search in
   std::vector<const TMVA::BinarySearchTreeNode*> nodes; // BST nodes found

   // do range searching
   const Double_t sumOfWeights = fBst->SearchVolume(&volume, &nodes);

   // store density based on total number of events
   event_density = nodes.size() * probevolume_inv;

   // return:  N_total(weighted) / cell_volume
   return (sumOfWeights + 0.1) * probevolume_inv;
}
