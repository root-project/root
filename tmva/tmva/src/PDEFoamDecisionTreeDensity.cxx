// @(#)root/tmva $Id$
// Author: Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDecisionTreeDensity                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      This class provides an interface between the Binary search tree           *
 *      and the PDEFoam object.  In order to build-up the foam one needs to       *
 *      calculate the density of events at a given point (sampling during         *
 *      Foam build-up).  The function PDEFoamDecisionTreeDensity::Density()       *
 *      does this job. It uses a binary search tree, filled with training         *
 *      events, in order to provide this density.                                 *
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

/*! \class TMVA::PDEFoamDecisionTreeDensity
\ingroup TMVA

This is a concrete implementation of PDEFoam.  The Density(...)
function returns allways 0.  The function FillHistograms() is
added, which returns all events in a given TMVA::Volume.
*/

#include "TMVA/PDEFoamDecisionTreeDensity.h"

#include "TMVA/BinarySearchTree.h"
#include "TMVA/MethodPDERS.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/PDEFoamDensityBase.h"
#include "TMVA/Types.h"
#include "TMVA/Volume.h"

#include "RtypesCore.h"
#include "TH1D.h"

#include <limits>

ClassImp(TMVA::PDEFoamDecisionTreeDensity);

////////////////////////////////////////////////////////////////////////////////

TMVA::PDEFoamDecisionTreeDensity::PDEFoamDecisionTreeDensity()
: PDEFoamDensityBase()
   , fClass(0)
{}

////////////////////////////////////////////////////////////////////////////////
/// User constructor:
///
/// Parameters:
///
/// - box - size of the range-searching box (n-dimensional
///   std::vector)
///
/// - cls - event class used for the range-searching

TMVA::PDEFoamDecisionTreeDensity::PDEFoamDecisionTreeDensity(std::vector<Double_t> box, UInt_t cls)
   : PDEFoamDensityBase(box)
   , fClass(cls)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamDecisionTreeDensity::PDEFoamDecisionTreeDensity(const PDEFoamDecisionTreeDensity &distr)
   : PDEFoamDensityBase(distr)
   , fClass(distr.fClass)
{
}

////////////////////////////////////////////////////////////////////////////////
/// This function is not used in the decision tree like PDEFoam,
/// instead FillHist() is used.

Double_t TMVA::PDEFoamDecisionTreeDensity::Density(std::vector<Double_t>& /* Xarg */,
                                                   Double_t&              /* event_density */)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill the given histograms with signal and background events,
/// which are found in the volume.
///
/// Parameters:
///
/// - volume - volume box to search in
///
/// - hsig, hbkg, hsig_unw, hbkg_unw - histograms with weighted and
///   unweighted signal and background events

void TMVA::PDEFoamDecisionTreeDensity::FillHistograms(TMVA::Volume &volume, std::vector<TH1D*> &hsig,
                                                      std::vector<TH1D*> &hbkg, std::vector<TH1D*> &hsig_unw,
                                                      std::vector<TH1D*> &hbkg_unw)
{
   // sanity check
   if (hsig.size() != volume.fLower->size()
       || hbkg.size() != volume.fLower->size()
       || hsig_unw.size() != volume.fLower->size()
       || hbkg_unw.size() != volume.fLower->size())
      Log() << kFATAL << "<PDEFoamDistr::FillHistograms> Edge histograms have wrong size!" << Endl;

   // check histograms
   for (UInt_t idim = 0; idim < hsig.size(); ++idim) {
      if (!hsig.at(idim) || !hbkg.at(idim) ||
          !hsig_unw.at(idim) || !hbkg_unw.at(idim))
         Log() << kFATAL << "<PDEFoamDistr::FillHist> Histograms not initialized!" << Endl;
   }

   // BST nodes found in volume
   std::vector<const TMVA::BinarySearchTreeNode*> nodes;

   // do range searching
   fBst->SearchVolume(&volume, &nodes);

   // calc xmin and xmax of events found in cell
   std::vector<Float_t> xmin(volume.fLower->size(), std::numeric_limits<float>::max());
   std::vector<Float_t> xmax(volume.fLower->size(), -std::numeric_limits<float>::max());
   for (std::vector<const TMVA::BinarySearchTreeNode*>::const_iterator it = nodes.begin();
        it != nodes.end(); ++it) {
      std::vector<Float_t> ev = (*it)->GetEventV();
      for (UInt_t idim = 0; idim < xmin.size(); ++idim) {
         if (ev.at(idim) < xmin.at(idim))  xmin.at(idim) = ev.at(idim);
         if (ev.at(idim) > xmax.at(idim))  xmax.at(idim) = ev.at(idim);
      }
   }

   // reset histogram ranges to xmin, xmax found in volume
   for (UInt_t idim = 0; idim < hsig.size(); ++idim) {
      hsig.at(idim)->GetXaxis()->SetLimits(xmin.at(idim), xmax.at(idim));
      hbkg.at(idim)->GetXaxis()->SetLimits(xmin.at(idim), xmax.at(idim));
      hsig_unw.at(idim)->GetXaxis()->SetLimits(xmin.at(idim), xmax.at(idim));
      hbkg_unw.at(idim)->GetXaxis()->SetLimits(xmin.at(idim), xmax.at(idim));
      hsig.at(idim)->Reset();
      hbkg.at(idim)->Reset();
      hsig_unw.at(idim)->Reset();
      hbkg_unw.at(idim)->Reset();
   }

   // fill histograms with events found
   for (std::vector<const TMVA::BinarySearchTreeNode*>::const_iterator it = nodes.begin();
        it != nodes.end(); ++it) {
      std::vector<Float_t> ev = (*it)->GetEventV();
      Float_t              wt = (*it)->GetWeight();
      for (UInt_t idim = 0; idim < ev.size(); ++idim) {
         if ((*it)->GetClass() == fClass) {
            hsig.at(idim)->Fill(ev.at(idim), wt);
            hsig_unw.at(idim)->Fill(ev.at(idim), 1);
         } else {
            hbkg.at(idim)->Fill(ev.at(idim), wt);
            hbkg_unw.at(idim)->Fill(ev.at(idim), 1);
         }
      }
   }
}
