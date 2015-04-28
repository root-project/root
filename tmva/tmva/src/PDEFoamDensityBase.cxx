// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDensityBase                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      This class provides an interface between the Binary search tree           *
 *      and the PDEFoam object.  In order to build-up the foam one needs to       *
 *      calculate the density of events at a given point (sampling during         *
 *      Foam build-up).  The function PDEFoamDensityBase::Density() does this job. It *
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
// PDEFoamDensityBase
//
// This is an abstract class, which provides an interface for a
// PDEFoam density estimator.  Derived classes have to implement the
// Density(...) function, which returns the density of a certain
// quantity at a given phase-space point during the foam build-up.
//
// Variants of PDEFoamDensityBase are:
//
//   - PDEFoamEventDensity
//   - PDEFoamDiscriminantDensity
//   - PDEFoamTargetDensity
//   - PDEFoamDecisionTreeDensity
//
// Usage:
//
// The user has to instantiate a child class of PDEFoamDensityBase and
// set the pointer to the owner, which is a PDEFoam object:
//
//   PDEFoamDensityBase *dens = new MyDensity();
//   pdefoam->SetDensity(dens);
//
// Afterwards the binary search tree should be filled with TMVA
// events, by either using
//
//   pdefoam->FillBinarySearchTree(event);
//
// or
//
//   dens->FillBinarySearchTree(event);
// _____________________________________________________________________

#include <numeric>
#include <functional>

#ifndef ROOT_TMVA_PDEFoamDensityBase
#include "TMVA/PDEFoamDensityBase.h"
#endif

ClassImp(TMVA::PDEFoamDensityBase)

//_____________________________________________________________________
TMVA::PDEFoamDensityBase::PDEFoamDensityBase()
   : TObject(),
     fBox(),
     fBoxVolume(1.0),
     fBoxHasChanged(kTRUE),
     fBst(new TMVA::BinarySearchTree()),
     fLogger(new MsgLogger("PDEFoamDensityBase"))
{}

//_____________________________________________________________________
TMVA::PDEFoamDensityBase::PDEFoamDensityBase(std::vector<Double_t> box)
   : TObject(),
     fBox(box),
     fBoxVolume(1.0),
     fBoxHasChanged(kTRUE),
     fBst(new TMVA::BinarySearchTree()),
     fLogger(new MsgLogger("PDEFoamDensityBase"))
{
   // User constructor
   //
   // - box - range-searching box, where box.size() == dimension of
   //         the PDEFoam == periode of the binary search tree

   if (box.empty())
      Log() << kFATAL << "Dimension of PDEFoamDensityBase is zero" << Endl;

   // set periode (number of variables) of binary search tree
   fBst->SetPeriode(box.size());
}

//_____________________________________________________________________
TMVA::PDEFoamDensityBase::~PDEFoamDensityBase()
{
   // destructor
   if (fBst)    delete fBst;
   if (fLogger) delete fLogger;
}

//_____________________________________________________________________
TMVA::PDEFoamDensityBase::PDEFoamDensityBase(const PDEFoamDensityBase &distr)
   : TObject(),
     fBox(distr.fBox),
     fBoxVolume(distr.fBoxVolume),
     fBoxHasChanged(distr.fBoxHasChanged),
     fBst(new BinarySearchTree(*distr.fBst)),
     fLogger(new MsgLogger(*distr.fLogger))
{
   // Copy constructor
   //
   // Creates a deep copy, using the copy constructor of
   // TMVA::BinarySearchTree
}

//_____________________________________________________________________
void TMVA::PDEFoamDensityBase::FillBinarySearchTree(const Event* ev)
{
   // This method inserts the given event 'ev' it into the binary
   // search tree.

   if (fBst == NULL)
      Log() << kFATAL << "<PDEFoamDensityBase::FillBinarySearchTree> "
            << "Binary tree is not set!" << Endl;

   // insert into binary search tree
   fBst->Insert(ev);
}

//_____________________________________________________________________
Double_t TMVA::PDEFoamDensityBase::GetBoxVolume()
{
   // Returns the volume of range searching box fBox.
   //
   // If the range searching box 'fBox' has changed (fBoxHasChanged is
   // kTRUE), recalculate the box volume and set fBoxHasChanged to
   // kFALSE
   if (fBoxHasChanged) {
      fBoxHasChanged = kFALSE;
      fBoxVolume = std::accumulate(fBox.begin(), fBox.end(), 1.0,
                                   std::multiplies<Double_t>());
   }
   return fBoxVolume;
}
