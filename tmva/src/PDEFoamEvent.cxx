// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamEvent                                                          *
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
// PDEFoamEvent
//
// This PDEFoam variant stores in every cell the sum of event weights
// and the sum of the squared event weights.  It therefore acts as
// event density estimator.  It should be booked together with the
// PDEFoamEventDensity density estimator, which returns the event
// weight density at a given phase space point during the foam
// build-up.
//
//_____________________________________________________________________

#ifndef ROOT_TMVA_PDEFoamEvent
#include "TMVA/PDEFoamEvent.h"
#endif

ClassImp(TMVA::PDEFoamEvent)

//_____________________________________________________________________
TMVA::PDEFoamEvent::PDEFoamEvent()
   : PDEFoam()
{
   // Default constructor for streamer, user should not use it.
}

//_____________________________________________________________________
TMVA::PDEFoamEvent::PDEFoamEvent(const TString& Name)
   : PDEFoam(Name)
{}

//_____________________________________________________________________
TMVA::PDEFoamEvent::PDEFoamEvent(const PDEFoamEvent &From)
   : PDEFoam(From)
{
   // Copy Constructor  NOT IMPLEMENTED (NEVER USED)
   Log() << kFATAL << "COPY CONSTRUCTOR NOT IMPLEMENTED" << Endl;
}

//_____________________________________________________________________
void TMVA::PDEFoamEvent::FillFoamCells(const Event* ev, Float_t wt)
{
   // This function fills an event weight 'wt' into the PDEFoam.  Cell
   // element 0 is filled with the weight 'wt', and element 1 is
   // filled with the squared weight.

   // find corresponding foam cell
   std::vector<Float_t> values  = ev->GetValues();
   std::vector<Float_t> tvalues = VarTransform(values);
   PDEFoamCell *cell = FindCell(tvalues);

   // 0. Element: Sum of event weights 'wt'
   // 1. Element: Sum of squared event weights 'wt'
   SetCellElement(cell, 0, GetCellElement(cell, 0) + wt);
   SetCellElement(cell, 1, GetCellElement(cell, 1) + wt * wt);
}
