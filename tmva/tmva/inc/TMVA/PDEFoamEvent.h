// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamEvent                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Concrete PDEFoam sub-class.  This foam stores the number of               *
 *      events with every cell, as well as the statistical error on               *
 *      the event number.                                                         *
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

#ifndef ROOT_TMVA_PDEFoamEvent
#define ROOT_TMVA_PDEFoamEvent

#include "TMVA/PDEFoam.h"

namespace TMVA
{

   class PDEFoamEvent : public PDEFoam
   {

   protected:

      PDEFoamEvent(const PDEFoamEvent&); // Copy Constructor  NOT USED

      // ---------- Public functions ----------------------------------
   public:
      PDEFoamEvent();                  // Default constructor (used only by ROOT streamer)
      PDEFoamEvent(const TString&);    // Principal user-defined constructor
      virtual ~PDEFoamEvent() {}       // Default destructor

      // function to fill created cell with given value
      virtual void FillFoamCells(const Event* ev, Float_t wt);

      // ---------- ROOT class definition
      ClassDef(PDEFoamEvent, 1) // Tree of PDEFoamCells
         }; // end of PDEFoamEvent

}  // namespace TMVA

// ---------- Inline functions

#endif
