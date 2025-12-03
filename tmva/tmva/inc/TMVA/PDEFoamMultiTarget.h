// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamMultiTarget                                                    *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Concrete PDEFoamEvent sub-class.  This foam stores the number             *
 *      of events with every cell, as well as the statistical error on            *
 *      the event number.  It overrides GetCellValue() for projecting             *
 *      the target values given an incomplete event map with                      *
 *      N_variables < dimension of foam.                                          *
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
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDEFoamMultiTarget
#define ROOT_TMVA_PDEFoamMultiTarget

#include "TMVA/PDEFoamEvent.h"

#include <vector>
#include <map>

namespace TMVA
{

   // target selection method
   enum ETargetSelection { kMean = 0, kMpv = 1 };

   class PDEFoamMultiTarget : public PDEFoamEvent
   {

   protected:
      ETargetSelection fTargetSelection; // the target selection method

      PDEFoamMultiTarget(const PDEFoamMultiTarget&); // Copy Constructor  NOT USED
      virtual void CalculateMpv(std::map<Int_t, Float_t>&, const std::vector<PDEFoamCell*>&);  // Calculate mpv target
      virtual void CalculateMean(std::map<Int_t, Float_t>&, const std::vector<PDEFoamCell*>&); // Calculate mean target

      // ---------- Public functions ----------------------------------
   public:
      PDEFoamMultiTarget();                  // Default constructor (used only by ROOT streamer)
      PDEFoamMultiTarget(const TString&, ETargetSelection); // Principal user-defined constructor
      virtual ~PDEFoamMultiTarget() {}       // Default destructor

      // overridden from PDEFoam: extract the targets from the foam
      std::vector<Float_t> GetCellValue(const std::map<Int_t, Float_t>&, ECellValue) override;
      using PDEFoam::GetCellValue;

      // ---------- ROOT class definition
      ClassDefOverride(PDEFoamMultiTarget, 1) // Tree of PDEFoamCells
         }; // end of PDEFoamMultiTarget

}  // namespace TMVA

// ---------- Inline functions

#endif
