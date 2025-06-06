// @(#)root/tmva $Id$
// Author: Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamKernelTrivial                                                  *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Trivial PDEFoam kernel                                                    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *                                                                                *
 * Copyright (c) 2010:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDEFoamKernelTrivial
#define ROOT_TMVA_PDEFoamKernelTrivial

#include "TMVA/PDEFoam.h"
#include "TMVA/PDEFoamKernelBase.h"

#include <vector>

namespace TMVA
{

   class PDEFoamKernelTrivial : public PDEFoamKernelBase
   {

   public:
      PDEFoamKernelTrivial();                 // Constructor
      PDEFoamKernelTrivial(const PDEFoamKernelTrivial&); // Copy Constructor
      virtual ~PDEFoamKernelTrivial() {}      // Destructor

      // kernel estimator
      Float_t Estimate(PDEFoam*, std::vector<Float_t>&, ECellValue) override;

      ClassDefOverride(PDEFoamKernelTrivial, 1) // trivial PDEFoam kernel estimator
         }; // end of PDEFoamKernelTrivial
}  // namespace TMVA

#endif
