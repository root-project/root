// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDiscriminant                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *    Concrete PDEFoam sub-class.  This foam stores the discriminant D            *
 *    = N_sig / (N_bg + N_sig) with every cell, as well as the                    *
 *    statistical error on the discriminant.                                      *
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

#ifndef ROOT_TMVA_PDEFoamDiscriminant
#define ROOT_TMVA_PDEFoamDiscriminant

#include "TMVA/PDEFoam.h"

namespace TMVA
{

   class PDEFoamDiscriminant : public PDEFoam
   {

   protected:
      UInt_t fClass; // signal class

      PDEFoamDiscriminant(const PDEFoamDiscriminant&); // Copy Constructor  NOT USED

      // ---------- Public functions ----------------------------------
   public:
      PDEFoamDiscriminant();                  // Default constructor (used only by ROOT streamer)
      PDEFoamDiscriminant(const TString&, UInt_t); // Principal user-defined constructor
      virtual ~PDEFoamDiscriminant() {}       // Default destructor

      // function to fill created cell with given value
      virtual void FillFoamCells(const Event* ev, Float_t wt);

      // function to call after foam is grown
      virtual void Finalize();

      // 2-dimensional projection
      virtual TH2D* Project2(Int_t, Int_t, ECellValue, PDEFoamKernelBase*, UInt_t);

      // ---------- ROOT class definition
      ClassDef(PDEFoamDiscriminant, 1) // Tree of PDEFoamCells
         }; // end of PDEFoamDiscriminant

}  // namespace TMVA

// ---------- Inline functions

#endif
