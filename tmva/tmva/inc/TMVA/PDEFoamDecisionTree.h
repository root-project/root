// @(#)root/tmva $Id$
// Author: Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDecisionTree                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Class for decision tree like PDEFoam.  It overrides                       *
 *      PDEFoam::Explore() to use the decision tree like cell split               *
 *      algorithm, given a specific separation type.                              *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDEFoamDecisionTree
#define ROOT_TMVA_PDEFoamDecisionTree

#ifndef ROOT_TMVA_PDEFoamDiscriminant
#include "TMVA/PDEFoamDiscriminant.h"
#endif
#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif

namespace TMVA
{

   class PDEFoamDecisionTree : public PDEFoamDiscriminant
   {

   private:
      SeparationBase *fSepType;    // separation type

   protected:

      virtual void Explore(PDEFoamCell *Cell);     // Exploration of the cell

      PDEFoamDecisionTree(const PDEFoamDecisionTree&); // Copy Constructor  NOT USED

   public:
      PDEFoamDecisionTree();               // Default constructor (used only by ROOT streamer)
      PDEFoamDecisionTree(const TString&, SeparationBase *sepType, UInt_t cls); // Principal user-defined constructor
      virtual ~PDEFoamDecisionTree();      // Default destructor

      // ---------- ROOT class definition
      ClassDef(PDEFoamDecisionTree, 1) // Decision tree like PDEFoam
   }; // end of PDEFoamDecisionTree

}  // namespace TMVA

#endif
