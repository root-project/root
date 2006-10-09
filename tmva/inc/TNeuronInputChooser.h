// @(#)root/tmva $Id: TNeuronInputChooser.h,v 1.3 2006/09/30 19:59:32 stelzer Exp $
// Author: Matt Jachowski 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TNeuronInputChooser                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Class for easily choosing neuron input functions.                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/
 

#ifndef ROOT_TMVA_TNeuronInputChooser
#define ROOT_TMVA_TNeuronInputChooser

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNeuronInputChooser                                                  //
//                                                                      //
// Class for easily choosing neuron input functions                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include "TString.h"

#ifndef ROOT_TMVA_TActivation
#include "TNeuronInput.h"
#endif
#ifndef ROOT_TMVA_TNeuronInputSum
#include "TNeuronInputSum.h"
#endif
#ifndef ROOT_TMVA_TNeuronInputSqSum
#include "TNeuronInputSqSum.h"
#endif
#ifndef ROOT_TMVA_TNeuronInputAbs
#include "TNeuronInputAbs.h"
#endif

namespace TMVA {

   class TNeuron;
  
   class TNeuronInputChooser {
    
   public:

      TNeuronInputChooser()
      {
         SUM    = "sum";
         SQSUM  = "sqsum";
         ABSSUM = "abssum";
      }
      virtual ~TNeuronInputChooser() {}

      enum NeuronInputType { kSum = 0,
                             kSqSum,
                             kAbsSum
      };

      TNeuronInput* CreateNeuronInput(const NeuronInputType type) const
      {
         switch (type) {
         case kSum:    return new TNeuronInputSum();
         case kSqSum:  return new TNeuronInputSqSum();
         case kAbsSum: return new TNeuronInputAbs();
         default: return NULL;
         }
         return NULL;
      }
     
      TNeuronInput* CreateNeuronInput(const TString type) const
      {
         if      (type == SUM)    return CreateNeuronInput(kSum);
         else if (type == SQSUM)  return CreateNeuronInput(kSqSum);
         else if (type == ABSSUM) return CreateNeuronInput(kAbsSum);
         else                     return NULL;
      }
     
      std::vector<TString>* GetAllNeuronInputNames() const
      {
         std::vector<TString>* names = new std::vector<TString>();
         names->push_back(SUM);
         names->push_back(SQSUM);
         names->push_back(ABSSUM);
         return names;
      }
      
   private:
     
      TString SUM; 
      TString SQSUM;
      TString ABSSUM;

      ClassDef(TNeuronInputChooser,0) // Class for choosing neuron input functions
   };

} // namespace TMVA

#endif
