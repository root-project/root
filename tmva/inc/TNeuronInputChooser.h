// @(#)root/tmva $Id: TNeuronInputChooser.h,v 1.4 2006/10/10 17:43:52 andreas.hoecker Exp $
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
         fSUM    = "sum";
         fSQSUM  = "sqsum";
         fABSSUM = "abssum";
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
         if      (type == fSUM)    return CreateNeuronInput(kSum);
         else if (type == fSQSUM)  return CreateNeuronInput(kSqSum);
         else if (type == fABSSUM) return CreateNeuronInput(kAbsSum);
         else                      return NULL;
      }
     
      std::vector<TString>* GetAllNeuronInputNames() const
      {
         std::vector<TString>* names = new std::vector<TString>();
         names->push_back(fSUM);
         names->push_back(fSQSUM);
         names->push_back(fABSSUM);
         return names;
      }
      
   private:
     
      TString fSUM;    // neuron input type name
      TString fSQSUM;  // neuron input type name
      TString fABSSUM; // neuron input type name

      ClassDef(TNeuronInputChooser,0) // Class for choosing neuron input functions
         ;
   };

} // namespace TMVA

#endif
