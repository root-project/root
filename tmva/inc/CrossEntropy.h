// @(#)root/tmva $Id: CrossEntropy.h,v 1.10 2007/04/19 06:53:01 brun Exp $       
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : CrossEntropy                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Implementation of the CrossEntropy as separation criterion        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      Heidelberg U., Germany                                                    * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_CrossEntropy
#define ROOT_TMVA_CrossEntropy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// CrossEntropy                                                         //
//                                                                      //
// Implementation of the CrossEntropy as separation criterion           //
//  -p log (p) - (1-p)log(1-p);     p=purity = s/(s+b)                  // 
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif

namespace TMVA {

   class CrossEntropy : public SeparationBase {

   public:
    
     // default constructor
      CrossEntropy() { fName = "CE"; }

      // copy constructor
      CrossEntropy( const CrossEntropy& g): SeparationBase(g) {}

      // destructor
      virtual ~CrossEntropy(){}

      // return the separation Index  -p log (p) - (1-p)log(1-p);     p=purity = s/(s+b) 
      virtual Double_t GetSeparationIndex( const Double_t &s, const Double_t &b );
    
   protected:
    
      ClassDef(CrossEntropy,0) // Implementation of the CrossEntropy as separation criterion
   };

} // namespace TMVA

#endif

