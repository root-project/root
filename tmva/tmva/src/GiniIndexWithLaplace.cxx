// @(#)root/tmva $Id$ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GiniIndex                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Implementation of the GiniIndex With Laplace correction           *
 *              as separation criterion                                           *
 *              Gini(Sample M) = 1 - (c(1)/N)^2 - (c(2)/N)^2 .... - (c(k)/N)^2    * 
 *              Where: M is a smaple of whatever N elements (events)              *
 *                     that belong to K different classes                         *
 *                     c(k) is the number of elements that belong to class k      *
 *              Laplace's correction to the prob.density c/N --> (c+1)/(N+2)      *
 *              for just Signal and Background classes this then boils down to:   *
 *              Gini(Sample) = 2(s*b+s+b+1)/(s+b+2)^2                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      Heidelberg U., Germany                                                    * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Implementation of the GiniIndexWithLaplace as separation criterion              
//                                                                      
//_______________________________________________________________________

#include "TMVA/GiniIndexWithLaplace.h"

ClassImp(TMVA::GiniIndexWithLaplace)

//_______________________________________________________________________
Double_t TMVA::GiniIndexWithLaplace::GetSeparationIndex( const Double_t &s, const Double_t &b )
{
   //     Gini(Sample M) = 1 - (c(1)/N)^2 - (c(2)/N)^2 .... - (c(k)/N)^2    
   //              Where: M is a smaple of whatever N elements (events)              
   //                      that belong to K different classes                         
   //                      c(k) is the number of elements that belong to class k      
   //               Laplace's correction to the prob.density c/N --> (c+1)/(N+2)      
   //               for just Signal and Background classes this then boils down to:   
   //               Gini(Sample) = 2(s*b+s+b+1)/(s+b+2)^2                               
   
   if (s+b <= 0)      return 0;
   if (s<=0 || b <=0) return 0;
   else               return (s*b+s+b+1)/(s+b+2)/(s+b+2); 
}


