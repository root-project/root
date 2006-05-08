// @(#)root/tmva $Id: TMVA_CrossEntropy.cpp,v 1.4 2006/05/02 23:27:40 helgevoss Exp $       
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_CrossEntropy                                                     *
 *                                                                                *
 * Description: Implementation of the CrossEntropy as separation criterion        *
 *              -p log (p) - (1-p)log(1-p);     p=purity                          * 
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      Heidelberg U., Germany,                                                   * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Implementation of the CrossEntropy as separation criterion           
//                                                                      
//_______________________________________________________________________

#include <math.h>
#include "TMVA_CrossEntropy.h"

ClassImp(TMVA_CrossEntropy)

//_______________________________________________________________________
Double_t  TMVA_CrossEntropy::GetSeparationIndex( const Double_t &s, const Double_t &b )
{
  if (s+b <= 0) return 0;
  Double_t p = s/(s+b);
  if (p<=0 || p >=1) return 0;
  return - ( p * log (p) + (1-p)*log(1-p) );
}


