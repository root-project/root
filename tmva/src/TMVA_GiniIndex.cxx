// @(#)root/tmva $Id: TMVA_GiniIndex.cpp,v 1.5 2006/05/02 23:27:40 helgevoss Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GiniIndex                                                        *
 *                                                                                *
 * Description: Implementation of the GiniIndex as separation criterion           *
 *                                                                                * 
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
 * File and Version Information:                                                  *
 *      $Id: TMVA_GiniIndex.cpp,v 1.5 2006/05/02 23:27:40 helgevoss Exp $       *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Implementation of the GiniIndex as separation criterion              
//                                                                      
//_______________________________________________________________________

#include "TMVA_GiniIndex.h"

ClassImp(TMVA_GiniIndex)

//_______________________________________________________________________
Double_t  TMVA_GiniIndex::GetSeparationIndex( const Double_t &s, const Double_t &b )
{
  // 2 * p * (1-p), with p=s/(s+b)  (s: correct selected events, b: wrong selected events)
  //  return 2 * s/(s+b) * ( 1 - s/(s+b)) ,  which can be simplified to:
  if (s+b <= 0) return 0.;
  if (s<=0 || b <=0) return 0.;
  //  else return 2 * s*b/(s+b)/(s+b);  
  else return s*b/(s+b)/(s+b);      //actually the "2" should not influence the result
}


