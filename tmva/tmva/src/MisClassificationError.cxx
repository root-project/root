// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MisClassificationError                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Implementation of the MisClassificationError as separation        *
 *              criterion:   1-max(p, 1-p) as 
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
// Implementation of the MisClassificationError as separation criterion 
//                                                                      
//_______________________________________________________________________


#include "TMVA/MisClassificationError.h"

ClassImp(TMVA::MisClassificationError)

//_______________________________________________________________________
Double_t  TMVA::MisClassificationError::GetSeparationIndex( const Double_t &s, const Double_t &b )
{
   // Misclassifiacton error   criterion: 1-max(p, 1-p)  (p: purity= s/(s+b))
   if ( s+b <= 0) return 0;

   Double_t p = s/(s+b);
   if (p >= 1-p)  return 1-p;
   else           return p;
}


