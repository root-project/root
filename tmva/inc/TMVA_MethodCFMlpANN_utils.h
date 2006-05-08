// @(#)root/tmva $Id: TMVA_MethodCFMlpANN_utils.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodCFMlpANN_utils                                             *
 *                                                                                *
 * Reference for the original FORTRAN version "mlpl3.F":                          *
 *      Authors  : J. Proriol and contributions from ALEPH-Clermont-Fd            *
 *                 Team members                                                   *
 *      Copyright: Laboratoire Physique Corpusculaire                             *
 *                 Universite de Blaise Pascal, IN2P3/CNRS                        *
 * Description:                                                                   *
 *      Utility routine, obtained via f2c from original mlpl3.F FORTRAN routine   *
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
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodCFMlpANN_utils.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $ 
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCFMlpANN_utils
#define ROOT_TMVA_MethodCFMlpANN_utils

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// typedefs                                                             //
//                                                                      //
// typedefs for CFMlpANN implementation                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

/* external read, write */
typedef struct 
{
  Int_t cierr; 
  Int_t ciunit; 
  Int_t ciend; 
  char *cifmt; 
  Int_t cirec; 
} cilist; 

/* open */
typedef struct
{
  Int_t oerr;
  Int_t ounit;
  char *ofnm;
  Int_t ofnmlen;
  char *osta;
  char *oacc;
  char *ofm;
  Int_t orl;
  char *oblnk;
} olist;

/* close */ 
typedef struct 
{
  Int_t cerr; 
  Int_t cunit; 
  char *csta; 
} cllist; 

#endif
