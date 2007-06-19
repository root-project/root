// @(#)root/tmva $Id: Methods.h,v 1.7 2007/04/19 06:53:01 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Methods                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Includes all TMVA methods                                                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_Methods
#define ROOT_TMVA_Methods

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Methods                                                              //
//                                                                      //
// Includes all TMVA methods                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_IMethod
#include "TMVA/IMethod.h"
#endif
#ifndef ROOT_TMVA_MethodCuts
#include "TMVA/MethodCuts.h"
#endif
#ifndef ROOT_TMVA_MethodFisher
#include "TMVA/MethodFisher.h"
#endif
#ifndef ROOT_TMVA_MethodKNN
#include "TMVA/MethodKNN.h"
#endif
#ifndef ROOT_TMVA_MethodMLP
#include "TMVA/MethodMLP.h"
#endif
#ifndef ROOT_TMVA_MethodTMlpANN
#include "TMVA/MethodTMlpANN.h"
#endif
#ifndef ROOT_TMVA_MethodCFMlpANN
#include "TMVA/MethodCFMlpANN.h"
#endif
#ifndef ROOT_TMVA_MethodLikelihood
#include "TMVA/MethodLikelihood.h"
#endif
#ifndef ROOT_TMVA_MethodVariable
#include "TMVA/MethodVariable.h"
#endif
#ifndef ROOT_TMVA_MethodHMatrix
#include "TMVA/MethodHMatrix.h"
#endif
#ifndef ROOT_TMVA_MethodPDERS
#include "TMVA/MethodPDERS.h"
#endif
#ifndef ROOT_TMVA_MethodBDT
#include "TMVA/MethodBDT.h"
#endif
#ifndef ROOT_TMVA_MethodSVM
#include "TMVA/MethodSVM.h"
#endif
#ifndef ROOT_TMVA_MethodRuleFit
#include "TMVA/MethodRuleFit.h"
#endif
#ifndef ROOT_TMVA_MethodBayesClassifier
#include "TMVA/MethodBayesClassifier.h"
#endif
#ifndef ROOT_TMVA_MethodFDA
#include "TMVA/MethodFDA.h"
#endif
#ifndef ROOT_TMVA_MethodSeedDistance
#include "TMVA/MethodSeedDistance.h"
#endif
#ifndef ROOT_TMVA_MethodCommittee
#include "TMVA/MethodCommittee.h"
#endif

#endif
