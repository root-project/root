// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer,Joerg Stelzer, Helge Voss, Omar Zapata

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableTransformBase                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Pre-transformation of input variables (base class)                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableTransform
#define ROOT_TMVA_VariableTransform

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableTransform                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TString;
namespace TMVA {
class DataSetInfo;
class TransformationHandler;
class MsgLogger;
void      CreateVariableTransforms(const TString& trafoDefinition,
                                   TMVA::DataSetInfo& dataInfo,
                                   TMVA::TransformationHandler& transformationHandler,
                                   TMVA::MsgLogger& log );
} // namespace TMVA

#endif
