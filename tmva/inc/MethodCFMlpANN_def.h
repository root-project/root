/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Header : MethodCFMlpANN_def                                                    *
 *                                                                                *
 * Description:                                                                   *
 *      Common definition for CFMlpANN method                                     *
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
 * $Id: MethodCFMlpANN_def.h,v 1.3 2006/05/19 23:02:50 andreas.hoecker Exp $     
 **********************************************************************************/

// ------------- common definitions used in several modules --------------
// recovered explicit array definitions from f2c override

#ifndef ROOT_TMVA_MethodCFMlpANN_def
#define ROOT_TMVA_MethodCFMlpANN_def

namespace TMVA {

  const int  max_Events_  = 200000;
  const int  max_nLayers_ = 6;
  const int  max_nNodes_  = 200;
  const int  max_nVar_    = 200;

} 

#endif
