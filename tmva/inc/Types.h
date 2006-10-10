// @(#)root/tmva $Id: Types.h,v 1.2 2006/05/23 13:03:15 brun Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Types                                                                 *
 *                                                                                *
 * Description:                                                                   *
 *      GLobal types                                                              *
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
 * (http://tmva.sourceforge.net/license.txt)                                      *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: Types.h,v 1.2 2006/05/23 13:03:15 brun Exp $        
 **********************************************************************************/

#ifndef ROOT_TMVA_Types
#define ROOT_TMVA_Types

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Types (namespace)                                                    //
//                                                                      //
// GLobal types used by TMVA                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace TMVA {
  
  namespace Types {
    
    // available MVA methods in TMVA
    enum MVA {
      Variable     = 1,
      Cuts         ,     
      Likelihood   ,
      PDERS        ,
      HMatrix      ,
      Fisher       ,
      CFMlpANN     ,
      TMlpANN      , 
      BDT          ,     
      RuleFit
    };
  }
}

#endif
