// @(#)root/tmva $Id: MStream.h,v 1.2 2006/08/30 22:19:58 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MStream                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      ostream derivative to redirect and format output  MStream message stream  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MStream
#define ROOT_TMVA_MStream

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MStream                                                              //
//                                                                      //
// ostream derivative to redirect and format output                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include <iostream>
#include <map>

class TString;

namespace TMVA {

   const TString _MStream_prefix_ = "--- ";

   class MStream : public ostream {
  
   public:

      enum OutputLevel { DEBUG = 0,
                         VERBOSE,
                         INFO, 
                         WARNING, 
                         ERROR                          
      };      

      MStream( ostream& os, OutputLevel = INFO, const TString prefix = _MStream_prefix_ );
      MStream( OutputLevel ol = INFO, const TString prefix = _MStream_prefix_ );
      virtual ~MStream();

      // print to ostream (if ol == ERROR, abort program exectution)
      ostream& operator << (OutputLevel ol); 

   private:

      void InitMap();              // initialize map for output text

      OutputLevel m_outputLevel;   // the output level
      TString     m_prefix;        // prefix in front of any TMVA output

      std::map<OutputLevel,TString> m_outputText; // text-to-level mapping
      
      ClassDef(MStream,0)  // MStream wrapper for redirected output
   };

} // namespace TMVA


#endif 
