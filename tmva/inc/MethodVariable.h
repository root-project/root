// @(#)root/tmva $Id: MethodVariable.h,v 1.13 2006/08/30 22:19:59 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodVariable                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Wrapper class for a single variable "MVA"; this is required for           *
 *      the evaluation of the single variable discrimination performance          *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodVariable
#define ROOT_TMVA_MethodVariable

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodVariable                                                       //
//                                                                      //
// Wrapper class for a single variable "MVA"; this is required for      //
// the evaluation of the single variable discrimination performance     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

namespace TMVA {

   class MethodVariable : public MethodBase {

   public:

      MethodVariable( TString jobName, 
                      TString methodTitle, 
                      DataSet& theData,
                      TString theOption = "Variable",
                      TDirectory* theTargetDir = 0 );
  
      virtual ~MethodVariable( void );
    
      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      virtual Double_t GetMvaValue();

      // write method specific histos to target file
      virtual void WriteHistosToFile( void ) const;

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

   protected:

   private:

      ClassDef(MethodVariable,0) // Wrapper class for a single variable "MVA" 
         };

} // namespace TMVA

#endif
