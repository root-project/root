// @(#)root/tmva $Id: MethodRuleFit.h,v 1.5 2006/05/23 09:53:10 stelzer Exp $    
// Author: Andreas Hoecker, Fredrik Tegenfeldt, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRuleFit                                                         *
 *                                                                                *
 * Description:                                                                   *
 *      Friedman's RuleFit method -- not yet implemented -- dummy class --        * 
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker    <Andreas.Hocker@cern.ch>     - CERN, Switzerland       *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-KP Heidelberg, Ger. *
 *      Kai Voss           <Kai.Voss@cern.ch>           - U. of Victoria, Canada  *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: MethodRuleFit.h,v 1.5 2006/05/23 09:53:10 stelzer Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodRuleFit
#define ROOT_TMVA_MethodRuleFit

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodRuleFit                                                        //
//                                                                      //
// Friedman's RuleFit method -- not yet implemented -- dummy class --   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMVA_TVectorD
#include "TVectorD.h"
#endif

namespace TMVA {

   class MethodRuleFit : public MethodBase {

   public:

      MethodRuleFit( TString jobName, 
                     vector<TString>* theVariables, 
                     TTree* theTree = 0, 
                     TString theOption = "",
                     TDirectory* theTargetDir = 0 );

      MethodRuleFit( vector<TString> *theVariables, 
                     TString theWeightFile,  
                     TDirectory* theTargetDir = NULL );

      virtual ~MethodRuleFit( void );
    
      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToFile( void );
  
      // read weights from file
      virtual void ReadWeightsFromFile( void );

      // calculate the MVA value
      virtual Double_t GetMvaValue(Event *e);

      // write method specific histos to target file
      virtual void WriteHistosToFile( void ) ;

      void InitRuleFit( void );

   protected:

   private:

      ClassDef(MethodRuleFit,0)  // Friedman's RuleFit method 
         };

} // namespace TMVA

#endif // MethodRuleFit_H
