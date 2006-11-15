// @(#)root/tmva $Id: IMethod.h,v 1.18 2006/11/02 15:44:49 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : IMethod                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Interface for all concrete MVA method implementations                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
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

#ifndef ROOT_TMVA_IMethod
#define ROOT_TMVA_IMethod

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// IMethod                                                              //
//                                                                      //
// Interface for all concrete MVA method implementations                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <iostream>
#include "TROOT.h"
#include "TTree.h"
#include "TDirectory.h"

#ifndef ROOT_TMVA_Ranking
#include "TMVA/Ranking.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

namespace TMVA {
   
   class IMethod : public TObject {
      
   public:
      
      // default constructur
      IMethod() : TObject() {}
      
      // default destructur
      virtual ~IMethod() {}

      // ------- virtual member functions to be implemented by each MVA method

      // calculate the MVA value
      virtual Double_t GetMvaValue() = 0;
      
      // training method
      virtual void Train( void ) = 0;

      // write weights to output stream
      virtual void WriteWeightsToStream( std::ostream & ) const = 0;

      // read weights from output stream
      virtual void ReadWeightsFromStream( std::istream & ) = 0;
      
      // write method specific monitoring histograms to target file
      virtual void WriteMonitoringHistosToFile( void ) const = 0;

      // create ranking
      virtual const Ranking* CreateRanking() = 0;

      virtual void DeclareOptions() = 0;
      virtual void ProcessOptions() = 0;

      // method that gives a guess whether an event is signal-like
      virtual Bool_t IsSignalLike() = 0;

      // ------- virtual member functions implemented in MethodBase, 
      //         that can be overwritten by MVA methods
            
      // prepare tree branch with the method's discriminating variable
      virtual void PrepareEvaluationTree( TTree*  ) = 0;

      // evaluate method (resulting discriminating variable) or input varible
      virtual void TestInit( TTree* tr = 0 ) = 0;

      // test the method
      virtual void Test( TTree * theTestTree = 0 ) = 0;

      // member functions for the "evaluation" 
      virtual Bool_t IsOK( void )  const = 0;

      // variables (and private menber functions) for the Evaluation:
      // get the effiency. It fills a histogram for efficiency/vs/bkg
      // and returns the one value fo the efficiency demanded for 
      // in the TString argument. (Watch the string format)
      virtual Double_t GetEfficiency  ( TString , TTree *) = 0;
      virtual Double_t GetTrainingEfficiency( TString ) = 0;
      virtual Double_t GetSignificance( void ) = 0;
      virtual Double_t GetSeparation  ( void ) = 0;
      virtual Double_t GetmuTransform ( TTree* ) = 0;

      virtual void WriteStateToStream(std::ostream& o) const = 0;
      virtual void ReadStateFromStream( std::istream& i) = 0;
      
      // ------- virtual member functions implemented in MethodBase, 
      //         that should not be overwritten by MVA methods

      // write method-specific histograms to file
      virtual void    WriteEvaluationHistosToFile( TDirectory* targetDir ) = 0;

      // job accessors
      virtual TString GetWeightFileExtension( void ) const = 0;
      virtual void    SetWeightFileExtension( TString fileExtension ) = 0;
      virtual TString GetWeightFileDir( void ) const = 0;
      virtual void    SetWeightFileDir( TString fileDir ) = 0;

      // method name specific functions
      virtual const TString&   GetJobName    ( void ) const = 0;
      virtual const TString&   GetMethodName ( void ) const = 0;
      virtual const TString&   GetMethodTitle( void ) const = 0;
      virtual const Types::MVA GetMethodType ( void ) const = 0;

      virtual Types::PreprocessingMethod GetPreprocessingMethod() const = 0;

      virtual void  SetJobName( TString jobName ) = 0;

      // general read class
      virtual void ReadStateFromFile( void ) = 0;

      ClassDef(IMethod,0) // Method Interface
         ;
   };
} // namespace TMVA

#endif
