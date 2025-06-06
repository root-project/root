// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodTMlpANN                                                         *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of interface for Root-integrated artificial neural         *
 *      network: TMultiLayerPerceptron, author: Christophe.Delaere@cern.ch        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodTMlpANN
#define ROOT_TMVA_MethodTMlpANN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodTMlpANN                                                        //
//                                                                      //
// Implementation of interface for Root-integrated artificial neural    //
// network: TMultiLayerPerceptron                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/MethodBase.h"

class TMultiLayerPerceptron;

namespace TMVA {

   class MethodTMlpANN : public MethodBase {

   public:

      MethodTMlpANN( const TString& jobName,
                     const TString& methodTitle,
                     DataSetInfo& theData,
                     const TString& theOption = "3000:N-1:N-2");

      MethodTMlpANN( DataSetInfo& theData,
                     const TString& theWeightFile);

      virtual ~MethodTMlpANN( void );

      Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets ) override;

      // training method
      void Train( void ) override;

      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void AddWeightsXMLTo( void* parent ) const override;

      // read weights from file
      void ReadWeightsFromStream( std::istream& istr ) override;
      void ReadWeightsFromXML(void* wghtnode) override;

      // calculate the MVA value ...
      // - here it is just a dummy, as it is done in the overwritten
      // - PrepareEvaluationtree... ugly but necessary due to the structure
      //   of TMultiLayerPercepton in ROOT grr... :-(
      Double_t GetMvaValue( Double_t* err = nullptr, Double_t* errUpper = nullptr ) override;

      void SetHiddenLayer(TString hiddenlayer = "" ) { fHiddenLayer=hiddenlayer; }

      // ranking of input variables
      const Ranking* CreateRanking() override { return nullptr; }

      // make ROOT-independent C++ class
      void MakeClass( const TString& classFileName = TString("") ) const override;

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const override;

      // get help message text
      void GetHelpMessage() const override;

   private:

      // the option handling methods
      void DeclareOptions() override;
      void ProcessOptions() override;

      void CreateMLPOptions( TString );

      // option string
      TString fLayerSpec;          ///< Layer specification option

      TMultiLayerPerceptron* fMLP; ///< the TMLP
      TTree*                 fLocalTrainingTree; ///< local copy of training tree

      TString  fHiddenLayer;        ///< string containing the hidden layer structure
      Int_t    fNcycles;            ///< number of training cycles
      Double_t fValidationFraction; ///< fraction of events in training tree used for cross validation
      TString  fMLPBuildOptions;    ///< option string to build the mlp

      TString  fLearningMethod;     ///< the learning method (given via option string)

      // default initialisation called by all constructors
      void Init( void ) override;

      ClassDefOverride(MethodTMlpANN,0); // Implementation of interface for TMultiLayerPerceptron
   };

} // namespace TMVA

#endif
