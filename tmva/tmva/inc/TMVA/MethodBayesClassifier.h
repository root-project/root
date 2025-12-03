// @(#)root/tmva $Id$
// Author: Abhishek Narain

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBayesClassifier                                                 *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Bayesian Classifier                                                       *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Abhishek Narain, <narainabhi@gmail.com> - University of Houston           *
 *                                                                                *
 * Copyright (c) 2005-2006:                                                       *
 *      University of Houston,                                                    *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodBayesClassifier
#define ROOT_TMVA_MethodBayesClassifier

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodBayesClassifier                                                //
//                                                                      //
// Description...                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/MethodBase.h"
#include "TMVA/Types.h"

namespace TMVA {

   class MethodBayesClassifier : public MethodBase {

   public:

      MethodBayesClassifier( const TString& jobName,
                             const TString& methodTitle,
                             DataSetInfo& theData,
                             const TString& theOption = "");

      MethodBayesClassifier( DataSetInfo& theData,
                             const TString& theWeightFile);

      virtual ~MethodBayesClassifier( void );

      Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets ) override;

      // training method
      void Train( void ) override;

      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void AddWeightsXMLTo( void* parent ) const override;

      // read weights from file
      void ReadWeightsFromStream( std::istream& istr ) override;
      void ReadWeightsFromXML   ( void* /*wghtnode*/ ) override {}

      // calculate the MVA value
      Double_t GetMvaValue( Double_t* err = nullptr, Double_t* errUpper = nullptr ) override;

      void Init( void ) override;

      // ranking of input variables
      const Ranking* CreateRanking() override { return nullptr; }

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const override;

      // get help message text
      void GetHelpMessage() const override;

   private:

      // the option handling methods
      void DeclareOptions() override;
      void ProcessOptions() override;

      ClassDefOverride(MethodBayesClassifier,0);  // Friedman's BayesClassifier method
   };

} // namespace TMVA

#endif // MethodBayesClassifier_H
