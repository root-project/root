// Author: Krzysztof Danielowski, Kamil Kraszewski, Maciej Kruk, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodLD                                                              *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Linear Discriminant (Simple Linear Regression)                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Krzysztof Danielowski <danielow@cern.ch> - IFJ PAN & AGH, Poland          *
 *      Kamil Kraszewski      <kalq@cern.ch>     - IFJ PAN & UJ, Poland           *
 *      Maciej Kruk           <mkruk@cern.ch>    - IFJ PAN & AGH, Poland          *
 *      Peter Speckmayer      <peter.speckmayer@cern.ch>  - CERN, Switzerland     *
 *      Jan Therhaag          <therhaag@physik.uni-bonn.de> - Uni Bonn, Germany   *
 *                                                                                *
 * Copyright (c) 2008-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      PAN, Poland                                                               *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodLD
#define ROOT_TMVA_MethodLD

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodLD                                                             //
//                                                                      //
// Linear Discriminant                                                  //
// Can compute multidimensional output for regression                   //
// (although it computes every dimension separately)                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#include "TMVA/MethodBase.h"
#include "TMatrixDfwd.h"

namespace TMVA {

   class MethodLD : public MethodBase {

   public:

      // constructor
      MethodLD( const TString& jobName,
                const TString& methodTitle,
                DataSetInfo& dsi,
                const TString& theOption = "LD");

      // constructor
      MethodLD( DataSetInfo& dsi,
                const TString& theWeightFile);

      // destructor
      virtual ~MethodLD( void );

      Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets ) override;

      // training method
      void Train( void ) override;

      // calculate the MVA value
      Double_t GetMvaValue( Double_t* err = nullptr, Double_t* errUpper = nullptr ) override;

      // calculate the Regression value
      const std::vector<Float_t>& GetRegressionValues() override;

      using MethodBase::ReadWeightsFromStream;

      void AddWeightsXMLTo      ( void* parent ) const override;

      void ReadWeightsFromStream( std::istream & i ) override;
      void ReadWeightsFromXML   ( void* wghtnode ) override;

      const Ranking* CreateRanking() override;
      void DeclareOptions() override;
      void ProcessOptions() override;

   protected:

      void MakeClassSpecific( std::ostream&, const TString& ) const override;
      void GetHelpMessage() const override;

   private:

      Int_t fNRegOut; ///< size of the output

      TMatrixD *fSumMatx;              ///< Sum of coordinates product matrix
      TMatrixD *fSumValMatx;           ///< Sum of values multiplied by coordinates
      TMatrixD *fCoeffMatx;            ///< Matrix of coefficients
      std::vector< std::vector<Double_t>* > *fLDCoeff; ///< LD coefficients

      // default initialisation called by all constructors
      void Init( void ) override;

      // Initialization and allocation of matrices
      void InitMatrices( void );

      // Compute fSumMatx
      void GetSum( void );

      // Compute fSumValMatx
      void GetSumVal( void );

      // get LD coefficients
      void GetLDCoeff( void );

      // nice output
      void PrintCoefficients( void );

      ClassDefOverride(MethodLD,0); //Linear discriminant analysis
   };
} // namespace TMVA

#endif // MethodLD_H
