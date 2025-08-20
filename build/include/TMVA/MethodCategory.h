// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss,Or Cohen

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCompositeBase                                                   *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker    <Andreas.Hocker@cern.ch> - CERN, Switzerland           *
 *      Joerg Stelzer      <Joerg.Stelzer@cern.ch>  - CERN, Switzerland           *
 *      Peter Speckmayer   <Peter.Speckmayer@cern.ch> - CERN, Switzerland         *
 *      Helge Voss         <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany   *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>        - U of Bonn, Germany          *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCategory
#define ROOT_TMVA_MethodCategory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodCategory                                                       //
//                                                                      //
// Class for categorizing the phase space                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iosfwd>
#include <vector>

#include "TMVA/MethodBase.h"

#include "TMVA/MethodCompositeBase.h"

namespace TMVA {

   class Factory;  // DSMTEST
   class Reader;   // DSMTEST
   class MethodBoost;   // DSMTEST
   class DataSetManager;  // DSMTEST
   namespace Experimental {
   class Classification;
   }
   class MethodCategory : public MethodCompositeBase {
      friend class Experimental::Classification;

   public :

      // constructors
      MethodCategory( const TString& jobName,
                      const TString& methodTitle,
                      DataSetInfo& theData,
                      const TString& theOption = "" );

      MethodCategory( DataSetInfo& dsi,
                      const TString& theWeightFile );

      virtual ~MethodCategory( void );

      Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ ) override;
      // training and boosting all the classifiers
      void Train( void ) override;

      // ranking of input variables
      const Ranking* CreateRanking() override;

      // saves the name and options string of the boosted classifier
      TMVA::IMethod* AddMethod(const TCut&,
                               const TString& theVariables,
                               Types::EMVA theMethod,
                               const TString& theTitle,
                               const TString& theOptions);

      void AddWeightsXMLTo( void* parent ) const override;
      void ReadWeightsFromXML( void* wghtnode ) override;

      Double_t GetMvaValue( Double_t* err = nullptr, Double_t* errUpper = nullptr ) override;

      // regression response
      const std::vector<Float_t>& GetRegressionValues() override;

      // multi class response
      const std::vector<Float_t> &GetMulticlassValues() override;

      void MakeClass( const TString& = TString("") ) const override {};

   protected :

      // signal/background classification response for all current set of data
      std::vector<Double_t> GetMvaValues(Long64_t firstEvt = 0, Long64_t lastEvt = -1, Bool_t logProgress = false) override;

   private:
      // initializing mostly monitoring tools of the category process
      void Init() override;

      // the option handling methods
      void DeclareOptions() override;
      void ProcessOptions() override;

      // build the cut formula for event categorization
      Bool_t PassesCut( const Event* ev, UInt_t methodIdx );

   protected:

      // vectors that contain the added methods and the cuts on which they are to be called
      std::vector<IMethod*>               fMethods;
      std::vector<TCut>                   fCategoryCuts;
      std::vector<UInt_t>                 fCategorySpecIdx;
      std::vector<TString>                fVars;
      std::vector <std::vector <UInt_t> > fVarMaps;

      // get help message text
      void GetHelpMessage() const override;

      TMVA::DataSetInfo& CreateCategoryDSI(const TCut&, const TString&, const TString&);

   private:

      void InitCircularTree(const DataSetInfo& dsi);

      TTree *                    fCatTree; //! needed in conjunction with TTreeFormulas for evaluation category expressions
      std::vector<TTreeFormula*> fCatFormulas;

      DataSetManager* fDataSetManager; // DSMTEST
      friend class Factory; // DSMTEST
      friend class Reader;  // DSMTEST
      friend class MethodBoost;  // DSMTEST

      ClassDefOverride(MethodCategory,0);
   };
}

#endif
