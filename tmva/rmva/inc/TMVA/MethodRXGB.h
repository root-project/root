// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RMethodRXGB                                                           *
 *                                                                                *
 * Description:                                                                   *
 *      R´s Package xgboost  method based on ROOTR                                *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_RMethodXGB
#define ROOT_TMVA_RMethodXGB

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// RMethodRXGB                                                          //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/RMethodBase.h"
#include <vector>

namespace TMVA {

   class Factory;  // DSMTEST
   class Reader;   // DSMTEST
   class DataSetManager;  // DSMTEST
   class Types;
   class MethodRXGB: public RMethodBase {

   public :

      // constructors
      MethodRXGB(const TString &jobName,
                 const TString &methodTitle,
                 DataSetInfo &theData,
                 const TString &theOption = "");

      MethodRXGB(DataSetInfo &dsi,
                 const TString &theWeightFile);


      ~MethodRXGB(void);
      void Train() override;
      // options treatment
      void Init() override;
      void DeclareOptions() override;
      void ProcessOptions() override;
      // create ranking
      const Ranking *CreateRanking() override
      {
         return nullptr;  // = 0;
      }


      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets) override;

      // performs classifier testing
      void TestClassification() override;


      Double_t GetMvaValue(Double_t *errLower = nullptr, Double_t *errUpper = nullptr) override;
      void MakeClass(const TString &classFileName = TString("")) const override;  //required for model persistence
      using MethodBase::ReadWeightsFromStream;
      // the actual "weights"
      void AddWeightsXMLTo(void * /*parent*/) const override {}  // = 0;
      void ReadWeightsFromXML(void * /*wghtnode*/) override {} // = 0;
      void ReadWeightsFromStream(std::istream &) override {} //= 0;       // backward compatibility

      void ReadModelFromFile();

      // signal/background classification response for all current set of data
      std::vector<Double_t> GetMvaValues(Long64_t firstEvt = 0, Long64_t lastEvt = -1, Bool_t logProgress = false) override;

   private :
      DataSetManager    *fDataSetManager;     // DSMTEST
      friend class Factory;                   // DSMTEST
      friend class Reader;                    // DSMTEST
   protected:


      //RXGBfunction options
      //https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
      UInt_t fNRounds;
      Double_t fEta;
      UInt_t fMaxDepth;
      static Bool_t IsModuleLoaded;

      std::vector<UInt_t>  fFactorNumeric;   //factors creations
      //xgboost  require a numeric factor then background=0 signal=1 from fFactorTrain

      ROOT::R::TRFunctionImport predict;
      ROOT::R::TRFunctionImport xgbtrain;
      ROOT::R::TRFunctionImport xgbdmatrix;
      ROOT::R::TRFunctionImport xgbsave;
      ROOT::R::TRFunctionImport xgbload;
      ROOT::R::TRFunctionImport asfactor;
      ROOT::R::TRFunctionImport asmatrix;
      ROOT::R::TRObject *fModel;


      // get help message text
      void GetHelpMessage() const override;

      ClassDefOverride(MethodRXGB, 0)
   };
} // namespace TMVA
#endif
