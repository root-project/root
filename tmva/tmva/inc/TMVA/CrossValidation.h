// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson and Pourya Vakilipourtakalou. 2016


#ifndef ROOT_TMVA_CrossValidation
#define ROOT_TMVA_CrossValidation

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TMultiGraph
#include "TMultiGraph.h"
#endif

#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif

#ifndef ROOT_TMVA_DataLoader
#include<TMVA/DataLoader.h>
#endif

namespace TMVA {

  class Factory;
  class DataLoader;

   class CrossValidationResult
   {
     friend class CrossValidation;
   private:
       std::map<UInt_t,Float_t>        fROCs;       //!
       std::shared_ptr<TMultiGraph>    fROCCurves;  //!
       Float_t              fROCAVG;

       std::vector<Double_t> fSigs;
       std::vector<Double_t> fSeps;
       std::vector<Double_t> fEff01s;
       std::vector<Double_t> fEff10s;
       std::vector<Double_t> fEff30s;
       std::vector<Double_t> fEffAreas;
       std::vector<Double_t> fTrainEff01s;
       std::vector<Double_t> fTrainEff10s;
       std::vector<Double_t> fTrainEff30s;
   public:
       CrossValidationResult();
       CrossValidationResult(const CrossValidationResult &);
       ~CrossValidationResult();
       
       void SetROCValue(UInt_t fold,Float_t rocint);
       
       std::map<UInt_t,Float_t> GetROCValues(){return fROCs;}
       Float_t GetROCAverage() const;
       std::shared_ptr<TMultiGraph> &GetROCCurves();
       void Print() const ;
       
       TCanvas* Draw(const TString name="CrossValidation") const;

       std::vector<Double_t> GetSigValues(){return fSigs;}
       std::vector<Double_t> GetSepValues(){return fSeps;}
       std::vector<Double_t> GetEff01Values(){return fEff01s;}
       std::vector<Double_t> GetEff10Values(){return fEff10s;}
       std::vector<Double_t> GetEff30Values(){return fEff30s;}
       std::vector<Double_t> GetEffAreaValues(){return fEffAreas;}
       std::vector<Double_t> GetTrainEff01Values(){return fTrainEff01s;}
       std::vector<Double_t> GetTrainEff10Values(){return fTrainEff10s;}
       std::vector<Double_t> GetTrainEff30Values(){return fTrainEff30s;}

       //        ClassDef(CrossValidationResult,0);
   };

   class CrossValidation : public Configurable {
   public:

     CrossValidation();
     CrossValidation(DataLoader *loader);
     ~CrossValidation();

     CrossValidationResult* CrossValidate( TString theMethodName, TString methodTitle, TString theOption = "", int NumFolds = 5);
     CrossValidationResult* CrossValidate( Types::EMVA theMethod,  TString methodTitle, TString theOption = "" );

     inline void SetDataLoader(DataLoader *loader){fDataLoader=loader;}
     inline DataLoader *GetDataLoader(){return fDataLoader;}

   private:
     Factory    *fClassifier;
     DataLoader *fDataLoader;

   public:
     //        ClassDef(CrossValidation,1);
   };
} 


#endif



