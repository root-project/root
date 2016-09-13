// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson.


#ifndef ROOT_TMVA_HyperParameterOptimisation
#define ROOT_TMVA_HyperParameterOptimisation


#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TMultiGraph
#include "TMultiGraph.h"
#endif

#ifndef ROOT_TMVA_IMethod
#include "TMVA/IMethod.h"
#endif
#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_Results
#include<TMVA/Results.h>
#endif

#ifndef ROOT_TMVA_Factory
#include<TMVA/Factory.h>
#endif

#ifndef ROOT_TMVA_DataLoader
#include<TMVA/DataLoader.h>
#endif


namespace TMVA {

   class HyperParameterOptimisationResult:public TObject
   {
     friend class HyperParameterOptimisation;
   private:
       std::vector<Float_t> fROCs;
       Float_t              fROCAVG;
       TMultiGraph         *fROCCurves;

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
       HyperParameterOptimisationResult();
       ~HyperParameterOptimisationResult();

       std::vector<std::map<TString,Double_t> > fFoldParameters;
       
       std::vector<Float_t> GetROCValues(){return fROCs;}
       Float_t GetROCAverage(){return fROCAVG;}
       TMultiGraph *GetROCCurves(Bool_t fLegend=kTRUE);

       std::vector<Double_t> GetSigValues(){return fSigs;}
       std::vector<Double_t> GetSepValues(){return fSeps;}
       std::vector<Double_t> GetEff01Values(){return fEff01s;}
       std::vector<Double_t> GetEff10Values(){return fEff10s;}
       std::vector<Double_t> GetEff30Values(){return fEff30s;}
       std::vector<Double_t> GetEffAreaValues(){return fEffAreas;}
       std::vector<Double_t> GetTrainEff01Values(){return fTrainEff01s;}
       std::vector<Double_t> GetTrainEff10Values(){return fTrainEff10s;}
       std::vector<Double_t> GetTrainEff30Values(){return fTrainEff30s;}

//        ClassDef(HyperParameterOptimisationResult,0);  
   };
    
   class HyperParameterOptimisation : public Configurable {
   public:

    //HyperParameterOptimisation();
       HyperParameterOptimisation(DataLoader *loader,TString fomType="Separation", TString fitType="Minuit");
       ~HyperParameterOptimisation();
       
       HyperParameterOptimisationResult* Optimise( TString theMethodName, TString methodTitle, TString theOption = "", int NumFolds = 5);
       //HyperParameterOptimisationResult* Optimise( Types::EMVA theMethod,  TString methodTitle, TString theOption = "", int NumFolds = 5);
       
       inline void SetDataLoader(DataLoader *loader){fDataLoader=loader;}
       inline DataLoader *GetDataLoader(){return fDataLoader;}
       
   private:
       DataLoader *fDataLoader;
       Factory    *fClassifier;
       TString     fFomType;
       TString     fFitType;

   public:
//        ClassDef(HyperParameterOptimisation,1);  
   };
} 


#endif



