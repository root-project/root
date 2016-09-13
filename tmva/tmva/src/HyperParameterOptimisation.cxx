// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson.

#include <iostream>
#include <vector>

#include "TMVA/HyperParameterOptimisation.h"
#include "TMVA/MethodBase.h"
#include "TMVA/ResultsClassification.h"
#include "TSystem.h"
#include "TGraph.h"
#include "TString.h"

//HyperParameterOptimisationResult stuff
// ClassImp(TMVA::HyperParameterOptimisationResult)

TMVA::HyperParameterOptimisationResult::HyperParameterOptimisationResult():TObject()
{
  fROCCurves = new TMultiGraph("ROCCurves","ROCCurves");
  
}

TMVA::HyperParameterOptimisationResult::~HyperParameterOptimisationResult()
{
    if(fROCCurves) delete fROCCurves;
}

TMultiGraph *TMVA::HyperParameterOptimisationResult::GetROCCurves(Bool_t fLegend)
{

  return fROCCurves;
}

//HyperParameterOptimisation class stuff
// ClassImp(TMVA::HyperParameterOptimisation)//serialization is not support yet in so many class TMVA

/*TMVA::HyperParameterOptimisation::HyperParameterOptimisation():Configurable( ),
fDataLoader(0)
{
    fClassifier=new TMVA::Factory("CrossValidation","!V:Silent:Color:DrawProgressBar:AnalysisType=Classification");
    }*/


TMVA::HyperParameterOptimisation::HyperParameterOptimisation(TMVA::DataLoader *loader, TString fomType, TString fitType):Configurable(),
fDataLoader(loader),
fFomType(fomType),
fFitType(fitType)															 
{
    fClassifier=new TMVA::Factory("CrossValidation","!V:Silent:Color:DrawProgressBar:AnalysisType=Classification");    
}

TMVA::HyperParameterOptimisation::~HyperParameterOptimisation()
{
    if(fClassifier) delete fClassifier;
}

TMVA::HyperParameterOptimisationResult* TMVA::HyperParameterOptimisation::Optimise(TString theMethodName, TString methodTitle, TString theOption, int NumFolds)
{
    //TODO by Thomas Stevenson 
    //

  HyperParameterOptimisationResult * result = new HyperParameterOptimisationResult();

  fDataLoader->MakeKFoldDataSet(NumFolds);
  
  for(Int_t i = 0; i < NumFolds; ++i){
    
    TString foldTitle = methodTitle;
    foldTitle += "_opt";
    foldTitle += i+1;
    
    Event::SetIsTraining(kTRUE);

    fDataLoader->PrepareFoldDataSet(i, TMVA::Types::kTraining);
    
    fClassifier->BookMethod(fDataLoader, theMethodName, methodTitle, theOption);

    TMVA::MethodBase * smethod = dynamic_cast<TMVA::MethodBase*>(fClassifier->fMethodsMap[fDataLoader->GetName()][0][0]);

    result->fFoldParameters.push_back(smethod->OptimizeTuningParameters(fFomType,fFitType));

    //smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTesting, Types::kClassification);
    smethod->Data()->DeleteResults(smethod->GetMethodName(), Types::kTraining, Types::kClassification);

    fClassifier->DeleteAllMethods();

    fClassifier->fMethodsMap.clear();

  }

  for(UInt_t j=0; j<result->fFoldParameters.size(); ++j){
    std::cout << "===========================================================" << std::endl;
    std::cout << "Optimisation for " << theMethodName << " fold " << j+1 << std::endl;
  
    std::map<TString,Double_t>::iterator iter;
    for(iter=result->fFoldParameters.at(j).begin(); iter!=result->fFoldParameters.at(j).end(); iter++){
      std::cout << iter->first << "     " << iter->second << std::endl;
    }
  }

  return result;
}
