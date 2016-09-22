// @(#)root/tmva $Id$
// Author: Omar Zapata

#include <iostream>

#include "TMVA/Algorithm.h"
#include "TMVA/MethodBase.h"
#include "TMVA/ResultsClassification.h"
#include "TSystem.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TMVA/tmvaglob.h"
#include "TMVA/DataLoader.h"

using namespace TMVA;

//______________________________________________________________________________
Algorithm::Algorithm(const TString &name,DataLoader *dalaloader,TFile *file,const TString options):Configurable(options),fDataLoader(dalaloader),fFile(file),fVerbose(kFALSE)
{
    SetName(name.Data());
}

//______________________________________________________________________________
Algorithm::~Algorithm()
{}

//______________________________________________________________________________
Bool_t  Algorithm::IsSilentFile(){return fFile==nullptr;}

//______________________________________________________________________________
TFile* Algorithm::GetFile(){return fFile.get();}
// TFile* Algorithm::GetFile(){return fFile==nullptr?0:fFile.get();}

//______________________________________________________________________________
void   Algorithm::SetFile(TFile *file){fFile=std::shared_ptr<TFile>(file);}

//______________________________________________________________________________
Bool_t Algorithm::IsVerbose(){return fVerbose;}

//______________________________________________________________________________
void Algorithm::SetVerbose(Bool_t status){fVerbose=status;}

//______________________________________________________________________________
OptionMap &Algorithm::GetMethod(){     return fMethod;}

//______________________________________________________________________________
DataLoader *Algorithm::GetDataLoader(){    return fDataLoader.get();}

//______________________________________________________________________________
void Algorithm::SetDataLoader(DataLoader *dalaloader){
        fDataLoader=std::shared_ptr<DataLoader>(dalaloader) ;
}

//______________________________________________________________________________
Bool_t TMVA::Algorithm::IsModelPersistence(){return fModelPersistence; }

//______________________________________________________________________________
void TMVA::Algorithm::SetModelPersistence(Bool_t status){fModelPersistence=status;}

//______________________________________________________________________________
void TMVA::Algorithm::BookMethod(Types::EMVA method, TString methodTitle, TString options){
    return BookMethod(Types::Instance().GetMethodName( method ),methodTitle,options);
}

//______________________________________________________________________________
void TMVA::Algorithm::BookMethod(TString methodName, TString methodTitle, TString options){
    fMethod["MethodName"]    = methodName;
    fMethod["MethodTitle"]   = methodTitle;
    fMethod["MethodOptions"] = options;
}
