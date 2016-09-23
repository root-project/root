// @(#)root/tmva $Id$
// Author: Omar Zapata

#include "TMVA/Envelope.h"

#include "TMVA/Configurable.h"
#include "TMVA/DataLoader.h"
#include "TMVA/MethodBase.h"
#include "TMVA/OptionMap.h"
#include "TMVA/ResultsClassification.h"
#include "TMVA/Types.h"

#include "TAxis.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TSystem.h"

#include <iostream>

using namespace TMVA;

//______________________________________________________________________________
Envelope::Envelope(const TString &name,DataLoader *dalaloader,TFile *file,const TString options):Configurable(options),fDataLoader(dalaloader),fFile(file),fVerbose(kFALSE)
{
    SetName(name.Data());
}

//______________________________________________________________________________
Envelope::~Envelope()
{}

//______________________________________________________________________________
Bool_t  Envelope::IsSilentFile(){return fFile==nullptr;}

//______________________________________________________________________________
TFile* Envelope::GetFile(){return fFile.get();}
// TFile* Envelope::GetFile(){return fFile==nullptr?0:fFile.get();}

//______________________________________________________________________________
void   Envelope::SetFile(TFile *file){fFile=std::shared_ptr<TFile>(file);}

//______________________________________________________________________________
Bool_t Envelope::IsVerbose(){return fVerbose;}

//______________________________________________________________________________
void Envelope::SetVerbose(Bool_t status){fVerbose=status;}

//______________________________________________________________________________
OptionMap &Envelope::GetMethod(){     return fMethod;}

//______________________________________________________________________________
DataLoader *Envelope::GetDataLoader(){    return fDataLoader.get();}

//______________________________________________________________________________
void Envelope::SetDataLoader(DataLoader *dalaloader){
        fDataLoader=std::shared_ptr<DataLoader>(dalaloader) ;
}

//______________________________________________________________________________
Bool_t TMVA::Envelope::IsModelPersistence(){return fModelPersistence; }

//______________________________________________________________________________
void TMVA::Envelope::SetModelPersistence(Bool_t status){fModelPersistence=status;}

//______________________________________________________________________________
void TMVA::Envelope::BookMethod(Types::EMVA method, TString methodTitle, TString options){
    return BookMethod(Types::Instance().GetMethodName( method ),methodTitle,options);
}

//______________________________________________________________________________
void TMVA::Envelope::BookMethod(TString methodName, TString methodTitle, TString options){
    fMethod["MethodName"]    = methodName;
    fMethod["MethodTitle"]   = methodTitle;
    fMethod["MethodOptions"] = options;
}
