// @(#)root/tmva $Id$
// Author: Omar Zapata

/*! \class TMVA::Envelope
\ingroup TMVA

Base class for all machine learning algorithms

*/

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

////////////////////////////////////////////////////////////////////////////////

Envelope::Envelope(const TString &name,DataLoader *dalaloader,TFile *file,const TString options):Configurable(options),fDataLoader(dalaloader),fFile(file),fVerbose(kFALSE)
{
    SetName(name.Data());
}

////////////////////////////////////////////////////////////////////////////////

Envelope::~Envelope()
{}

////////////////////////////////////////////////////////////////////////////////

Bool_t  Envelope::IsSilentFile(){return fFile==nullptr;}

////////////////////////////////////////////////////////////////////////////////

TFile* Envelope::GetFile(){return fFile.get();}
// TFile* Envelope::GetFile(){return fFile==nullptr?0:fFile.get();}

////////////////////////////////////////////////////////////////////////////////

void   Envelope::SetFile(TFile *file){fFile=std::shared_ptr<TFile>(file);}

////////////////////////////////////////////////////////////////////////////////

Bool_t Envelope::IsVerbose(){return fVerbose;}

////////////////////////////////////////////////////////////////////////////////

void Envelope::SetVerbose(Bool_t status){fVerbose=status;}

////////////////////////////////////////////////////////////////////////////////

OptionMap &Envelope::GetMethod(){     return fMethod;}

////////////////////////////////////////////////////////////////////////////////

DataLoader *Envelope::GetDataLoader(){    return fDataLoader.get();}

////////////////////////////////////////////////////////////////////////////////

void Envelope::SetDataLoader(DataLoader *dalaloader){
        fDataLoader=std::shared_ptr<DataLoader>(dalaloader) ;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TMVA::Envelope::IsModelPersistence(){return fModelPersistence; }

////////////////////////////////////////////////////////////////////////////////

void TMVA::Envelope::SetModelPersistence(Bool_t status){fModelPersistence=status;}

////////////////////////////////////////////////////////////////////////////////

void TMVA::Envelope::BookMethod(Types::EMVA method, TString methodTitle, TString options){
    return BookMethod(Types::Instance().GetMethodName( method ),methodTitle,options);
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::Envelope::BookMethod(TString methodName, TString methodTitle, TString options){
    fMethod["MethodName"]    = methodName;
    fMethod["MethodTitle"]   = methodTitle;
    fMethod["MethodOptions"] = options;
}
