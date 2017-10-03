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

//_______________________________________________________________________
/**
Constructor for the initialization of Envelopes,
differents Envelopes may needs differents constructors then
this is a generic one protected.
\param name the name algorithm.
\param dataloader TMVA::DataLoader object with the data.
\param file optional file to save the results.
\param options extra options for the algorithm.
*/
Envelope::Envelope(const TString &name,DataLoader *dalaloader,TFile *file,const TString options):Configurable(options),fDataLoader(dalaloader),fFile(file),fVerbose(kFALSE)
{
    SetName(name.Data());
    fModelPersistence = kTRUE;
}

//_______________________________________________________________________
Envelope::~Envelope()
{}

//_______________________________________________________________________
/**
Method to see if a file is available to save results
\return Boolean with the status.
*/
Bool_t  Envelope::IsSilentFile(){return fFile==nullptr;}

//_______________________________________________________________________
/**
Method to get the pointer to TFile object.
\return pointer to TFile object.
*/
TFile* Envelope::GetFile(){return fFile.get();}

//_______________________________________________________________________
/**
Method to set the pointer to TFile object,
with a writable file.
\param file pointer to TFile object.
*/
void   Envelope::SetFile(TFile *file){fFile=std::shared_ptr<TFile>(file);}

//_______________________________________________________________________
/**
Method to see if the algorithm should print extra information.
\return Boolean with the status.
*/
Bool_t Envelope::IsVerbose(){return fVerbose;}

//_______________________________________________________________________
/**
Method enable print extra information in the algorithms.
\param status Boolean with the status.
*/
void Envelope::SetVerbose(Bool_t status){fVerbose=status;}

//_______________________________________________________________________
/**
Method get the Booked methods in a option map object.
\return vector of TMVA::OptionMap objects with the information of the Booked method
*/
std::vector<OptionMap> &Envelope::GetMethods()
{
   return fMethods;
}

//_______________________________________________________________________
/**
Method to get the pointer to TMVA::DataLoader object.
\return  pointer to TMVA::DataLoader object.
*/

DataLoader *Envelope::GetDataLoader(){    return fDataLoader.get();}

//_______________________________________________________________________
/**
Method to set the pointer to TMVA::DataLoader object.
\param dalaloader pointer to TMVA::DataLoader object.
*/
void Envelope::SetDataLoader(DataLoader *dalaloader){
        fDataLoader=std::shared_ptr<DataLoader>(dalaloader) ;
}

//_______________________________________________________________________
/**
Method to see if the algorithm model is saved in xml or serialized files.
\return Boolean with the status.
*/
Bool_t TMVA::Envelope::IsModelPersistence(){return fModelPersistence; }

//_______________________________________________________________________
/**
Method enable model persistence, then algorithms model is saved in xml or serialized files.
\param status Boolean with the status.
*/
void TMVA::Envelope::SetModelPersistence(Bool_t status){fModelPersistence=status;}

//_______________________________________________________________________
/**
Method to book the machine learning method to perform the algorithm.
\param method enum TMVA::Types::EMVA with the type of the mva method
\param methodtitle String with the method title.
\param options String with the options for the method.
*/
void TMVA::Envelope::BookMethod(Types::EMVA method, TString methodTitle, TString options){
   BookMethod(Types::Instance().GetMethodName(method), methodTitle, options);
}

//_______________________________________________________________________
/**
Method to book the machine learning method to perform the algorithm.
\param methodname String with the name of the mva method
\param methodtitle String with the method title.
\param options String with the options for the method.
*/
void TMVA::Envelope::BookMethod(TString methodName, TString methodTitle, TString options){
   for (auto &meth : fMethods) {
      if (meth.GetValue<TString>("MethodName") == methodName && meth.GetValue<TString>("MethodTitle") == methodTitle) {
         Log() << kFATAL << "Booking failed since method with title <" << methodTitle << "> already exists "
               << "in with DataSet Name <" << fDataLoader->GetName() << ">  " << Endl;
      }
   }
   OptionMap fMethod;
   fMethod["MethodName"] = methodName;
   fMethod["MethodTitle"] = methodTitle;
   fMethod["MethodOptions"] = options;

   fMethods.push_back(fMethod);
}
