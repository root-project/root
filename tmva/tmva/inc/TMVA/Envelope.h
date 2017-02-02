// @(#)root/tmva:$Id$
// Author: Omar Zapata   2016

/*************************************************************************
 * Copyright (C) 2016, Omar Andres Zapata Mesa                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TMVA_Envelope
#define ROOT_TMVA_Envelope

#include <sstream>
#include<iostream>
#include <memory>

#ifndef ROOT_TROOT
#include<TROOT.h>
#endif

#ifndef ROOT_TStopwatch
#include<TStopwatch.h>
#endif

#ifndef ROOT_TMVA_OptionMap
#include<TMVA/OptionMap.h>
#endif

#ifndef ROOT_TMVA_Config
#include "TMVA/Config.h"
#endif

#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif

#ifndef ROOT_TMVA_DataLoader
#include "TMVA/DataLoader.h"
#endif

#ifndef ROOT_TMVA_ROCCurve
#include "TMVA/DataLoader.h"
#endif


namespace TMVA {

      class Envelope:public Configurable
      {
      protected:
          OptionMap                    fMethod;           //Booked method information
          std::shared_ptr<DataLoader>  fDataLoader;       //data
          std::shared_ptr<TFile>       fFile;             //!file to save the results
          Bool_t                       fModelPersistence; //flag to save the trained model
          Bool_t                       fVerbose;          //flag for extra information

          /**
           Constructor for the initialization of Envelopes,
           differents Envelopes may needs differents constructors then
           this is a generic one protected.
           \param name the name algorithm.
           \param dataloader TMVA::DataLoader object with the data.
           \param file optional file to save the results.
           \param options extra options for the algorithm.
          */

          Envelope(const TString &name,DataLoader *dataloader=nullptr,TFile *file=nullptr,const TString options="");

      public:
          /**
           Default destructor
           */
          ~Envelope();

          /**
            Method to book the machine learning method to perform the algorithm.
            \param methodname String with the name of the mva method
            \param methodtitle String with the method title.
            \param options String with the options for the method.
          */
          virtual void BookMethod( TString methodname, TString methodtitle, TString options = "");
          /**
            Method to book the machine learning method to perform the algorithm.
            \param method enum TMVA::Types::EMVA with the type of the mva method
            \param methodtitle String with the method title.
            \param options String with the options for the method.
          */
          virtual void BookMethod( Types::EMVA method,  TString methodtitle, TString options = "");

          /**
            Method to see if a file is available to save results
            \return Boolean with the status.
          */
          Bool_t  IsSilentFile();
          /**
            Method to get the pointer to TFile object.
            \return pointer to TFile object.
          */
          TFile* GetFile();
          /**
            Method to set the pointer to TFile object,
            with a writable file.
            \param file pointer to TFile object.
          */
          void   SetFile(TFile *file);

          /**
            Method to get the pointer to TMVA::DataLoader object.
            \return  pointer to TMVA::DataLoader object.
          */
          DataLoader *GetDataLoader();

          /**
            Method to set the pointer to TMVA::DataLoader object.
            \param file pointer to TFile object.
          */
          void SetDataLoader(DataLoader *dalaloader);

          /**
            Method to see if the algorithm model is saved in xml or serialized files.
            \return Boolean with the status.
          */
          Bool_t IsModelPersistence();

          /**
            Method enable model persistence, then algorithms model is saved in xml or serialized files.
            \param status Boolean with the status.
          */
          void SetModelPersistence(Bool_t status=kTRUE);

          /**
            Method to see if the algorithm should print extra information.
            \return Boolean with the status.
          */
          Bool_t IsVerbose();

          /**
            Method enable print extra information in the algorithms.
            \param status Boolean with the status.
          */
          void SetVerbose(Bool_t status);

          /**
            Virtual method to be implemented with your algorithm.
          */
          virtual void Evaluate() = 0;

      protected:
          /**
            Method get the Booked method in a option map object.
            \return TMVA::OptionMap with the information of the Booked method
          */
          OptionMap &GetMethod();

          /**
            Utility method to get TMVA::DataInputHandler reference from the DataLoader.
            \return TMVA::DataInputHandler reference.
          */
          DataInputHandler&        GetDataLoaderDataInput() { return *fDataLoader->fDataInputHandler; }

          /**
            Utility method to get TMVA::DataSetInfo reference from the DataLoader.
            \return TMVA::DataSetInfo reference.
          */
          DataSetInfo&             GetDataLoaderDataSetInfo(){return fDataLoader->DefaultDataSetInfo();}

          /**
            Utility method to get TMVA::DataSetManager pointer from the DataLoader.
            \return TMVA::DataSetManager pointer.
          */
          DataSetManager*          GetDataLoaderDataSetManager(){return fDataLoader->fDataSetManager;}
          ClassDef(Envelope,0);

      };
}

#endif
