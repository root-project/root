// @(#)root/tmva/pymva $Id$
// Author: Anirudh Dagar, 2020

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyTorch                                                         *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Interface for PyTorch python based scientific package supporting          *
 *      automatic differentiation for machine learning.                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Anirudh Dagar <anirudhdagar6@gmail.com> - IIT, Roorkee                    *
 *                                                                                *
 * Copyright (c) 2020:                                                            *
 *      CERN, Switzerland                                                         *
 *      IIT, Roorkee                                                              *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPyTorch
#define ROOT_TMVA_MethodPyTorch

#include "TMVA/PyMethodBase.h"
#include <vector>

namespace TMVA {

   class MethodPyTorch : public PyMethodBase {

   public :

      // constructors
      MethodPyTorch(const TString &jobName,
            const TString &methodTitle,
            DataSetInfo &dsi,
            const TString &theOption = "");
      MethodPyTorch(DataSetInfo &dsi,
            const TString &theWeightFile);
      ~MethodPyTorch();

      void Train();
      void Init();
      void DeclareOptions();
      void ProcessOptions();

      // Check whether the given analysis type (regression, classification, ...)
      // is supported by this method
      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t);
      // Get signal probability of given event
      Double_t GetMvaValue(Double_t *errLower, Double_t *errUpper);
      std::vector<Double_t> GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t logProgress);
      // Get regression values of given event
      std::vector<Float_t>& GetRegressionValues();
      // Get all regression values for all the events in the data set
      std::vector<Float_t> GetAllRegressionValues();
      // Get class probabilities of given event
      std::vector<Float_t>& GetMulticlassValues();
      // Get all multiclass values for all the events in the data set
      std::vector<Float_t> GetAllMulticlassValues();

      const Ranking *CreateRanking() { return nullptr; }
      virtual void TestClassification();
      virtual void AddWeightsXMLTo(void*) const{}
      virtual void ReadWeightsFromXML(void*){}
      virtual void ReadWeightsFromStream(std::istream&) {} // backward compatibility
      virtual void ReadWeightsFromStream(TFile&){} // backward compatibility
      void ReadModelFromFile();

      void GetHelpMessage() const;


    private:

      TString fFilenameModel;                         // Filename of the previously exported PyTorch model
      UInt_t fBatchSize {0};                          // Training batch size
      UInt_t fNumEpochs {0};                          // Number of training epochs
      Int_t fNumThreads {0};                          // Number of CPU threads (if 0 uses default values)

      Bool_t fContinueTraining;                       // Load weights from previous training
      Bool_t fSaveBestOnly;                           // Store only weights with smallest validation loss
      TString fLearningRateSchedule;                  // Set new learning rate at specific epochs

      TString fNumValidationString;                   // option string defining the number of validation events

      TString fUserCodeName;                          // filename of the user script that will be executed before loading the PyTorch model

      bool fModelIsSetup = false;                     // flag whether model is loaded, needed for getMvaValue during evaluation
      std::vector<float> fVals;                       // input variables array
      std::vector<float> fOutput;                     // model output array
      UInt_t fNVars {0};                              // number of variables
      UInt_t fNOutputs {0};                           // number of outputs (classes or targets)
      TString fFilenameTrainedModel;                  // output filename for trained model
      PyObject * fPyVals = nullptr;                             // Python array object for input data
      PyObject * fPyOutput = nullptr;                           // Python array object for output data

      void SetupPyTorchModel(Bool_t loadTrainedModel);  // setups the needed variables, loads the model
      void InitEvaluation(size_t nEvents);              // allocate arrays for evaluation
      UInt_t  GetNumValidationSamples();                // get number of validation events according to given option

      ClassDef(MethodPyTorch, 0);
   };

} // namespace TMVA

#endif // ROOT_TMVA_MethodPyTorch
