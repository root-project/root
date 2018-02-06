// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodC50                                                             *
 * Web    : http://oproject.org                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Decision Trees and Rule-Based Models                                      *
 *                                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/

#include <iomanip>

#include "TMath.h"
#include "Riostream.h"
#include "TMatrix.h"
#include "TMatrixD.h"
#include "TVectorD.h"

#include "TMVA/VariableTransformBase.h"
#include "TMVA/MethodC50.h"
#include "TMVA/Tools.h"
#include "TMVA/Config.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

#include "TMVA/Results.h"
#include "TMVA/Timer.h"

using namespace TMVA;

REGISTER_METHOD(C50)

ClassImp(MethodC50);

//creating an Instance
Bool_t MethodC50::IsModuleLoaded = ROOT::R::TRInterface::Instance().Require("C50");

//_______________________________________________________________________
MethodC50::MethodC50(const TString &jobName,
                     const TString &methodTitle,
                     DataSetInfo &dsi,
                     const TString &theOption) : RMethodBase(jobName, Types::kC50, methodTitle, dsi, theOption),
   fNTrials(1),
   fRules(kFALSE),
   fMvaCounter(0),
   predict("predict.C5.0"),
                                                 //predict("predict"),
   C50("C5.0"),
   C50Control("C5.0Control"),
   asfactor("as.factor"),
   fModel(NULL)
{
   // standard constructor for the C50

   //C5.0Control options
   fControlSubset = kTRUE;
   fControlBands = 0;
   fControlWinnow = kFALSE;
   fControlNoGlobalPruning = kFALSE;
   fControlCF = 0.25;
   fControlMinCases = 2;
   fControlFuzzyThreshold = kFALSE;
   fControlSample = 0;
   r["sample.int(4096, size = 1) - 1L"] >> fControlSeed;
   fControlEarlyStopping = kTRUE;

   ListOfVariables = DataInfo().GetListOfVariables();
}

//_______________________________________________________________________
MethodC50::MethodC50(DataSetInfo &theData, const TString &theWeightFile)
   : RMethodBase(Types::kC50, theData, theWeightFile),
     fNTrials(1),
     fRules(kFALSE),
     fMvaCounter(0),
     predict("predict.C5.0"),
     C50("C5.0"),
     C50Control("C5.0Control"),
     asfactor("as.factor"),
     fModel(NULL)
{

   // constructor from weight file
   fControlSubset = kTRUE;
   fControlBands = 0;
   fControlWinnow = kFALSE;
   fControlNoGlobalPruning = kFALSE;
   fControlCF = 0.25;
   fControlMinCases = 2;
   fControlFuzzyThreshold = kFALSE;
   fControlSample = 0;
   r["sample.int(4096, size = 1) - 1L"] >> fControlSeed;
   fControlEarlyStopping = kTRUE;
}


//_______________________________________________________________________
MethodC50::~MethodC50(void)
{
   if (fModel) delete fModel;
}

//_______________________________________________________________________
Bool_t MethodC50::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void     MethodC50::Init()
{

   if (!IsModuleLoaded) {
      Error("Init", "R's package C50 can not be loaded.");
      Log() << kFATAL << " R's package C50 can not be loaded."
            << Endl;
      return;
   }
}

void MethodC50::Train()
{
   if (Data()->GetNTrainingEvents() == 0) Log() << kFATAL << "<Train> Data() has zero events" << Endl;
   SEXP Model = C50(ROOT::R::Label["x"] = fDfTrain, \
                    ROOT::R::Label["y"] = asfactor(fFactorTrain), \
                    ROOT::R::Label["trials"] = fNTrials, \
                    ROOT::R::Label["rules"] = fRules, \
                    ROOT::R::Label["weights"] = fWeightTrain, \
                    ROOT::R::Label["control"] = fModelControl);
   fModel = new ROOT::R::TRObject(Model);
   if (IsModelPersistence())  
   {
        TString path = GetWeightFileDir() +  "/" + GetName() + ".RData";
        Log() << Endl;
        Log() << gTools().Color("bold") << "--- Saving State File In:" << gTools().Color("reset") << path << Endl;
        Log() << Endl;
        r["C50Model"] << Model;
        r << "save(C50Model,file='" + path + "')";
   }
}

//_______________________________________________________________________
void MethodC50::DeclareOptions()
{
   //
   DeclareOptionRef(fNTrials, "NTrials", "An integer specifying the number of boosting iterations");
   DeclareOptionRef(fRules, "Rules", "A logical: should the tree be decomposed into a rule-basedmodel?");

   //C5.0Control Options
   DeclareOptionRef(fControlSubset, "ControlSubset", "A logical: should the model evaluate groups of discrete \
                                      predictors for splits? Note: the C5.0 command line version defaults this \
                                      parameter to ‘FALSE’, meaning no attempted gropings will be evaluated \
                                      during the tree growing stage.");
   DeclareOptionRef(fControlBands, "ControlBands", "An integer between 2 and 1000. If ‘TRUE’, the model orders \
                                     the rules by their affect on the error rate and groups the \
                                     rules into the specified number of bands. This modifies the \
                                     output so that the effect on the error rate can be seen for \
                                     the groups of rules within a band. If this options is \
                                     selected and ‘rules = kFALSE’, a warning is issued and ‘rules’ \
                                     is changed to ‘kTRUE’.");
   DeclareOptionRef(fControlWinnow, "ControlWinnow", "A logical: should predictor winnowing (i.e feature selection) be used?");
   DeclareOptionRef(fControlNoGlobalPruning, "ControlNoGlobalPruning", "A logical to toggle whether the final, global pruning \
                                                                         step to simplify the tree.");
   DeclareOptionRef(fControlCF, "ControlCF", "A number in (0, 1) for the confidence factor.");
   DeclareOptionRef(fControlMinCases, "ControlMinCases", "an integer for the smallest number of samples that must be \
                                                           put in at least two of the splits.");

   DeclareOptionRef(fControlFuzzyThreshold, "ControlFuzzyThreshold", "A logical toggle to evaluate possible advanced splits \
                                                                      of the data. See Quinlan (1993) for details and examples.");
   DeclareOptionRef(fControlSample, "ControlSample", "A value between (0, .999) that specifies the random \
                                                       proportion of the data should be used to train the model. By \
                                                       default, all the samples are used for model training. Samples \
                                                       not used for training are used to evaluate the accuracy of \
                                                       the model in the printed output.");
   DeclareOptionRef(fControlSeed, "ControlSeed", " An integer for the random number seed within the C code.");
   DeclareOptionRef(fControlEarlyStopping, "ControlEarlyStopping", " A logical to toggle whether the internal method for \
                                                                      stopping boosting should be used.");


}

//_______________________________________________________________________
void MethodC50::ProcessOptions()
{
   if (fNTrials <= 0) {
      Log() << kERROR << " fNTrials <=0... that does not work !! "
            << " I set it to 1 .. just so that the program does not crash"
            << Endl;
      fNTrials = 1;
   }
   fModelControl = C50Control(ROOT::R::Label["subset"] = fControlSubset, \
                              ROOT::R::Label["bands"] = fControlBands, \
                              ROOT::R::Label["winnow"] = fControlWinnow, \
                              ROOT::R::Label["noGlobalPruning"] = fControlNoGlobalPruning, \
                              ROOT::R::Label["CF"] = fControlCF, \
                              ROOT::R::Label["minCases"] = fControlMinCases, \
                              ROOT::R::Label["fuzzyThreshold"] = fControlFuzzyThreshold, \
                              ROOT::R::Label["sample"] = fControlSample, \
                              ROOT::R::Label["seed"] = fControlSeed, \
                              ROOT::R::Label["earlyStopping"] = fControlEarlyStopping);
}

//_______________________________________________________________________
void MethodC50::TestClassification()
{
   Log() << kINFO << "Testing Classification C50 METHOD  " << Endl;
   MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodC50::GetMvaValue(Double_t *errLower, Double_t *errUpper)
{
   NoErrorCalc(errLower, errUpper);
   Double_t mvaValue;
   const TMVA::Event *ev = GetEvent();
   const UInt_t nvar = DataInfo().GetNVariables();
   ROOT::R::TRDataFrame fDfEvent;
   for (UInt_t i = 0; i < nvar; i++) {
      fDfEvent[DataInfo().GetListOfVariables()[i].Data()] = ev->GetValues()[i];
   }
   //if using persistence model
   if (IsModelPersistence())  ReadStateFromFile();
   
   TVectorD result = predict(*fModel, fDfEvent, ROOT::R::Label["type"] = "prob");
   mvaValue = result[1]; //returning signal prob
   return mvaValue;
}


////////////////////////////////////////////////////////////////////////////////
/// get all the MVA values for the events of the current Data type
std::vector<Double_t> MethodC50::GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t logProgress)
{
   Long64_t nEvents = Data()->GetNEvents();
   if (firstEvt > lastEvt || lastEvt > nEvents) lastEvt = nEvents;
   if (firstEvt < 0) firstEvt = 0;

   nEvents = lastEvt-firstEvt; 

   UInt_t nvars = Data()->GetNVariables();

   // use timer
   Timer timer( nEvents, GetName(), kTRUE );
   if (logProgress) 
      Log() << kINFO<<Form("Dataset[%s] : ",DataInfo().GetName())<< "Evaluation of " << GetMethodName() << " on "
            << (Data()->GetCurrentType()==Types::kTraining?"training":"testing") << " sample (" << nEvents << " events)" << Endl;
 

   // fill R DATA FRAME with events data
   std::vector<std::vector<Float_t> > inputData(nvars);
   for (UInt_t i = 0; i < nvars; i++) {
      inputData[i] =  std::vector<Float_t>(nEvents); 
   }
   
   for (Int_t ievt=firstEvt; ievt<lastEvt; ievt++) {
     Data()->SetCurrentEvent(ievt);
      const TMVA::Event *e = Data()->GetEvent();
      assert(nvars == e->GetNVariables());
      for (UInt_t i = 0; i < nvars; i++) {
         inputData[i][ievt] = e->GetValue(i);
      }
      // if (ievt%100 == 0)
      //    std::cout << "Event " << ievt << "  type" << DataInfo().IsSignal(e) << " : " << pValue[ievt*nvars] << "  " << pValue[ievt*nvars+1] << "  " << pValue[ievt*nvars+2] << std::endl;
   }

   ROOT::R::TRDataFrame evtData;
   for (UInt_t i = 0; i < nvars; i++) {
      evtData[DataInfo().GetListOfVariables()[i].Data()] = inputData[i];
   }
   //if using persistence model
   if (IsModelPersistence())  ReadModelFromFile();

   std::vector<Double_t> mvaValues(nEvents);
   ROOT::R::TRObject result = predict(*fModel, evtData, ROOT::R::Label["type"] = "prob");
   std::vector<Double_t> probValues(2*nEvents);
   probValues = result.As<std::vector<Double_t>>(); 
   assert(probValues.size() == 2*mvaValues.size());
   std::copy(probValues.begin()+nEvents, probValues.end(), mvaValues.begin() ); 

   if (logProgress) {
      Log() << kINFO <<Form("Dataset[%s] : ",DataInfo().GetName())<< "Elapsed time for evaluation of " << nEvents <<  " events: "
            << timer.GetElapsedTime() << "       " << Endl;
   }

   return mvaValues;

}

//_______________________________________________________________________
void MethodC50::GetHelpMessage() const
{
// get help message text
//
// typical length of text line:
//         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "Decision Trees and Rule-Based Models " << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "<None>" << Endl;
}

//_______________________________________________________________________
void TMVA::MethodC50::ReadModelFromFile()
{
   ROOT::R::TRInterface::Instance().Require("C50");
   TString path = GetWeightFileDir() +  "/" + GetName() + ".RData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Loading State File From:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;
   r << "load('" + path + "')";
   SEXP Model;
   r["C50Model"] >> Model;
   fModel = new ROOT::R::TRObject(Model);

}

//_______________________________________________________________________
void TMVA::MethodC50::MakeClass(const TString &/*theClassFileName*/) const
{
}
