// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015


/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRSNNS                                                           *
 * Web    : http://oproject.org                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Neural Networks in R using the Stuttgart Neural Network Simulator         *
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
#include "TMVA/MethodRSNNS.h"
#include "TMVA/Tools.h"
#include "TMVA/Config.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

#include "TMVA/Results.h"
#include "TMVA/Timer.h"

using namespace TMVA;

REGISTER_METHOD(RSNNS)

ClassImp(MethodRSNNS);

//creating an Instance
Bool_t MethodRSNNS::IsModuleLoaded = ROOT::R::TRInterface::Instance().Require("RSNNS");

//_______________________________________________________________________
MethodRSNNS::MethodRSNNS(const TString &jobName,
                         const TString &methodTitle,
                         DataSetInfo &dsi,
                         const TString &theOption) :
   RMethodBase(jobName, Types::kRSNNS, methodTitle, dsi, theOption),
   fMvaCounter(0),
   predict("predict"),
   mlp("mlp"),
   asfactor("as.factor"),
   fModel(NULL)
{
   fNetType = methodTitle;
   if (fNetType != "RMLP") {
      Log() << kFATAL << " Unknow Method" + fNetType
            << Endl;
      return;
   }

   // standard constructor for the RSNNS
   //RSNNS Options for all NN methods
   fSize = "c(5)";
   fMaxit = 100;

   fInitFunc = "Randomize_Weights";
   fInitFuncParams = "c(-0.3,0.3)"; //the maximun number of pacameter is 5 see RSNNS::getSnnsRFunctionTable() type 6

   fLearnFunc = "Std_Backpropagation"; //
   fLearnFuncParams = "c(0.2,0)";

   fUpdateFunc = "Topological_Order";
   fUpdateFuncParams = "c(0)";

   fHiddenActFunc = "Act_Logistic";
   fShufflePatterns = kTRUE;
   fLinOut = kFALSE;
   fPruneFunc = "NULL";
   fPruneFuncParams = "NULL";

}

//_______________________________________________________________________
MethodRSNNS::MethodRSNNS(DataSetInfo &theData, const TString &theWeightFile)
   : RMethodBase(Types::kRSNNS, theData, theWeightFile),
     fMvaCounter(0),
     predict("predict"),
     mlp("mlp"),
     asfactor("as.factor"),
     fModel(NULL)

{
   fNetType = "RMLP"; //GetMethodName();//GetMethodName() is not returning RMLP is reting MethodBase why?
   if (fNetType != "RMLP") {
      Log() << kFATAL << " Unknow Method = " + fNetType
            << Endl;
      return;
   }

   // standard constructor for the RSNNS
   //RSNNS Options for all NN methods
   fSize = "c(5)";
   fMaxit = 100;

   fInitFunc = "Randomize_Weights";
   fInitFuncParams = "c(-0.3,0.3)"; //the maximun number of pacameter is 5 see RSNNS::getSnnsRFunctionTable() type 6

   fLearnFunc = "Std_Backpropagation"; //
   fLearnFuncParams = "c(0.2,0)";

   fUpdateFunc = "Topological_Order";
   fUpdateFuncParams = "c(0)";

   fHiddenActFunc = "Act_Logistic";
   fShufflePatterns = kTRUE;
   fLinOut = kFALSE;
   fPruneFunc = "NULL";
   fPruneFuncParams = "NULL";
}


//_______________________________________________________________________
MethodRSNNS::~MethodRSNNS(void)
{
   if (fModel) delete fModel;
}

//_______________________________________________________________________
Bool_t MethodRSNNS::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void     MethodRSNNS::Init()
{
   if (!IsModuleLoaded) {
      Error("Init", "R's package RSNNS can not be loaded.");
      Log() << kFATAL << " R's package RSNNS can not be loaded."
            << Endl;
      return;
   }
   //factors creations
   //RSNNS mlp require a numeric factor then background=0 signal=1 from fFactorTrain/fFactorTest
   UInt_t size = fFactorTrain.size();
   fFactorNumeric.resize(size);

   for (UInt_t i = 0; i < size; i++) {
      if (fFactorTrain[i] == "signal") fFactorNumeric[i] = 1;
      else fFactorNumeric[i] = 0;
   }
}

void MethodRSNNS::Train()
{
   if (Data()->GetNTrainingEvents() == 0) Log() << kFATAL << "<Train> Data() has zero events" << Endl;
   if (fNetType == "RMLP") {
      ROOT::R::TRObject PruneFunc;
      if (fPruneFunc == "NULL") PruneFunc = r.Eval("NULL");
      else PruneFunc = r.Eval(Form("'%s'", fPruneFunc.Data()));

      SEXP Model = mlp(ROOT::R::Label["x"] = fDfTrain,
                       ROOT::R::Label["y"] = fFactorNumeric,
                       ROOT::R::Label["size"] = r.Eval(fSize),
                       ROOT::R::Label["maxit"] = fMaxit,
                       ROOT::R::Label["initFunc"] = fInitFunc,
                       ROOT::R::Label["initFuncParams"] = r.Eval(fInitFuncParams),
                       ROOT::R::Label["learnFunc"] = fLearnFunc,
                       ROOT::R::Label["learnFuncParams"] = r.Eval(fLearnFuncParams),
                       ROOT::R::Label["updateFunc"] = fUpdateFunc,
                       ROOT::R::Label["updateFuncParams"] = r.Eval(fUpdateFuncParams),
                       ROOT::R::Label["hiddenActFunc"] = fHiddenActFunc,
                       ROOT::R::Label["shufflePatterns"] = fShufflePatterns,
                       ROOT::R::Label["libOut"] = fLinOut,
                       ROOT::R::Label["pruneFunc"] = PruneFunc,
                       ROOT::R::Label["pruneFuncParams"] = r.Eval(fPruneFuncParams));
      fModel = new ROOT::R::TRObject(Model);
      //if model persistence is enabled saving it is R serialziation.
      if (IsModelPersistence())  
      {
            TString path = GetWeightFileDir() +  "/" + GetName() + ".RData";
            Log() << Endl;
            Log() << gTools().Color("bold") << "--- Saving State File In:" << gTools().Color("reset") << path << Endl;
            Log() << Endl;
            r["RMLPModel"] << Model;
            r << "save(RMLPModel,file='" + path + "')";
      }
   }
}

//_______________________________________________________________________
void MethodRSNNS::DeclareOptions()
{
   //RSNNS Options for all NN methods
//       TVectorF  fSize;//number of units in the hidden layer(s)
   DeclareOptionRef(fSize, "Size", "number of units in the hidden layer(s)");
   DeclareOptionRef(fMaxit, "Maxit", "Maximum of iterations to learn");

   DeclareOptionRef(fInitFunc, "InitFunc", "the initialization function to use");
   DeclareOptionRef(fInitFuncParams, "InitFuncParams", "the parameters for the initialization function");

   DeclareOptionRef(fLearnFunc, "LearnFunc", "the learning function to use");
   DeclareOptionRef(fLearnFuncParams, "LearnFuncParams", "the parameters for the learning function");

   DeclareOptionRef(fUpdateFunc, "UpdateFunc", "the update function to use");
   DeclareOptionRef(fUpdateFuncParams, "UpdateFuncParams", "the parameters for the update function");

   DeclareOptionRef(fHiddenActFunc, "HiddenActFunc", "the activation function of all hidden units");
   DeclareOptionRef(fShufflePatterns, "ShufflePatterns", "should the patterns be shuffled?");
   DeclareOptionRef(fLinOut, "LinOut", "sets the activation function of the output units to linear or logistic");

   DeclareOptionRef(fPruneFunc, "PruneFunc", "the prune function to use");
   DeclareOptionRef(fPruneFuncParams, "PruneFuncParams", "the parameters for the pruning function. Unlike the\
                                                     other functions, these have to be given in a named list. See\
                                                     the pruning demos for further explanation.the update function to use");

}

//_______________________________________________________________________
void MethodRSNNS::ProcessOptions()
{
   if (fMaxit <= 0) {
      Log() << kERROR << " fMaxit <=0... that does not work !! "
            << " I set it to 50 .. just so that the program does not crash"
            << Endl;
      fMaxit = 1;
   }
   // standard constructor for the RSNNS
   //RSNNS Options for all NN methods

}

//_______________________________________________________________________
void MethodRSNNS::TestClassification()
{
   Log() << kINFO << "Testing Classification " << fNetType << " METHOD  " << Endl;

   MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodRSNNS::GetMvaValue(Double_t *errLower, Double_t *errUpper)
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
   if (IsModelPersistence()) ReadModelFromFile();
   
   TVectorD result = predict(*fModel, fDfEvent, ROOT::R::Label["type"] = "prob");
   mvaValue = result[0]; //returning signal prob
   return mvaValue;
}

////////////////////////////////////////////////////////////////////////////////
/// get all the MVA values for the events of the current Data type
std::vector<Double_t> MethodRSNNS::GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t logProgress)
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
   if (IsModelPersistence()) ReadModelFromFile();

   std::vector<Double_t> mvaValues(nEvents);
   ROOT::R::TRObject result = predict(*fModel, evtData, ROOT::R::Label["type"] = "prob");
   //std::vector<Double_t> probValues(2*nEvents);
   mvaValues = result.As<std::vector<Double_t>>(); 
   // assert(probValues.size() == 2*mvaValues.size());
   // std::copy(probValues.begin()+nEvents, probValues.end(), mvaValues.begin() ); 

   if (logProgress) {
      Log() << kINFO <<Form("Dataset[%s] : ",DataInfo().GetName())<< "Elapsed time for evaluation of " << nEvents <<  " events: "
            << timer.GetElapsedTime() << "       " << Endl;
   }

   return mvaValues;

}


//_______________________________________________________________________
void TMVA::MethodRSNNS::ReadModelFromFile()
{
   ROOT::R::TRInterface::Instance().Require("RSNNS");
   TString path = GetWeightFileDir() +  "/" + GetName() + ".RData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Loading State File From:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;
   r << "load('" + path + "')";
   SEXP Model;
   r["RMLPModel"] >> Model;
   fModel = new ROOT::R::TRObject(Model);

}


//_______________________________________________________________________
void MethodRSNNS::GetHelpMessage() const
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

