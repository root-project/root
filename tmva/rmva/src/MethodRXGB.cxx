// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRXGB                                                            *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *     R eXtreme Gradient Boosting                                                *
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
#include "TMVA/MethodRXGB.h"
#include "TMVA/Tools.h"
#include "TMVA/Config.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

#include "TMVA/Results.h"

using namespace TMVA;

REGISTER_METHOD(RXGB)

ClassImp(MethodRXGB)

//creating an Instance
Bool_t MethodRXGB::IsModuleLoaded = ROOT::R::TRInterface::Instance().Require("xgboost");

//_______________________________________________________________________
MethodRXGB::MethodRXGB(const TString &jobName,
                       const TString &methodTitle,
                       DataSetInfo &dsi,
                       const TString &theOption,
                       TDirectory *theTargetDir) : RMethodBase(jobName, Types::kRXGB, methodTitle, dsi, theOption, theTargetDir),
   fNRounds(10),
   fEta(0.3),
   fMaxDepth(6),
   predict("predict", "xgboost"),
   xgbtrain("xgboost"),
   xgbdmatrix("xgb.DMatrix"),
   xgbsave("xgb.save"),
   xgbload("xgb.load"),
   asfactor("as.factor"),
   asmatrix("as.matrix"),
   fModel(NULL)
{
   // standard constructor for the RXGB

// default extension for weight files
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);
}

//_______________________________________________________________________
MethodRXGB::MethodRXGB(DataSetInfo &theData, const TString &theWeightFile, TDirectory *theTargetDir)
   : RMethodBase(Types::kRXGB, theData, theWeightFile, theTargetDir),
     fNRounds(10),
     fEta(0.3),
     fMaxDepth(6),
     predict("predict", "xgboost"),
     xgbtrain("xgboost"),
     xgbdmatrix("xgb.DMatrix"),
     xgbsave("xgb.save"),
     xgbload("xgb.load"),
     asfactor("as.factor"),
     asmatrix("as.matrix"),
     fModel(NULL)
{

// default extension for weight files
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);
}


//_______________________________________________________________________
MethodRXGB::~MethodRXGB(void)
{
   if (fModel) delete fModel;
}

//_______________________________________________________________________
Bool_t MethodRXGB::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void     MethodRXGB::Init()
{

   if (!IsModuleLoaded) {
      Error("Init", "R's package xgboost can not be loaded.");
      Log() << kFATAL << " R's package xgboost can not be loaded."
            << Endl;
      return;
   }
   //factors creations
   //xgboost require a numeric factor then background=0 signal=1 from fFactorTrain
   UInt_t size = fFactorTrain.size();
   fFactorNumeric.resize(size);

   for (UInt_t i = 0; i < size; i++) {
      if (fFactorTrain[i] == "signal") fFactorNumeric[i] = 1;
      else fFactorNumeric[i] = 0;
   }



}

void MethodRXGB::Train()
{
   if (Data()->GetNTrainingEvents() == 0) Log() << kFATAL << "<Train> Data() has zero events" << Endl;
   ROOT::R::TRObject dmatrix = xgbdmatrix(ROOT::R::Label["data"] = asmatrix(fDfTrain), ROOT::R::Label["label"] = fFactorNumeric);
   ROOT::R::TRDataFrame params;
   params["eta"] = fEta;
   params["max.depth"] = fMaxDepth;

   SEXP Model = xgbtrain(ROOT::R::Label["data"] = dmatrix,
                         ROOT::R::Label["label"] = fFactorNumeric,
                         ROOT::R::Label["weight"] = fWeightTrain,
                         ROOT::R::Label["nrounds"] = fNRounds,
                         ROOT::R::Label["params"] = params);

   fModel = new ROOT::R::TRObject(Model);
   TString path = GetWeightFileDir() + "/RXGBModel.RData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Saving State File In:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;
   xgbsave(Model, path);
}

//_______________________________________________________________________
void MethodRXGB::DeclareOptions()
{
   DeclareOptionRef(fNRounds, "NRounds", "The max number of iterations");
   DeclareOptionRef(fEta, "Eta", "Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.");
   DeclareOptionRef(fMaxDepth, "MaxDepth", "Maximum depth of the tree");
}

//_______________________________________________________________________
void MethodRXGB::ProcessOptions()
{
}

//_______________________________________________________________________
void MethodRXGB::TestClassification()
{
   Log() << kINFO << "Testing Classification RXGB METHOD  " << Endl;
   MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodRXGB::GetMvaValue(Double_t *errLower, Double_t *errUpper)
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
   if (!fModel) {
      ReadStateFromFile();
   }
   mvaValue = (Double_t)predict(*fModel, xgbdmatrix(ROOT::R::Label["data"] = asmatrix(fDfEvent)));
   return mvaValue;
}

//_______________________________________________________________________
void MethodRXGB::GetHelpMessage() const
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
void TMVA::MethodRXGB::ReadStateFromFile()
{
   ROOT::R::TRInterface::Instance().Require("RXGB");
   TString path = GetWeightFileDir() + "/RXGBModel.RData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Loading State File From:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;

   SEXP Model = xgbload(path);
   fModel = new ROOT::R::TRObject(Model);

}

//_______________________________________________________________________
void TMVA::MethodRXGB::MakeClass(const TString &theClassFileName) const
{
}
