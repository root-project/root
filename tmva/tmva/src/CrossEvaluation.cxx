// @(#)root/tmva $Id$
// Author: Kim Albertsson

/*************************************************************************
 * Copyright (C) 2017, Kim Albertsson                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////////
#include "TMVA/CrossEvaluation.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
#include "TMVA/CvSplit.h"
#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodCrossEvaluation.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/ResultsClassification.h"
#include "TMVA/ResultsMulticlass.h"
#include "TMVA/ROCCurve.h"
#include "TMVA/tmvaglob.h"
#include "TMVA/Types.h"

#include "TSystem.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TMath.h"

#include <iostream>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
///    
///    TODO: fJobName for fFoldFactory and fFactory ("CrossEvaluation")
///    
///    TODO: Add optional file to fold factory to save output (for debugging at least).
///    

TMVA::CrossEvaluation::CrossEvaluation(TMVA::DataLoader *dataloader, TFile * outputFile, TString options)
   : TMVA::Envelope("CrossEvaluation", dataloader, nullptr, options),
     fAnalysisType(Types::kMaxAnalysisType),
     fFoldStatus(kFALSE),
     fNumFolds(2),
     fOutputFile(outputFile),
     fSplitSpectator(""),
     fTransformations( "" )
{
   InitOptions();
   ParseOptions();
   CheckForUnusedOptions();

   if (fAnalysisType != Types::kClassification and fAnalysisType != Types::kMulticlass) {
      Log() << kFATAL << "Only binary and multiclass classification supported so far." << Endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossEvaluation::CrossEvaluation(TMVA::DataLoader *dataloader, TString options)
   : CrossEvaluation(dataloader, nullptr, options)
{}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossEvaluation::~CrossEvaluation()
{}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::InitOptions()
{
   // Forwarding of Factory options
   DeclareOptionRef( fSilent,   "Silent", "Batch mode: boolean silent flag inhibiting any output from TMVA after the creation of the factory class object (default: False)" );
   DeclareOptionRef( fVerbose, "V", "Verbose flag" );
   DeclareOptionRef( fVerboseLevel=TString("Info"), "VerboseLevel", "VerboseLevel (Debug/Verbose/Info)" );
   AddPreDefVal(TString("Debug"));
   AddPreDefVal(TString("Verbose"));
   AddPreDefVal(TString("Info"));
   
   DeclareOptionRef( fTransformations, "Transformations", "List of transformations to test; formatting example: \"Transformations=I;D;P;U;G,D\", for identity, decorrelation, PCA, Uniform and Gaussianisation followed by decorrelation transformations" );
   
   TString analysisType("Auto");
   DeclareOptionRef( fAnalysisTypeStr, "AnalysisType", "Set the analysis type (Classification, Regression, Multiclass, Auto) (default: Auto)" );
   AddPreDefVal(TString("Classification"));
   AddPreDefVal(TString("Regression"));
   AddPreDefVal(TString("Multiclass"));
   AddPreDefVal(TString("Auto"));

   // Options specific to CE
   DeclareOptionRef( fSplitSpectator, "SplitSpectator", "The spectator variable to use for the fold splitting" );
   DeclareOptionRef( fNumFolds, "NumFolds", "Number of folds to generate" );
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::ParseOptions()
{
   this->MethodBase::ParseOptions();

   fAnalysisTypeStr.ToLower();
   if     ( fAnalysisTypeStr == "classification" ) fAnalysisType = Types::kClassification;
   else if( fAnalysisTypeStr == "regression" )     fAnalysisType = Types::kRegression;
   else if( fAnalysisTypeStr == "multiclass" )     fAnalysisType = Types::kMulticlass;
   else if( fAnalysisTypeStr == "auto" )           fAnalysisType = Types::kNoAnalysisType;


   TString fCvFactoryOptions = "";
   TString fOutputFactoryOptions = "";
   if (fVerbose) {
      fCvFactoryOptions += "V:";
      fOutputFactoryOptions += "V:";
   } else {
      fCvFactoryOptions += "!V:";
      fOutputFactoryOptions += "!V:";
   }

   fCvFactoryOptions += Form("VerboseLevel=%s:", fVerboseLevel.Data());
   fOutputFactoryOptions += Form("VerboseLevel=%s:", fVerboseLevel.Data());

   fCvFactoryOptions += Form("AnalysisType=%s:", fAnalysisTypeStr.Data());
   fOutputFactoryOptions += Form("AnalysisType=%s:", fAnalysisTypeStr.Data());

   if (fTransformations != "") {
      fCvFactoryOptions += Form("Transformations=%s:", fTransformations.Data());
      fOutputFactoryOptions += Form("Transformations=%s:", fTransformations.Data());
   }

   if (fModelPersistence) {
      fCvFactoryOptions += Form("ModelPersistence:");
   } else {
      fCvFactoryOptions += Form("!ModelPersistence:");
   }

   if (fSilent) {
      fCvFactoryOptions += Form("Silent:");
      fOutputFactoryOptions += Form("Silent:");
   }

   fFoldFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory("CrossEvaluation_internal", fCvFactoryOptions + "!ROC:!Color:!DrawProgressBar"));

   // The fOutputFactory should always have !ModelPersitence set since we use a custom code path for this.
   //    In this case we create a special method (MethodCrossEvaluation) that can only be used by
   //    CrossEvaluation and the Reader.
   if (fOutputFile == nullptr) {
      fFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory("CrossEvaluation",  fOutputFactoryOptions + "!ModelPersistence"));
   } else {
      fFactory = std::unique_ptr<TMVA::Factory>(new TMVA::Factory("CrossEvaluation", fOutputFile,  fOutputFactoryOptions + "!ModelPersistence"));
   }

   fSplit = std::unique_ptr<CvSplitCrossEvaluation>(new CvSplitCrossEvaluation(fNumFolds, fSplitSpectator));
   
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::SetNumFolds(UInt_t i)
{
   if (i != fNumFolds) {
      fNumFolds = i;
      fSplit = std::unique_ptr<CvSplitCrossEvaluation>(new CvSplitCrossEvaluation(fNumFolds, fSplitSpectator));
      fDataLoader->MakeKFoldDataSet(*fSplit.get());
      fFoldStatus=kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::SetSplitSpectator(TString spectatorName)
{
   if (spectatorName != fSplitSpectator) {
      fSplitSpectator = spectatorName;
      fSplit = std::unique_ptr<CvSplitCrossEvaluation>(new CvSplitCrossEvaluation(fNumFolds, fSplitSpectator));
      fDataLoader->MakeKFoldDataSet(*fSplit.get());
      fFoldStatus=kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// 

void TMVA::CrossEvaluation::StoreFoldResults(MethodBase * smethod) {
      DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
      ResultsClassification * resultTestSet =
         dynamic_cast<ResultsClassification *>( ds->GetResults(smethod->GetName(), 
                                                Types::kTesting,
                                                smethod->GetAnalysisType()));

      EventCollection_t evCollection = ds->GetEventCollection(Types::kTesting);

      fOutputsPerFold.push_back( *resultTestSet->GetValueVector()      );
      fClassesPerFold.push_back( *resultTestSet->GetValueVectorTypes() );
}

////////////////////////////////////////////////////////////////////////////////
/// 

void TMVA::CrossEvaluation::ClearFoldResultsCache() {
      fOutputsPerFold.clear();
      fClassesPerFold.clear();
      fOutputsPerFoldMulticlass.clear();
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::MergeFoldResults(MethodBase * smethod)
{
   DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
   EventOutputs_t outputs;
   EventTypes_t classes;
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      outputs.insert(outputs.end(), fOutputsPerFold.at(iFold).begin(), fOutputsPerFold.at(iFold).end());
      classes.insert(classes.end(), fClassesPerFold.at(iFold).begin(), fClassesPerFold.at(iFold).end());
   }

   TString              methodName   = smethod->GetName();
   Types::EAnalysisType analysisType = smethod->GetAnalysisType();

   ResultsClassification * metaResults;

   // For now this is a copy of the testing set. We might want to inject training data here.
   metaResults = dynamic_cast<ResultsClassification *>(ds->GetResults(methodName, Types::kTraining, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
   metaResults->GetValueVectorTypes()->insert(metaResults->GetValueVectorTypes()->begin(), classes.begin(), classes.end());

   metaResults = dynamic_cast<ResultsClassification *>(ds->GetResults(methodName, Types::kTesting, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
   metaResults->GetValueVectorTypes()->insert(metaResults->GetValueVectorTypes()->begin(), classes.begin(), classes.end());
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::StoreFoldResultsMulticlass(MethodBase * smethod)
{
      DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
      ResultsMulticlass * resultTestSet =
         dynamic_cast<ResultsMulticlass *>( ds->GetResults(smethod->GetName(),
                                            Types::kTesting,
                                            smethod->GetAnalysisType()));

      fOutputsPerFoldMulticlass.push_back( *resultTestSet->GetValueVector());
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::MergeFoldResultsMulticlass(MethodBase * smethod)
{
   DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
   EventOutputsMulticlass_t outputs;
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      outputs.insert(outputs.end(), fOutputsPerFoldMulticlass.at(iFold).begin(), fOutputsPerFoldMulticlass.at(iFold).end());
   }

   TString              methodName   = smethod->GetName();
   Types::EAnalysisType analysisType = smethod->GetAnalysisType();

   ResultsMulticlass * metaResults;

   // For now this is a copy of the testing set. We might want to inject training data here.
   metaResults = dynamic_cast<ResultsMulticlass *>(ds->GetResults(methodName, Types::kTraining, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());

   metaResults = dynamic_cast<ResultsMulticlass *>(ds->GetResults(methodName, Types::kTesting, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::ProcessFold(UInt_t iFold)
{
   TString methodName    = fMethod.GetValue<TString>("MethodName");
   TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   Log() << kDEBUG << "Fold (" << methodTitle << "): " << iFold << Endl;

   // Get specific fold of dataset and setup method
   TString foldTitle = methodTitle;
   foldTitle += "_fold";
   foldTitle += iFold+1;


   fDataLoader->PrepareFoldDataSet(*fSplit.get(), iFold, TMVA::Types::kTraining);
   MethodBase* smethod = fFoldFactory->BookMethod(fDataLoader.get(), methodName, foldTitle, methodOptions);

   // Train method (train method and eval train set)
   Event::SetIsTraining(kTRUE);
   smethod->TrainMethod();

   // Test method (evaluate the test set)
   Event::SetIsTraining(kFALSE);
   smethod->AddOutput(Types::kTesting, smethod->GetAnalysisType());

   switch (fAnalysisType) {
      case Types::kClassification: StoreFoldResults(smethod); break;
      case Types::kMulticlass    : StoreFoldResultsMulticlass(smethod); break;
      default:
         Log() << kFATAL << "CrossEvaluation currently supports only classification and multiclass classification." << Endl;
         break;
   }

   // Clean-up for this fold
   smethod->Data()->DeleteResults(foldTitle, Types::kTesting, smethod->GetAnalysisType());
   smethod->Data()->DeleteResults(foldTitle, Types::kTraining, smethod->GetAnalysisType());
   fFoldFactory->DeleteAllMethods();
   fFoldFactory->fMethodsMap.clear();
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::MergeFolds()
{

   TString methodName    = fMethod.GetValue<TString>("MethodName");
   TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   fFactory->BookMethod(fDataLoader.get(), methodName, methodTitle, methodOptions);

   MethodBase * smethod = dynamic_cast<MethodBase *>(fFactory->GetMethod(fDataLoader->GetName(), methodTitle));

   // Write data such as VariableTransformations to output file.
   if (fOutputFile != nullptr) {
      fFactory->WriteDataInformation(smethod->DataInfo());
   }

   // Merge results from the folds into a single result
   switch (fAnalysisType) {
      case Types::kClassification: MergeFoldResults(smethod); break;
      case Types::kMulticlass    : MergeFoldResultsMulticlass(smethod); break;
      default:
         Log() << kFATAL << "CrossEvaluation currently supports only classification and multiclass classification." << Endl;
         break;
   }

   // Merge inputs 
   fDataLoader->RecombineKFoldDataSet( *fSplit.get() );
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::Evaluate()
{
   // TODO: Inform user that the dataloader must be prepared with
   // a train/test split. The test split will be kept aside for
   // final evaluation. If this is not desirable, as in this case,
   // put the training set to 0 size. This could potentially be forced.

   TString methodName  = fMethod.GetValue<TString>("MethodName");
   TString methodTitle = fMethod.GetValue<TString>("MethodTitle");
   if(methodName == "") Log() << kFATAL << "No method booked for cross-validation" << Endl;

   TMVA::MsgLogger::EnableOutput();
   // TMVA::gConfig().SetSilent(kFALSE);
   Log() << kINFO << "Evaluate method: " << methodTitle << Endl;
   // TMVA::gConfig().SetSilent(kTRUE); // Return to prev value?

   // Generate K folds on given dataset
   if(!fFoldStatus){
       fDataLoader->MakeKFoldDataSet(*fSplit.get());
       fFoldStatus=kTRUE;
   }

   // Process K folds
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      ProcessFold(iFold);
   }

   // Merge and inject the results into DataSet
   MergeFolds();
   ClearFoldResultsCache();

   // Run produce final output (e.g. file)
   fFactory->EvaluateAllMethods();

   // Serialise the cross evaluated method
   if (fModelPersistence) {
      // Create new MethodCrossEvaluation
      TString methodCrossEvaluationName = Types::Instance().GetMethodName( Types::kCrossEvaluation );
      IMethod * im = ClassifierFactory::Instance().Create( methodCrossEvaluationName.Data(),
                                                           "", // jobname
                                                           "CrossEvaluation_"+methodTitle,   // title
                                                           fDataLoader->GetDataSetInfo(), // dsi
                                                           "" // options
                                                         ); 

      // Serialise it
      MethodBase * method = dynamic_cast<MethodBase *>(im);

      // Taken directly from what is done in Factory::BookMethod
      TString fFileDir = TString(fDataLoader->GetName()) + "/" + gConfig().GetIONames().fWeightFileDir;
      method->SetWeightFileDir(fFileDir);
      method->SetModelPersistence(fModelPersistence);
      method->SetAnalysisType(fAnalysisType);
      method->SetupMethod();
      method->ParseOptions();
      method->ProcessSetup();
      // method->SetFile(fgTargetFile);
      // method->SetSilentFile(IsSilentFile());

      // check-for-unused-options is performed; may be overridden by derived classes
      method->CheckSetup();

      // Pass info about the correct method name (method_title_base + foldNum)
      // Pass info about the number of folds
      // TODO: Parameterise the internal jobname
      MethodCrossEvaluation * method_ce = dynamic_cast<MethodCrossEvaluation *>(method);
      method_ce->fEncapsulatedMethodName     = "CrossEvaluation_internal_" + methodTitle;
      method_ce->fEncapsulatedMethodTypeName = methodName;
      method_ce->fNumFolds                   = fNumFolds;
      method_ce->fSplitSpectator             = fSplitSpectator;

      method->WriteStateToFile();
      // Not supported by MethodCrossEvaluation yet
      // if (fAnalysisType != Types::kRegression) { smethod->MakeClass(); }
   }

   // TMVA::gConfig().SetSilent(kFALSE);
   Log() << kINFO << "Evaluation done." << Endl;
   // TMVA::gConfig().SetSilent(kTRUE);
}
