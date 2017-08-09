// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson.

#include "TMVA/CrossEvaluation.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
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

// Generalisations
//    Dataloader
//       PrapareForCV( Split s ) <- Split is a generic splitter that fills vectors from incoming vectors 
//                                  (2 -> 2)
//                                  With this idea of Split, we can use new Split(new Split()) to do nesting
//                                  no, i think it better to do PrepareForCV(Split outer, Split inner) since 
//                                  Hmm, the best would be to have a HyperParam() that does what ever and 
//                                  returns a model for use in CE.
//                                  Operating on the model that the current data in the data set is that to 
//                                  be split and that no-one is allowed to modify it until is done. Yikes, 
//                                  feels it would be so much neater with the 
//                                  DSV approach..! (Although there are issues there as well.)
//                                  
//       PrapareForSplit(Uint_t iSplit) <- indep (iFold for K-folds, run rumber for bootstrap)
//       

////////////////////////////////////////////////////////////////////////////////
/// Inherit from configurable
///    CE configs + pass unrecognised options along to the int+ext fact?
///    Internal fact takes extra params, e.g. !ModelPersistance always
///    External takes verbatim from configuration.
///    
///    TODO: fModelPersistence in fClassifier should be dependent on 
///    fModelPersistence in CrossEvaluation.
///    
///    TODO: fJobName for fClassifier and fFactory ("CrossEvaluation")
///    

TMVA::CrossEvaluation::CrossEvaluation(TMVA::DataLoader *dataloader, TFile * outputFile, 
                                       TString splitSpectator, Types::EAnalysisType analysisType)
   : TMVA::Envelope("CrossEvaluation", dataloader),
     fNumFolds(5),
     fSplitSpectator(splitSpectator),
     fAnalysisType(analysisType),
     fClassifier(new TMVA::Factory("CrossEvaluation_internal",
         "!V:!ROC:Silent:ModelPersistence:!Color:!DrawProgressBar:AnalysisType=classification")),
     fFactory(new TMVA::Factory("CrossEvaluation", outputFile, 
         "!V:!ROC:Silent:!ModelPersistence:!Color:!DrawProgressBar:AnalysisType=classification"))
{
   fFoldStatus=kFALSE;

   if (fAnalysisType != Types::kClassification and fAnalysisType != Types::kMulticlass) {
      Log() << kFATAL << "Only binary and multiclass classification supported so far." << Endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossEvaluation::CrossEvaluation(TMVA::DataLoader *dataloader, TString splitSpectator, 
                                       Types::EAnalysisType analysisType)
   : TMVA::Envelope("CrossEvaluation", dataloader),
     fNumFolds(5),
     fSplitSpectator(splitSpectator),
     fAnalysisType(analysisType),
     fClassifier(new TMVA::Factory("CrossEvaluation_internal",
         "!V:!ROC:Silent:ModelPersistence:!Color:!DrawProgressBar:AnalysisType=classification")),
     fFactory(new TMVA::Factory("CrossEvaluation",
         "!V:!ROC:Silent:!ModelPersistence:!Color:!DrawProgressBar:AnalysisType=classification"))
{
   fFoldStatus=kFALSE;

   if (fAnalysisType != Types::kClassification and fAnalysisType != Types::kMulticlass) {
      Log() << kFATAL << "Only binary and multiclass classification supported so far." << Endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
///

TMVA::CrossEvaluation::~CrossEvaluation()
{
   fClassifier=nullptr;
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::SetNumFolds(UInt_t i)
{
   fNumFolds=i;
   fDataLoader->MakeKFoldDataSet(fNumFolds);
   fFoldStatus=kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// TODO: Scale to multiple methods
/// 

void TMVA::CrossEvaluation::StoreResults(MethodBase * smethod) {
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

void TMVA::CrossEvaluation::MergeResults(MethodBase * smethod)
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
   // metaResults = dynamic_cast<ResultsClassification *>(ds->GetResults(methodName, Types::kTraining, analysisType));
   // metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
   // metaResults->GetValueVectorTypes()->insert(metaResults->GetValueVectorTypes()->begin(), classes.begin(), classes.end());

   // TODO: For now this is a copy of the testign set. We might want to inject specific training results here. 
   metaResults = dynamic_cast<ResultsClassification *>(ds->GetResults(methodName, Types::kTesting, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());
   metaResults->GetValueVectorTypes()->insert(metaResults->GetValueVectorTypes()->begin(), classes.begin(), classes.end());
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::StoreResultsMulticlass(MethodBase * smethod)
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

void TMVA::CrossEvaluation::MergeResultsMulticlass(MethodBase * smethod)
{
   DataSet * ds = fDataLoader->GetDataSetInfo().GetDataSet();
   EventOutputsMulticlass_t outputs;
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      outputs.insert(outputs.end(), fOutputsPerFoldMulticlass.at(iFold).begin(), fOutputsPerFoldMulticlass.at(iFold).end());
   }

   TString              methodName   = smethod->GetName();
   Types::EAnalysisType analysisType = smethod->GetAnalysisType();

   ResultsMulticlass * metaResults;
   metaResults = dynamic_cast<ResultsMulticlass *>(ds->GetResults(methodName, Types::kTraining, analysisType));
   metaResults->GetValueVector()->insert(metaResults->GetValueVector()->begin(), outputs.begin(), outputs.end());

   // TODO: For now this is a copy of the testign set. We might want to inject training data here.
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


   fDataLoader->PrepareFoldDataSet(iFold, TMVA::Types::kTraining);
   MethodBase* smethod = fClassifier->BookMethod(fDataLoader.get(), methodName, foldTitle, methodOptions);

   // Train method (train method and eval train set)
   Event::SetIsTraining(kTRUE);
   smethod->TrainMethod();

   // Test method (evaluate the test set)
   Event::SetIsTraining(kFALSE);
   smethod->AddOutput(Types::kTesting, smethod->GetAnalysisType());

   switch (fAnalysisType) {
      case Types::kClassification: StoreResults(smethod); break;
      case Types::kMulticlass    : StoreResultsMulticlass(smethod); break;
      default:
         Log() << kFATAL << "CrossEvaluation currently supports only classification and multiclass classification." << Endl;
         break;
   }

   // Clean-up for this fold
   smethod->Data()->DeleteResults(methodName, Types::kTesting, smethod->GetAnalysisType());
   smethod->Data()->DeleteResults(methodName, Types::kTraining, smethod->GetAnalysisType());
   fClassifier->DeleteAllMethods();
   fClassifier->fMethodsMap.clear();
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::CrossEvaluation::MergeFolds()
{

   TString methodName    = fMethod.GetValue<TString>("MethodName");
   TString methodTitle   = fMethod.GetValue<TString>("MethodTitle");
   TString methodOptions = fMethod.GetValue<TString>("MethodOptions");

   fFactory->BookMethod(fDataLoader.get(), methodName, methodTitle, methodOptions);

   // TODO: This ensures some variables are created as they should. Could be replaced by what?
   fFactory->TrainAllMethods();
   MethodBase * smethod = dynamic_cast<MethodBase *>(fFactory->GetMethod(fDataLoader->GetName(), methodTitle));

   // Merge results from the folds into a single result
   switch (fAnalysisType) {
      case Types::kClassification: MergeResults(smethod); break;
      case Types::kMulticlass    : MergeResultsMulticlass(smethod); break;
      default:
         Log() << kFATAL << "CrossEvaluation currently supports only classification and multiclass classification." << Endl;
         break;
   }

   // Merge inputs 
   fDataLoader->MergeCustomSplit();
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
       fDataLoader->MakeKFoldDataSetCE(fNumFolds, fSplitSpectator);
       fFoldStatus=kTRUE;
   }

   // Process K folds
   for(UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      ProcessFold(iFold);
   }

   // Merge and inject the results into DataSet
   MergeFolds();

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
