// Author: Kim Albertsson

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCrossEvaluation                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::MethodCrossEvaluation
\ingroup TMVA
*/
#include "TMVA/MethodCrossEvaluation.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
#include "TMVA/CvSplit.h"
#include "TMVA/MethodCategory.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"

#include "TSystem.h"

REGISTER_METHOD(CrossEvaluation)

ClassImp(TMVA::MethodCrossEvaluation);

////////////////////////////////////////////////////////////////////////////////
/// 

TMVA::MethodCrossEvaluation::MethodCrossEvaluation( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData,
                            const TString& theOption ) :
   TMVA::MethodBase( jobName, Types::kCrossEvaluation, methodTitle, theData, theOption)
{
}

////////////////////////////////////////////////////////////////////////////////

TMVA::MethodCrossEvaluation::MethodCrossEvaluation( DataSetInfo& theData,
                            const TString& theWeightFile)
   : TMVA::MethodBase( Types::kCrossEvaluation, theData, theWeightFile)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
///

TMVA::MethodCrossEvaluation::~MethodCrossEvaluation( void )
{
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::MethodCrossEvaluation::DeclareOptions()
{
   DeclareOptionRef( fSplitExprString, "SplitExpr", "The expression used to assign events to folds" );
   DeclareOptionRef( fNumFolds, "NumFolds", "Number of folds to generate" );
   DeclareOptionRef( fEncapsulatedMethodName, "EncapsulatedMethodName", "");
   DeclareOptionRef( fEncapsulatedMethodTypeName, "EncapsulatedMethodTypeName", "");
}

////////////////////////////////////////////////////////////////////////////////
/// Options that are used ONLY for the READER to ensure backward compatibility.

void TMVA::MethodCrossEvaluation::DeclareCompatibilityOptions() {
   MethodBase::DeclareCompatibilityOptions();
}

////////////////////////////////////////////////////////////////////////////////
/// The option string is decoded, for available options see "DeclareOptions".

void TMVA::MethodCrossEvaluation::ProcessOptions()
{
   Log() << kINFO << "ProcessOptions -- fNumFolds: " << fNumFolds << Endl;
   Log() << kINFO << "ProcessOptions -- fEncapsulatedMethodName: " << fEncapsulatedMethodName << Endl;
   Log() << kINFO << "ProcessOptions -- fEncapsulatedMethodTypeName: " << fEncapsulatedMethodTypeName << Endl;
   // TODO: Validate fNumFolds
   // TODO: Validate fEncapsulatedMethodName
   // TODO: Validate fEncapsulatedMethodTypeName

   fSplitExpr = std::unique_ptr<CvSplitCrossEvaluationExpr>(new CvSplitCrossEvaluationExpr(DataInfo(), fSplitExprString));

   // TODO: To private method. DRY.
   for (UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      TString foldStr = Form( "Fold%i", iFold+1 );

      TString fileDir = GetWeightFileDir();
      TString weightfile  = fileDir + "/" + fJobName + "_" + fEncapsulatedMethodName + "_" + foldStr + ".weights.xml";

      Log() << kINFO << "Reading weightfile: " << weightfile << Endl;

      fEncapsulatedMethods.push_back(InstantiateMethodFromXML(fEncapsulatedMethodTypeName, weightfile));
   }

   // TODO: Verify that a method was instantiated
}

////////////////////////////////////////////////////////////////////////////////
/// Common initialisation with defaults for the Method.

void TMVA::MethodCrossEvaluation::Init( void )
{
}


////////////////////////////////////////////////////////////////////////////////
/// Reset the method, as if it had just been instantiated (forget all training etc.).

void TMVA::MethodCrossEvaluation::Reset( void )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Call the Optimizer with the set of parameters and ranges that
/// are meant to be tuned.

// std::map<TString,Double_t>  TMVA::MethodCrossEvaluation::OptimizeTuningParameters(TString fomType, TString fitType)
// {
// }

////////////////////////////////////////////////////////////////////////////////
/// Set the tuning parameters according to the argument.

// void TMVA::MethodCrossEvaluation::SetTuneParameters(std::map<TString,Double_t> tuneParameters)
// {
// }

////////////////////////////////////////////////////////////////////////////////
///  training.

void TMVA::MethodCrossEvaluation::Train()
{

}

////////////////////////////////////////////////////////////////////////////////
///
   
TMVA::MethodBase * TMVA::MethodCrossEvaluation::InstantiateMethodFromXML(TString methodTypeName, TString weightfile) const
{
       // recreate
      TMVA::MethodBase * m = dynamic_cast<MethodBase*>( ClassifierFactory::Instance()
                                      .Create(std::string(methodTypeName), DataInfo(), weightfile)
                                    );

      
      // TODO: We need to get a datasetmanager in here somehow       
      // if( m->GetMethodType() == Types::kCategory ){
      //   MethodCategory *methCat = (dynamic_cast<MethodCategory*>(m));
      //   if( !methCat ) {
      //      Log() << kFATAL << "Method with type kCategory cannot be casted to MethodCategory." << Endl;
      //   } else {
      //      methCat->fDataSetManager = DataInfo().GetDataSetManager();
      //   }
      // }

      // TODO: Should fFileDir not contain the correct value already?
      TString fileDir= DataInfo().GetName();
      fileDir += "/" + gConfig().GetIONames().fWeightFileDir;
      m->SetWeightFileDir(fileDir);
      // m->SetModelPersistence(fModelPersistence);
      // m->SetSilentFile(IsSilentFile());
      m->SetAnalysisType(fAnalysisType);
      m->SetupMethod();
      m->ReadStateFromFile();
      // m->SetTestvarName(testvarName);
      
      return m;
}

////////////////////////////////////////////////////////////////////////////////
/// Write weights to XML.

void TMVA::MethodCrossEvaluation::AddWeightsXMLTo( void* parent ) const
{  
   void* wght = gTools().AddChild(parent, "Weights");

   //TODO: Options in optionstring are handled auto. Just add these there.
   gTools().AddAttr( wght, "JobName", fJobName );
   gTools().AddAttr( wght, "SplitExpr", fSplitExprString );
   gTools().AddAttr( wght, "NumFolds", fNumFolds );
   gTools().AddAttr( wght, "EncapsulatedMethodName", fEncapsulatedMethodName );
   gTools().AddAttr( wght, "EncapsulatedMethodTypeName", fEncapsulatedMethodTypeName );


   for (UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      TString foldStr     = Form("fold%i", iFold+1);
      TString fileDir= DataInfo().GetName();
      fileDir += "/" + gConfig().GetIONames().fWeightFileDir;
      TString weightfile  = fileDir + "/" + fEncapsulatedMethodName + "_" + foldStr + ".weights.xml";

      // TODO: Add a swithch in options for using either split files or only one.
      // TODO: This would store the method inside MethodCrossEvaluation
      //       Another option is to store the folds as separate files.
      // //Retrieve encap. method for fold n
      // MethodBase * method = InstantiateMethodFromXML(fEncapsulatedMethodTypeName, weightfile);

      // // Serialise encapsulated method for fold n
      // void* foldNode = gTools().AddChild(parent, foldStr);
      // method->WriteStateToXML(foldNode);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Reads from the xml file.
/// 

void TMVA::MethodCrossEvaluation::ReadWeightsFromXML(void* parent)
{
   gTools().ReadAttr( parent, "JobName", fJobName );
   gTools().ReadAttr( parent, "SplitExpr", fSplitExprString );
   gTools().ReadAttr( parent, "NumFolds", fNumFolds );
   gTools().ReadAttr( parent, "EncapsulatedMethodName", fEncapsulatedMethodName );
   gTools().ReadAttr( parent, "EncapsulatedMethodTypeName", fEncapsulatedMethodTypeName );

   for (UInt_t iFold = 0; iFold < fNumFolds; ++iFold){
      TString foldStr = Form( "Fold%i", iFold+1 );
      // void* foldNode = gTools().GetChild(parent, foldStr);
      // if (foldNode == nullptr) {
      //    Log() << kFATAL << "Malformed data. Expected tag \"" << foldStr << "\" to exist." << Endl;
      //    return;
      // }

      TString fileDir = gSystem->DirName(GetWeightFileName());
      TString weightfile  = fileDir + "/" + fJobName + "_" + fEncapsulatedMethodName + "_" + foldStr + ".weights.xml";

      Log() << kDEBUG << "Reading weightfile: " << weightfile << std::endl;

      fEncapsulatedMethods.push_back(InstantiateMethodFromXML(fEncapsulatedMethodTypeName, weightfile));
   }

   fSplitExpr = std::unique_ptr<CvSplitCrossEvaluationExpr>(new CvSplitCrossEvaluationExpr(DataInfo(), fSplitExprString));
}

////////////////////////////////////////////////////////////////////////////////
/// Read the weights
/// 

void  TMVA::MethodCrossEvaluation::ReadWeightsFromStream( std::istream& /*istr*/ )
{
   Log() << kFATAL << "CrossEvaluation currently supports only reading from XML." << Endl;
}

////////////////////////////////////////////////////////////////////////////////
///

Double_t TMVA::MethodCrossEvaluation::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   const Event* ev = GetEvent();
   // auto val = ev->GetSpectator(fIdxSpec);
   // UInt_t iFold = (UInt_t)val % (UInt_t)fNumFolds;
   UInt_t iFold = fSplitExpr->Eval(fNumFolds, ev);

   return fEncapsulatedMethods.at(iFold)->GetMvaValue(err, errUpper);
}

////////////////////////////////////////////////////////////////////////////////
/// Get the multiclass MVA response.

const std::vector<Float_t> & TMVA::MethodCrossEvaluation::GetMulticlassValues()
{
   const Event *ev = GetEvent();
   // auto val = ev->GetSpectator(fIdxSpec);
   // UInt_t iFold = (UInt_t)val % (UInt_t)fNumFolds;
   UInt_t iFold = fSplitExpr->Eval(fNumFolds, ev);

   return fEncapsulatedMethods.at(iFold)->GetMulticlassValues();
}

////////////////////////////////////////////////////////////////////////////////
/// Get the regression value generated by the containing methods.

const std::vector<Float_t> & TMVA::MethodCrossEvaluation::GetRegressionValues()
{
   Log() << kFATAL << "Regression not implemented for CrossEvaluation" << Endl;
   return fNotImplementedRetValVec;
}

////////////////////////////////////////////////////////////////////////////////
///

void  TMVA::MethodCrossEvaluation::WriteMonitoringHistosToFile( void ) const
{
   // // Used for evaluation, which is outside the life time of MethodCrossEval.
   // Log() << kFATAL << "Method CrossEvaluation should not be created manually,"
   //                    " only as part of using TMVA::Reader." << Endl;
   // return;
}

////////////////////////////////////////////////////////////////////////////////
///

void TMVA::MethodCrossEvaluation::GetHelpMessage() const
{
   Log() << kWARNING << "Method CrossEvaluation should not be created manually,"
                      " only as part of using TMVA::Reader." << Endl;
}

////////////////////////////////////////////////////////////////////////////////
///

const TMVA::Ranking * TMVA::MethodCrossEvaluation::CreateRanking()
{
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TMVA::MethodCrossEvaluation::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets )
{
   return kTRUE;
   // if (fEncapsulatedMethods.size() == 0) {return kFALSE;}
   // if (fEncapsulatedMethods.at(0) == nullptr) {return kFALSE;}
   // return fEncapsulatedMethods.at(0)->HasAnalysisType(type, numberClasses, numberTargets);
}

////////////////////////////////////////////////////////////////////////////////
/// Make ROOT-independent C++ class for classifier response (classifier-specific implementation).

void TMVA::MethodCrossEvaluation::MakeClassSpecific( std::ostream& /*fout*/, const TString& /*className*/ ) const
{
   Log() << kWARNING << "MakeClassSpecific not implemented for CrossEvaluation" << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Specific class header.

void TMVA::MethodCrossEvaluation::MakeClassSpecificHeader(  std::ostream& /*fout*/, const TString& /*className*/) const
{
   Log() << kWARNING << "MakeClassSpecificHeader not implemented for CrossEvaluation" << Endl;
}

