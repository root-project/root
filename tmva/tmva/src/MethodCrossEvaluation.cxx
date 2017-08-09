// Author: Kim Albertsson

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCrossEvaluation                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Doug Schouten   <dschoute@sfu.ca>        - Simon Fraser U., Canada        *
 *      Jan Therhaag    <jan.therhaag@cern.ch>   - U. of Bonn, Germany            *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>        - U of Bonn, Germany       *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
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
#include "TMVA/MethodCategory.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"

#include "TSystem.h"

REGISTER_METHOD(CrossEvaluation)

ClassImp(TMVA::MethodCrossEvaluation);

// TODO: 
//    Organise after life time?
//       Construction - Train - Eval - Application

const Int_t TMVA::MethodCrossEvaluation::fgDebugLevel = 0;

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
   gTools().AddAttr( wght, "SplitSpectator", fSplitSpectator );
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
   std::cout << __FILE__ << __LINE__ << std::endl;

   gTools().ReadAttr( parent, "SplitSpectator", fSplitSpectator );
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
      TString weightfile  = fileDir + "/" + fEncapsulatedMethodName + "_" + foldStr + ".weights.xml";

      std::cout << "Reading weightfile: " << weightfile << std::endl;

      fEncapsulatedMethods.push_back(InstantiateMethodFromXML(fEncapsulatedMethodTypeName, weightfile));
   }

   // TODO: This is init of a variable, should it be here?
   // No, it should be in Init method which is run after Options are parsed :)
   std::vector<VariableInfo> spectatorInfos = DataInfo().GetSpectatorInfos();
   fIdxSpec = -1;
   for (UInt_t iSpectator = 0; iSpectator < spectatorInfos.size(); ++iSpectator) {
      VariableInfo vi = spectatorInfos[iSpectator];
      if (vi.GetName() == fSplitSpectator) {
         fIdxSpec = iSpectator;
         break;
      } else if (vi.GetLabel() == fSplitSpectator) {
         fIdxSpec = iSpectator;
         break;
      } else if (vi.GetExpression() == fSplitSpectator) {
         fIdxSpec = iSpectator;
         break;
      }
   };

   Log() << kDEBUG << "Spectator variable\"" << fSplitSpectator << "\" has index: " << fIdxSpec << Endl;

   if (fIdxSpec == -1) {
      Log() << kFATAL << "Spectator variable\"" << fSplitSpectator << "\" not found." << Endl;
      return;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Read the weights
/// 

void  TMVA::MethodCrossEvaluation::ReadWeightsFromStream( std::istream& istr )
{

}

////////////////////////////////////////////////////////////////////////////////
///

Double_t TMVA::MethodCrossEvaluation::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   const Event* ev = GetEvent();
   auto val = ev->GetSpectator(fIdxSpec);
   UInt_t iFold = (UInt_t)val % (UInt_t)fNumFolds;

   return fEncapsulatedMethods.at(iFold)->GetMvaValue(err, errUpper);
}

////////////////////////////////////////////////////////////////////////////////
/// Get the multiclass MVA response.

const std::vector<Float_t> & TMVA::MethodCrossEvaluation::GetMulticlassValues()
{
   Log() << kFATAL << "Multiclass not implemented for CrossEvaluation" << Endl;
   return fNotImplementedRetValVec;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the regression value generated by the containing methods.

const std::vector<Float_t> & TMVA::MethodCrossEvaluation::GetRegressionValues()
{
   Log() << kFATAL << "Regression not implemented for CrossEvaluation" << Endl;
   return fNotImplementedRetValVec;
}

////////////////////////////////////////////////////////////////////////////////
/// Here we could write some histograms created during the processing
/// to the output file.

void  TMVA::MethodCrossEvaluation::WriteMonitoringHistosToFile( void ) const
{
   // Used for evaluation, which is outside the life time of MethodCrossEval.
   Log() << kFATAL << "Method CrossEvaluation should not be created manually,"
                      " only as part of using TMVA::Reader." << Endl;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Get help message text.

void TMVA::MethodCrossEvaluation::GetHelpMessage() const
{
   // Log() << Endl;
   // Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   // Log() << Endl;
   // Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   // Log() << Endl;
   // Log() << "By the nature of the binary splits performed on the individual" << Endl;
   // Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   // Log() << Endl;
   // Log() << "The two most important parameters in the configuration are the  " << Endl;

}

////////////////////////////////////////////////////////////////////////////////

const TMVA::Ranking * TMVA::MethodCrossEvaluation::CreateRanking()
{
   Log() << kFATAL << "Ranking not implemented for CrossEvaluation" << Endl;
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TMVA::MethodCrossEvaluation::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets )
{
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Make ROOT-independent C++ class for classifier response (classifier-specific implementation).

void TMVA::MethodCrossEvaluation::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{

}

////////////////////////////////////////////////////////////////////////////////
/// Specific class header.

void TMVA::MethodCrossEvaluation::MakeClassSpecificHeader(  std::ostream& fout, const TString& className) const
{

}

