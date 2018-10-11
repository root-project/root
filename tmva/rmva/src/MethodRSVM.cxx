// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRSVM-                                                           *
 * Web    : http://oproject.org                                           *
 *                                                                                *
 * Description:                                                                   *
 *       Support Vector Machines                                                  *
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
#include "TMVA/MethodRSVM.h"
#include "TMVA/Tools.h"
#include "TMVA/Config.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

#include "TMVA/Results.h"
#include "TMVA/Timer.h"

using namespace TMVA;

REGISTER_METHOD(RSVM)

ClassImp(MethodRSVM);
//creating an Instance
Bool_t MethodRSVM::IsModuleLoaded = ROOT::R::TRInterface::Instance().Require("e1071");


//_______________________________________________________________________
MethodRSVM::MethodRSVM(const TString &jobName,
                       const TString &methodTitle,
                       DataSetInfo &dsi,
                       const TString &theOption) :
   RMethodBase(jobName, Types::kRSVM, methodTitle, dsi, theOption),
   fMvaCounter(0),
   svm("svm"),
   predict("predict"),
   asfactor("as.factor"),
   fModel(NULL)
{
   // standard constructor for the RSVM
   //Booking options
   fScale = kTRUE;
   fType = "C-classification";
   fKernel = "radial";
   fDegree = 3;

   fGamma = (fDfTrain.GetNcols() == 1) ? 1.0 : (1.0 / fDfTrain.GetNcols());
   fCoef0 = 0;
   fCost = 1;
   fNu = 0.5;
   fCacheSize = 40;
   fTolerance = 0.001;
   fEpsilon = 0.1;
   fShrinking = kTRUE;
   fCross = 0;
   fProbability = kFALSE;
   fFitted = kTRUE;
}

//_______________________________________________________________________
MethodRSVM::MethodRSVM(DataSetInfo &theData, const TString &theWeightFile)
   : RMethodBase(Types::kRSVM, theData, theWeightFile),
     fMvaCounter(0),
     svm("svm"),
     predict("predict"),
     asfactor("as.factor"),
     fModel(NULL)
{
   // standard constructor for the RSVM
   //Booking options
   fScale = kTRUE;
   fType = "C-classification";
   fKernel = "radial";
   fDegree = 3;

   fGamma = (fDfTrain.GetNcols() == 1) ? 1.0 : (1.0 / fDfTrain.GetNcols());
   fCoef0 = 0;
   fCost = 1;
   fNu = 0.5;
   fCacheSize = 40;
   fTolerance = 0.001;
   fEpsilon = 0.1;
   fShrinking = kTRUE;
   fCross = 0;
   fProbability = kTRUE;
   fFitted = kTRUE;
}


//_______________________________________________________________________
MethodRSVM::~MethodRSVM(void)
{
   if (fModel) delete fModel;
}

//_______________________________________________________________________
Bool_t MethodRSVM::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void     MethodRSVM::Init()
{
   if (!IsModuleLoaded) {
      Error("Init", "R's package e1071 can not be loaded.");
      Log() << kFATAL << " R's package e1071 can not be loaded."
            << Endl;
      return;
   }
}

void MethodRSVM::Train()
{
   if (Data()->GetNTrainingEvents() == 0) Log() << kFATAL << "<Train> Data() has zero events" << Endl;
   //SVM require a named vector
   ROOT::R::TRDataFrame ClassWeightsTrain;
   ClassWeightsTrain["background"] = Data()->GetNEvtBkgdTrain();
   ClassWeightsTrain["signal"] = Data()->GetNEvtSigTrain();

   Log() << kINFO
         << " Probability is " << fProbability
         << " Tolerance is " << fTolerance
         << " Type is "  << fType
         << Endl;


   SEXP Model = svm(ROOT::R::Label["x"] = fDfTrain, \
                    ROOT::R::Label["y"] = asfactor(fFactorTrain), \
                    ROOT::R::Label["scale"] = fScale, \
                    ROOT::R::Label["type"] = fType, \
                    ROOT::R::Label["kernel"] = fKernel, \
                    ROOT::R::Label["degree"] = fDegree, \
                    ROOT::R::Label["gamma"] = fGamma, \
                    ROOT::R::Label["coef0"] = fCoef0, \
                    ROOT::R::Label["cost"] = fCost, \
                    ROOT::R::Label["nu"] = fNu, \
                    ROOT::R::Label["class.weights"] = ClassWeightsTrain, \
                    ROOT::R::Label["cachesize"] = fCacheSize, \
                    ROOT::R::Label["tolerance"] = fTolerance, \
                    ROOT::R::Label["epsilon"] = fEpsilon, \
                    ROOT::R::Label["shrinking"] = fShrinking, \
                    ROOT::R::Label["cross"] = fCross, \
                    ROOT::R::Label["probability"] = fProbability, \
                    ROOT::R::Label["fitted"] = fFitted);
   fModel = new ROOT::R::TRObject(Model);
   if (IsModelPersistence())
   {
        TString path = GetWeightFileDir() +  "/" + GetName() + ".RData";
        Log() << Endl;
        Log() << gTools().Color("bold") << "--- Saving State File In:" << gTools().Color("reset") << path << Endl;
        Log() << Endl;
        r["RSVMModel"] << Model;
        r << "save(RSVMModel,file='" + path + "')";
   }
}

//_______________________________________________________________________
void MethodRSVM::DeclareOptions()
{
   DeclareOptionRef(fScale, "Scale", "A logical vector indicating the variables to be scaled. If\
                                       ‘scale’ is of length 1, the value is recycled as many times \
                                       as needed.  Per default, data are scaled internally (both ‘x’\
                                       and ‘y’ variables) to zero mean and unit variance. The center \
                                       and scale values are returned and used for later predictions.");
   DeclareOptionRef(fType, "Type", "‘svm’ can be used as a classification machine, as a \
                                     regression machine, or for novelty detection.  Depending of\
                                     whether ‘y’ is a factor or not, the default setting for\
                                     ‘type’ is ‘C-classification’ or ‘eps-regression’,\
                                     respectively, but may be overwritten by setting an explicit value.\
                                     Valid options are:\
                                      - ‘C-classification’\
                                      - ‘nu-classification’\
                                      - ‘one-classification’ (for novelty detection)\
                                      - ‘eps-regression’\
                                      - ‘nu-regression’");
   DeclareOptionRef(fKernel, "Kernel", "the kernel used in training and predicting. You might\
                                        consider changing some of the following parameters, depending on the kernel type.\
                                        linear: u'*v\
                                        polynomial: (gamma*u'*v + coef0)^degree\
                                        radial basis: exp(-gamma*|u-v|^2)\
                                        sigmoid: tanh(gamma*u'*v + coef0)");
   DeclareOptionRef(fDegree, "Degree", "parameter needed for kernel of type ‘polynomial’ (default: 3)");
   DeclareOptionRef(fGamma, "Gamma", "parameter needed for all kernels except ‘linear’ (default:1/(data dimension))");
   DeclareOptionRef(fCoef0, "Coef0", "parameter needed for kernels of type ‘polynomial’ and ‘sigmoid’ (default: 0)");
   DeclareOptionRef(fCost, "Cost", "cost of constraints violation (default: 1)-it is the ‘C’-constant of the regularization term in the Lagrange formulation.");
   DeclareOptionRef(fNu, "Nu", "parameter needed for ‘nu-classification’, ‘nu-regression’,and ‘one-classification’");
   DeclareOptionRef(fCacheSize, "CacheSize", "cache memory in MB (default 40)");
   DeclareOptionRef(fTolerance, "Tolerance", "tolerance of termination criterion (default: 0.001)");
   DeclareOptionRef(fEpsilon, "Epsilon", "epsilon in the insensitive-loss function (default: 0.1)");
   DeclareOptionRef(fShrinking, "Shrinking", "option whether to use the shrinking-heuristics (default:‘TRUE’)");
   DeclareOptionRef(fCross, "Cross", "if a integer value k>0 is specified, a k-fold cross validation on the training data is performed to assess the quality of the model: the accuracy rate for classification and the Mean Squared Error for regression");
   DeclareOptionRef(fProbability, "Probability", "logical indicating whether the model should allow for probability predictions");
   DeclareOptionRef(fFitted, "Fitted", "logical indicating whether the fitted values should be computed and included in the model or not (default: ‘TRUE’)");

}

//_______________________________________________________________________
void MethodRSVM::ProcessOptions()
{
   r["RMVA.RSVM.Scale"] = fScale;
   r["RMVA.RSVM.Type"] = fType;
   r["RMVA.RSVM.Kernel"] = fKernel;
   r["RMVA.RSVM.Degree"] = fDegree;
   r["RMVA.RSVM.Gamma"] = fGamma;
   r["RMVA.RSVM.Coef0"] = fCoef0;
   r["RMVA.RSVM.Cost"] = fCost;
   r["RMVA.RSVM.Nu"] = fNu;
   r["RMVA.RSVM.CacheSize"] = fCacheSize;
   r["RMVA.RSVM.Tolerance"] = fTolerance;
   r["RMVA.RSVM.Epsilon"] = fEpsilon;
   r["RMVA.RSVM.Shrinking"] = fShrinking;
   r["RMVA.RSVM.Cross"] = fCross;
   r["RMVA.RSVM.Probability"] = fProbability;
   r["RMVA.RSVM.Fitted"] = fFitted;

}

//_______________________________________________________________________
void MethodRSVM::TestClassification()
{
   Log() << kINFO << "Testing Classification RSVM METHOD  " << Endl;

   MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodRSVM::GetMvaValue(Double_t *errLower, Double_t *errUpper)
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
   if (IsModelPersistence()) ReadStateFromFile();

   ROOT::R::TRObject result = predict(*fModel, fDfEvent, ROOT::R::Label["decision.values"] = kTRUE, ROOT::R::Label["probability"] = kTRUE);
   TVectorD values = result.GetAttribute("decision.values");
   mvaValue = values[0]; //returning signal prob
   return mvaValue;
}

////////////////////////////////////////////////////////////////////////////////
/// get all the MVA values for the events of the current Data type
std::vector<Double_t> MethodRSVM::GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t logProgress)
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


   ROOT::R::TRObject result = predict(*fModel, evtData, ROOT::R::Label["decision.values"] = kTRUE, ROOT::R::Label["probability"] = kTRUE);

   r["result"] << result;
   r << "v2 <- attr(result, \"probabilities\") ";
   int probSize = 0;
   r["length(v2)"] >> probSize; 
   //r << "print(v2)";
   if (probSize > 0) {
      std::vector<Double_t> probValues  = result.GetAttribute("probabilities");
      // probabilities are for both cases
      assert(probValues.size() == 2*mvaValues.size());
      for (int i = 0; i < nEvents; ++i)
         // R stores vector column-wise (as in Fortran)
         // and signal probabilities are the second column
         mvaValues[i] = probValues[nEvents+i];
      
   }
   // use decision values
   else {
      Log() << kINFO << " : Probabilities are not available. Use decision values instead !" << Endl;
      //std::cout << "examine the result " << std::endl;
      std::vector<Double_t> probValues = result.GetAttribute("decision.values");
      mvaValues = probValues; 
   // std::cout << "decision values " << values1.size() << std::endl;
   // for ( auto & v : values1) std::cout << v << "  ";
   // std::cout << std::endl;
   }
   

   if (logProgress) {
      Log() << kINFO <<Form("Dataset[%s] : ",DataInfo().GetName())<< "Elapsed time for evaluation of " << nEvents <<  " events: "
            << timer.GetElapsedTime() << "       " << Endl;
   }

   return mvaValues;

}

//_______________________________________________________________________
void TMVA::MethodRSVM::ReadModelFromFile()
{
   ROOT::R::TRInterface::Instance().Require("e1071");
   TString path = GetWeightFileDir() +  "/" + GetName() + ".RData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Loading State File From:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;
   r << "load('" + path + "')";
   SEXP Model;
   r["RSVMModel"] >> Model;
   fModel = new ROOT::R::TRObject(Model);

}

//_______________________________________________________________________
void MethodRSVM::GetHelpMessage() const
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

