// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss,Or Cohen, Eckhard von Toerne

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCompositeBase                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Or Cohen        <orcohenor@gmail.com>    - Weizmann Inst., Israel         *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>        - U of Bonn, Germany          *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                    #include "TMVA/Timer.h"         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// This class is meant to boost a single classifier. Boosting means    //
// training the classifier a few times. Everytime the wieghts of the   //
// events are modified according to how well the classifier performed  //
// on the test sample.                                                 //
//_______________________________________________________________________
#include <algorithm>
#include <iomanip>
#include <vector>
#include <cmath>

#include "Riostream.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TObjString.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TSpline.h"
#include "TDirectory.h"

#include "TMVA/MethodCompositeBase.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodBoost.h"
#include "TMVA/MethodCategory.h"
#include "TMVA/Tools.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Timer.h"
#include "TMVA/Types.h"
#include "TMVA/PDF.h"
#include "TMVA/Results.h"
#include "TMVA/Config.h"

#include "TMVA/SeparationBase.h"
#include "TMVA/GiniIndex.h"
#include "TMVA/RegressionVariance.h"

REGISTER_METHOD(Boost)

ClassImp(TMVA::MethodBoost)

//_______________________________________________________________________
TMVA::MethodBoost::MethodBoost( const TString& jobName,
                                const TString& methodTitle,
                                DataSetInfo& theData,
                                const TString& theOption,
                                TDirectory* theTargetDir ) :
   TMVA::MethodCompositeBase( jobName, Types::kBoost, methodTitle, theData, theOption, theTargetDir )
   , fBoostNum(0)
   , fMethodError(0)
   , fOrigMethodError(0)
   , fBoostWeight(0)
   , fADABoostBeta(0)
   , fRandomSeed(0)
   , fBoostedMethodTitle(methodTitle)
   , fBoostedMethodOptions(theOption)
   , fMonitorHist(0)
   , fMonitorBoostedMethod(kFALSE)
   , fMonitorTree(0)
   , fBoostStage(Types::kBoostProcBegin)
   , fDefaultHistNum(0)
   , fRecalculateMVACut(kFALSE)
   , fROC_training(0.0)
   , fOverlap_integral(0.0)
   , fMVAvalues(0)
{
   fMVAvalues = new std::vector<Float_t>;
}

//_______________________________________________________________________
TMVA::MethodBoost::MethodBoost( DataSetInfo& dsi,
                                const TString& theWeightFile,
                                TDirectory* theTargetDir )
   : TMVA::MethodCompositeBase( Types::kBoost, dsi, theWeightFile, theTargetDir )
   , fBoostNum(0)
   , fMethodError(0)
   , fOrigMethodError(0)
   , fBoostWeight(0)
   , fADABoostBeta(0)
   , fRandomSeed(0)
   , fBoostedMethodTitle("")
   , fBoostedMethodOptions("")
   , fMonitorHist(0)
   , fMonitorBoostedMethod(kFALSE)
   , fMonitorTree(0)
   , fBoostStage(Types::kBoostProcBegin)
   , fDefaultHistNum(0)
   , fRecalculateMVACut(kFALSE)
   , fROC_training(0.0)
   , fOverlap_integral(0.0)
   , fMVAvalues(0)
{
   fMVAvalues = new std::vector<Float_t>;
}

//_______________________________________________________________________
TMVA::MethodBoost::~MethodBoost( void )
{
   // destructor
   fMethodWeight.clear();

   // the histogram themselves are deleted when the file is closed

   if(fMonitorHist) {
      for ( std::vector<TH1*>::iterator it = fMonitorHist->begin(); it != fMonitorHist->end(); ++it) delete *it;
      delete fMonitorHist;
   }
   fTrainSigMVAHist.clear();
   fTrainBgdMVAHist.clear();
   fBTrainSigMVAHist.clear();
   fBTrainBgdMVAHist.clear();
   fTestSigMVAHist.clear();
   fTestBgdMVAHist.clear();

   if (fMVAvalues) {
      delete fMVAvalues;
      fMVAvalues = 0;
   }
}


//_______________________________________________________________________
Bool_t TMVA::MethodBoost::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // Boost can handle classification with 2 classes and regression with one regression-target
   if( type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   //   if( type == Types::kRegression && numberTargets == 1 ) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void TMVA::MethodBoost::DeclareOptions()
{
   DeclareOptionRef( fBoostNum = 1, "Boost_Num",
                     "Number of times the classifier is boosted");

   DeclareOptionRef( fMonitorBoostedMethod = kTRUE, "Boost_MonitorMethod",
                     "Whether to write monitoring histogram for each boosted classifier");
   
   DeclareOptionRef(fBoostType  = "AdaBoost", "Boost_Type", "Boosting type for the classifiers");
   AddPreDefVal(TString("AdaBoost"));
   AddPreDefVal(TString("Bagging"));
   AddPreDefVal(TString("HighEdgeGauss"));
   AddPreDefVal(TString("HighEdgeCoPara"));

   DeclareOptionRef(fMethodWeightType = "ByError", "Boost_MethodWeightType",
                    "How to set the final weight of the boosted classifiers");
   AddPreDefVal(TString("ByError"));
   AddPreDefVal(TString("Average"));
   AddPreDefVal(TString("ByROC"));
   AddPreDefVal(TString("ByOverlap"));
   AddPreDefVal(TString("LastMethod"));

   DeclareOptionRef(fRecalculateMVACut = kTRUE, "Boost_RecalculateMVACut",
                    "Whether to recalculate the classifier MVA Signallike cut at every boost iteration");

   DeclareOptionRef(fADABoostBeta = 1.0, "Boost_AdaBoostBeta",
                    "The ADA boost parameter that sets the effect of every boost step on the events' weights");
   
   DeclareOptionRef(fTransformString = "step", "Boost_Transform",
                    "Type of transform applied to every boosted method linear, log, step");
   AddPreDefVal(TString("step"));
   AddPreDefVal(TString("linear"));
   AddPreDefVal(TString("log"));

   DeclareOptionRef(fRandomSeed = 0, "Boost_RandomSeed",
                    "Seed for random number generator used for bagging");

   TMVA::MethodCompositeBase::fMethods.reserve(fBoostNum);;
}

//_______________________________________________________________________
Bool_t TMVA::MethodBoost::BookMethod( Types::EMVA theMethod, TString methodTitle, TString theOption )
{
   // just registering the string from which the boosted classifier will be created
   fBoostedMethodName = Types::Instance().GetMethodName( theMethod );
   fBoostedMethodTitle = methodTitle;
   fBoostedMethodOptions = theOption;
   return kTRUE;
}

//_______________________________________________________________________
void TMVA::MethodBoost::Init()
{}

//_______________________________________________________________________
void TMVA::MethodBoost::InitHistos()
{
   // initialisation routine
   if(fMonitorHist) {
      for ( std::vector<TH1*>::iterator it = fMonitorHist->begin(); it != fMonitorHist->end(); ++it) delete *it;
      delete fMonitorHist;
   }
   fMonitorHist = new std::vector<TH1*>();
   fMonitorHist->push_back(new TH1F("MethodWeight","Normalized Classifier Weight",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("BoostWeight","Boost Weight",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("ErrFraction","Error Fraction (by boosted event weights)",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("OrigErrFraction","Error Fraction (by original event weights)",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("ROCIntegral_test","ROC integral of single classifier (testing sample)",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("ROCIntegralBoosted_test","ROC integral of boosted method (testing sample)",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("ROCIntegral_train","ROC integral of single classifier (training sample)",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("ROCIntegralBoosted_train","ROC integral of boosted method (training sample)",fBoostNum,0,fBoostNum));
   fMonitorHist->push_back(new TH1F("OverlapIntegal_train","Overlap integral (training sample)",fBoostNum,0,fBoostNum));
   for ( std::vector<TH1*>::iterator it = fMonitorHist->begin(); it != fMonitorHist->end(); ++it ) (*it)->SetDirectory(0);
   fDefaultHistNum = fMonitorHist->size();
   (*fMonitorHist)[0]->GetXaxis()->SetTitle("Index of boosted classifier");
   (*fMonitorHist)[0]->GetYaxis()->SetTitle("Classifier Weight");
   (*fMonitorHist)[1]->GetXaxis()->SetTitle("Index of boosted classifier");
   (*fMonitorHist)[1]->GetYaxis()->SetTitle("Boost Weight");
   (*fMonitorHist)[2]->GetXaxis()->SetTitle("Index of boosted classifier");
   (*fMonitorHist)[2]->GetYaxis()->SetTitle("Error Fraction");
   (*fMonitorHist)[3]->GetXaxis()->SetTitle("Index of boosted classifier");
   (*fMonitorHist)[3]->GetYaxis()->SetTitle("Error Fraction");
   (*fMonitorHist)[4]->GetXaxis()->SetTitle("Index of boosted classifier");
   (*fMonitorHist)[4]->GetYaxis()->SetTitle("ROC integral of single classifier");
   (*fMonitorHist)[5]->GetXaxis()->SetTitle("Number of boosts");
   (*fMonitorHist)[5]->GetYaxis()->SetTitle("ROC integral boosted");
   (*fMonitorHist)[6]->GetXaxis()->SetTitle("Index of boosted classifier");
   (*fMonitorHist)[6]->GetYaxis()->SetTitle("ROC integral of single classifier");
   (*fMonitorHist)[7]->GetXaxis()->SetTitle("Number of boosts");
   (*fMonitorHist)[7]->GetYaxis()->SetTitle("ROC integral boosted");
   (*fMonitorHist)[8]->GetXaxis()->SetTitle("Index of boosted classifier");
   (*fMonitorHist)[8]->GetYaxis()->SetTitle("Overlap integral");

   fMonitorTree= new TTree("MonitorBoost","Boost variables");
   fMonitorTree->Branch("iMethod",&fMethodIndex,"iMethod/I");
   fMonitorTree->Branch("boostWeight",&fBoostWeight,"boostWeight/D");
   fMonitorTree->Branch("errorFraction",&fMethodError,"errorFraction/D");
   fMonitorBoostedMethod = kTRUE;
}


//_______________________________________________________________________
void TMVA::MethodBoost::CheckSetup()
{
   Log() << kDEBUG << "CheckSetup: fBoostType="<<fBoostType<<" fMethodWeightType=" << fMethodWeightType << Endl;
   Log() << kDEBUG << "CheckSetup: fADABoostBeta="<<fADABoostBeta<<Endl;
   Log() << kDEBUG << "CheckSetup: fBoostWeight="<<fBoostWeight<<Endl;
   Log() << kDEBUG << "CheckSetup: fMethodError="<<fMethodError<<Endl;
   Log() << kDEBUG << "CheckSetup: fOrigMethodError="<<fOrigMethodError<<Endl;
   Log() << kDEBUG << "CheckSetup: fBoostNum="<<fBoostNum<< " fMonitorHist="<< fMonitorHist<< Endl;
   Log() << kDEBUG << "CheckSetup: fRandomSeed=" << fRandomSeed<< Endl;
   Log() << kDEBUG << "CheckSetup: fDefaultHistNum=" << fDefaultHistNum << " fRecalculateMVACut=" << (fRecalculateMVACut? "true" : "false") << Endl;
   Log() << kDEBUG << "CheckSetup: fTrainSigMVAHist.size()="<<fTrainSigMVAHist.size()<<Endl;
   Log() << kDEBUG << "CheckSetup: fTestSigMVAHist.size()="<<fTestSigMVAHist.size()<<Endl;
   Log() << kDEBUG << "CheckSetup: fMonitorBoostedMethod=" << (fMonitorBoostedMethod? "true" : "false") << Endl;
   Log() << kDEBUG << "CheckSetup: MName=" << fBoostedMethodName << " Title="<< fBoostedMethodTitle<< Endl;
   Log() << kDEBUG << "CheckSetup: MOptions="<< fBoostedMethodOptions << Endl;
   Log() << kDEBUG << "CheckSetup: fBoostStage=" << fBoostStage<<Endl;
   Log() << kDEBUG << "CheckSetup: fMonitorTree=" << fMonitorTree <<Endl;
   Log() << kDEBUG << "CheckSetup: fMethodIndex=" <<fMethodIndex << Endl;
   if (fMethods.size()>0) Log() << kDEBUG << "CheckSetup: fMethods[0]" <<fMethods[0]<<Endl;
   Log() << kDEBUG << "CheckSetup: fMethodWeight.size()" << fMethodWeight.size() << Endl;
   if (fMethodWeight.size()>0) Log() << kDEBUG << "CheckSetup: fMethodWeight[0]="<<fMethodWeight[0]<<Endl;
   Log() << kDEBUG << "CheckSetup: trying to repair things" << Endl;

   //TMVA::MethodBase::CheckSetup();
   if (fMonitorHist == 0){
      InitHistos();
      CheckSetup();
   }
}
//_______________________________________________________________________
void TMVA::MethodBoost::Train()
{
   Double_t    AllMethodsWeight=0;
   TDirectory* methodDir( 0 );
   TString     dirName,dirTitle;
   Int_t       StopCounter=0;

   if (Data()->GetNTrainingEvents()==0) Log() << kFATAL << "<Train> Data() has zero events" << Endl;
   Data()->SetCurrentType(Types::kTraining);

   if (fMethods.size() > 0) fMethods.clear();
   fMVAvalues->resize(Data()->GetNTrainingEvents(), 0.0);

   Log() << kINFO << "Training "<< fBoostNum << " " << fBoostedMethodName << " Classifiers ... patience please" << Endl;
   Timer timer( fBoostNum, GetName() );

   ResetBoostWeights();

   // clean boosted method options
   CleanBoostOptions();
   //
   // training and boosting the classifiers
   for (fMethodIndex=0;fMethodIndex<fBoostNum;fMethodIndex++) {
      // the first classifier shows the option string output, the rest not
      if (fMethodIndex>0) TMVA::MsgLogger::InhibitOutput();
      IMethod* method = ClassifierFactory::Instance().Create(std::string(fBoostedMethodName),
                                                             GetJobName(),
                                                             Form("%s_B%04i", fBoostedMethodName.Data(),fMethodIndex),
                                                             DataInfo(),
                                                             fBoostedMethodOptions);
      TMVA::MsgLogger::EnableOutput();

      // supressing the rest of the classifier output the right way
      MethodBase *meth = (dynamic_cast<MethodBase*>(method));

      if(meth==0) continue;

      // set fDataSetManager if MethodCategory (to enable Category to create datasetinfo objects) // DSMTEST
      if( meth->GetMethodType() == Types::kCategory ){ // DSMTEST
         MethodCategory *methCat = (dynamic_cast<MethodCategory*>(meth)); // DSMTEST
         if( !methCat ) // DSMTEST
            Log() << kFATAL << "Method with type kCategory cannot be casted to MethodCategory. /MethodBoost" << Endl; // DSMTEST
         methCat->fDataSetManager = fDataSetManager; // DSMTEST
      } // DSMTEST


      meth->SetMsgType(kWARNING);
      meth->SetupMethod();
      meth->ParseOptions();
      // put SetAnalysisType here for the needs of MLP
      meth->SetAnalysisType( GetAnalysisType() );
      meth->ProcessSetup();
      meth->CheckSetup();

      // creating the directory of the classifier
      if (fMonitorBoostedMethod)
         {
            methodDir=MethodBaseDir()->GetDirectory(dirName=Form("%s_B%04i",fBoostedMethodName.Data(),fMethodIndex));
            if (methodDir==0)
               methodDir=BaseDir()->mkdir(dirName,dirTitle=Form("Directory Boosted %s #%04i", fBoostedMethodName.Data(),fMethodIndex));
            MethodBase* m = dynamic_cast<MethodBase*>(method);
            if(m) {
               m->SetMethodDir(methodDir);
               m->BaseDir()->cd();
            }
         }

      // training
      TMVA::MethodCompositeBase::fMethods.push_back(method);
      timer.DrawProgressBar( fMethodIndex );
      if (fMethodIndex==0) method->MonitorBoost(SetStage(Types::kBoostProcBegin));
      method->MonitorBoost(SetStage(Types::kBeforeTraining));
      TMVA::MsgLogger::InhibitOutput(); //supressing Logger outside the method
      SingleTrain();
      TMVA::MsgLogger::EnableOutput();
      method->WriteMonitoringHistosToFile();
      
      // calculate MVA values of method on training sample
      CalcMVAValues();
      
      if (fMethodIndex==0 && fMonitorBoostedMethod) CreateMVAHistorgrams();
      
      // get ROC integral and overlap integral for single method on
      // training sample
      fROC_training = GetBoostROCIntegral(kTRUE, Types::kTraining, kTRUE);
	 
      // calculate method weight
      CalcMethodWeight();
      AllMethodsWeight += fMethodWeight.back();

      (*fMonitorHist)[4]->SetBinContent(fMethodIndex+1, GetBoostROCIntegral(kTRUE, Types::kTesting));
      (*fMonitorHist)[5]->SetBinContent(fMethodIndex+1, GetBoostROCIntegral(kFALSE, Types::kTesting));
      (*fMonitorHist)[6]->SetBinContent(fMethodIndex+1, fROC_training);
      (*fMonitorHist)[7]->SetBinContent(fMethodIndex+1, GetBoostROCIntegral(kFALSE, Types::kTraining));
      (*fMonitorHist)[8]->SetBinContent(fMethodIndex+1, fOverlap_integral);

      // boosting (reweight training sample)
      method->MonitorBoost(SetStage(Types::kBeforeBoosting));
      SingleBoost();
      method->MonitorBoost(SetStage(Types::kAfterBoosting));
      (*fMonitorHist)[1]->SetBinContent(fMethodIndex+1,fBoostWeight);
      (*fMonitorHist)[2]->SetBinContent(fMethodIndex+1,fMethodError);
      (*fMonitorHist)[3]->SetBinContent(fMethodIndex+1,fOrigMethodError);

      fMonitorTree->Fill();

      // stop boosting if needed when error has reached 0.5
      // thought of counting a few steps, but it doesn't seem to be necessary
      Log() << kDEBUG << "AdaBoost (methodErr) err = " << fMethodError << Endl;
      if (fMethodError > 0.49999) StopCounter++; 
      if (StopCounter > 0 && fBoostType != "Bagging")
         {
            timer.DrawProgressBar( fBoostNum );
            fBoostNum = fMethodIndex+1; 
            Log() << kINFO << "Error rate has reached 0.5, boosting process stopped at #" << fBoostNum << " classifier" << Endl;
            if (fBoostNum < 5)
               Log() << kINFO << "The classifier might be too strong to boost with Beta = " << fADABoostBeta << ", try reducing it." <<Endl;
            for (Int_t i=0;i<fDefaultHistNum;i++)
               (*fMonitorHist)[i]->SetBins(fBoostNum,0,fBoostNum);
            break;
         }
   }
   if (fMethodWeightType == "LastMethod") { fMethodWeight.back() = AllMethodsWeight = 1.0; }

   ResetBoostWeights();
   Timer* timer1=new Timer();
   // normalizing the weights of the classifiers
   for (fMethodIndex=0;fMethodIndex<fBoostNum;fMethodIndex++) {
      // pefroming post-boosting actions
      if (fMethods[fMethodIndex]->MonitorBoost(SetStage(Types::kBoostValidation))) {
         if (fMethodIndex==0) timer1 = new Timer( fBoostNum, GetName() );

         timer1->DrawProgressBar( fMethodIndex );

         if (fMethodIndex==fBoostNum) {
            Log() << kINFO << "Elapsed time: " << timer1->GetElapsedTime() 
                  << "                              " << Endl;
         }
      }

      if (AllMethodsWeight != 0.0)
         fMethodWeight[fMethodIndex] = fMethodWeight[fMethodIndex] / AllMethodsWeight;
      (*fMonitorHist)[0]->SetBinContent(fMethodIndex+1,fMethodWeight[fMethodIndex]);
   }

   // Ensure that in case of only 1 boost the method weight equals
   // 1.0.  This avoids unexpected behaviour in case of very bad
   // classifiers which have fBoostWeight=1 or fMethodError=0.5,
   // because their weight would be set to zero.  This behaviour is
   // not ok if one boosts just one time.
   if (fMethods.size()==1)  fMethodWeight[0] = 1.0;

   fMethods.back()->MonitorBoost(SetStage(Types::kBoostProcEnd));

   delete timer1;
}

//_______________________________________________________________________
void TMVA::MethodBoost::CleanBoostOptions()
{
   fBoostedMethodOptions=GetOptions(); 
}

//_______________________________________________________________________
void TMVA::MethodBoost::CreateMVAHistorgrams()
{
   if (fBoostNum <=0) Log() << kFATAL << "CreateHistorgrams called before fBoostNum is initialized" << Endl;
   // calculating histograms boundries and creating histograms..
   // nrms = number of rms around the average to use for outline (of the 0 classifier)
   Double_t meanS, meanB, rmsS, rmsB, xmin, xmax, nrms = 10;
   Int_t signalClass = 0;
   if (DataInfo().GetClassInfo("Signal") != 0) {
      signalClass = DataInfo().GetClassInfo("Signal")->GetNumber();
   }
   gTools().ComputeStat( Data()->GetEventCollection(), fMVAvalues,
                         meanS, meanB, rmsS, rmsB, xmin, xmax, signalClass );

   fNbins = gConfig().fVariablePlotting.fNbinsXOfROCCurve;
   xmin = TMath::Max( TMath::Min(meanS - nrms*rmsS, meanB - nrms*rmsB ), xmin );
   xmax = TMath::Min( TMath::Max(meanS + nrms*rmsS, meanB + nrms*rmsB ), xmax ) + 0.00001;

   // creating all the historgrams
   for (Int_t imtd=0; imtd<fBoostNum; imtd++) {
      fTrainSigMVAHist .push_back( new TH1F( Form("MVA_Train_S_%04i",imtd), "MVA_Train_S", fNbins, xmin, xmax ) );
      fTrainBgdMVAHist .push_back( new TH1F( Form("MVA_Train_B%04i",imtd), "MVA_Train_B", fNbins, xmin, xmax ) );
      fBTrainSigMVAHist.push_back( new TH1F( Form("MVA_BTrain_S%04i",imtd), "MVA_BoostedTrain_S", fNbins, xmin, xmax ) );
      fBTrainBgdMVAHist.push_back( new TH1F( Form("MVA_BTrain_B%04i",imtd), "MVA_BoostedTrain_B", fNbins, xmin, xmax ) );
      fTestSigMVAHist  .push_back( new TH1F( Form("MVA_Test_S%04i",imtd), "MVA_Test_S", fNbins, xmin, xmax ) );
      fTestBgdMVAHist  .push_back( new TH1F( Form("MVA_Test_B%04i",imtd), "MVA_Test_B", fNbins, xmin, xmax ) );
   }
}

//_______________________________________________________________________
void TMVA::MethodBoost::ResetBoostWeights()
{
   // resetting back the boosted weights of the events to 1
   for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
      Event *ev = Data()->GetEvent(ievt);
      ev->SetBoostWeight( 1.0 );
   }
}

//_______________________________________________________________________
void TMVA::MethodBoost::WriteMonitoringHistosToFile( void ) const
{
   TDirectory* dir=0;
   if (fMonitorBoostedMethod) {
      for (Int_t imtd=0;imtd<fBoostNum;imtd++) {

         //writing the histograms in the specific classifier's directory
         MethodBase* m = dynamic_cast<MethodBase*>(fMethods[imtd]);
         if(!m) continue;
         dir = m->BaseDir();
         dir->cd();
         fTrainSigMVAHist[imtd]->SetDirectory(dir);
         fTrainSigMVAHist[imtd]->Write();
         fTrainBgdMVAHist[imtd]->SetDirectory(dir);
         fTrainBgdMVAHist[imtd]->Write();
         fBTrainSigMVAHist[imtd]->SetDirectory(dir);
         fBTrainSigMVAHist[imtd]->Write();
         fBTrainBgdMVAHist[imtd]->SetDirectory(dir);
         fBTrainBgdMVAHist[imtd]->Write();
      }
   }

   // going back to the original folder
   BaseDir()->cd();
   for (UInt_t i=0;i<fMonitorHist->size();i++) {
      ((*fMonitorHist)[i])->Write(Form("Booster_%s",((*fMonitorHist)[i])->GetName()));
   }

   fMonitorTree->Write();
}

//_______________________________________________________________________
void TMVA::MethodBoost::TestClassification()
{
   MethodBase::TestClassification();
   if (fMonitorBoostedMethod) {
      UInt_t nloop = fTestSigMVAHist.size();
      if (fMethods.size()<nloop) nloop = fMethods.size();
      //running over all the events and populating the test MVA histograms
      Data()->SetCurrentType(Types::kTesting);
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         Event* ev = Data()->GetEvent(ievt);
         Float_t w = ev->GetWeight();
         if (DataInfo().IsSignal(ev)) {
            for (UInt_t imtd=0; imtd<nloop; imtd++) {
               fTestSigMVAHist[imtd]->Fill(fMethods[imtd]->GetMvaValue(),w);
            }
         }
         else {
            for (UInt_t imtd=0; imtd<nloop; imtd++) {
               fTestBgdMVAHist[imtd]->Fill(fMethods[imtd]->GetMvaValue(),w);
            }
         }
      }
      Data()->SetCurrentType(Types::kTraining);
   }
}

//_______________________________________________________________________
void TMVA::MethodBoost::WriteEvaluationHistosToFile(Types::ETreeType treetype)
{
   MethodBase::WriteEvaluationHistosToFile(treetype);
   if(treetype==Types::kTraining) return;
   UInt_t nloop = fTestSigMVAHist.size();
   if (fMethods.size()<nloop) nloop = fMethods.size();
   if (fMonitorBoostedMethod) {
      TDirectory* dir=0;
      for (UInt_t imtd=0;imtd<nloop;imtd++) {
         //writing the histograms in the specific classifier's directory
         MethodBase* mva = dynamic_cast<MethodBase*>(fMethods[imtd]);
         if(!mva) continue;
         dir = mva->BaseDir();
         if(dir==0) continue;
         dir->cd();
         fTestSigMVAHist[imtd]->SetDirectory(dir);
         fTestSigMVAHist[imtd]->Write();
         fTestBgdMVAHist[imtd]->SetDirectory(dir);
         fTestBgdMVAHist[imtd]->Write();
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodBoost::ProcessOptions()
{
   // process user options
}

//_______________________________________________________________________
void TMVA::MethodBoost::SingleTrain()
{
   // initialization
   Data()->SetCurrentType(Types::kTraining);
   MethodBase* meth = dynamic_cast<MethodBase*>(GetLastMethod());
   if(meth)
      meth->TrainMethod();
}

//_______________________________________________________________________
void TMVA::MethodBoost::FindMVACut()
{
   // find the CUT on the individual MVA that defines an event as 
   // correct or misclassified (to be used in the boosting process)

   MethodBase* lastMethod=dynamic_cast<MethodBase*>(fMethods.back());
   if (!lastMethod || lastMethod->GetMethodType() == Types::kDT ){ return;}

   if (!fRecalculateMVACut && fMethodIndex>0) {
      MethodBase* m = dynamic_cast<MethodBase*>(fMethods[0]);
      if(m)
         lastMethod->SetSignalReferenceCut(m->GetSignalReferenceCut());
   } else {

      // creating a fine histograms containing the error rate
      const Int_t nValBins=1000;
      Double_t* err=new Double_t[nValBins];
      const Double_t valmin=-1.5;
      const Double_t valmax=1.5;
      for (Int_t i=0;i<nValBins;i++) err[i]=0.;
      Double_t sum = 0.;
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         Double_t weight = GetEvent(ievt)->GetWeight();
         sum +=weight;
         Double_t val=lastMethod->GetMvaValue();
         Int_t ibin = (Int_t) (((val-valmin)/(valmax-valmin))*nValBins);
         
         if (ibin>=nValBins) ibin = nValBins-1;
         if (ibin<0) ibin = 0;
         if (DataInfo().IsSignal(Data()->GetEvent(ievt))){
            for (Int_t i=ibin;i<nValBins;i++) err[i]+=weight;
         }
         else {
            for (Int_t i=0;i<ibin;i++) err[i]+=weight;
         }
      }
      Double_t minerr=1.e6;
      Int_t minbin=-1;
      for (Int_t i=0;i<nValBins;i++){
         if (err[i]<=minerr){
            minerr=err[i];
            minbin=i;
         }
      }
      delete[] err;
      
      
      Double_t sigCutVal = valmin + ((valmax-valmin)*minbin)/Float_t(nValBins+1);
      lastMethod->SetSignalReferenceCut(sigCutVal);
      
      Log() << kDEBUG << "(old step) Setting method cut to " <<lastMethod->GetSignalReferenceCut()<< Endl;
      
   }
   
}

//_______________________________________________________________________
void TMVA::MethodBoost::SingleBoost()
{
   MethodBase* method =  dynamic_cast<MethodBase*>(fMethods.back());
   if(!method) return;
   Event * ev; Float_t w,v,wo; Bool_t sig=kTRUE;
   Double_t sumAll=0, sumWrong=0, sumAllOrig=0, sumWrongOrig=0, sumAll1=0;
   Bool_t* WrongDetection=new Bool_t[Data()->GetNEvents()];
   for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) WrongDetection[ievt]=kTRUE;

   // finding the wrong events and calculating their total weights
   for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
      ev = Data()->GetEvent(ievt);
      sig=DataInfo().IsSignal(ev);
      v = fMVAvalues->at(ievt);
      w = ev->GetWeight();
      wo = ev->GetOriginalWeight();
      if (sig && fMonitorBoostedMethod) {
         fBTrainSigMVAHist[fMethodIndex]->Fill(v,w);
         fTrainSigMVAHist[fMethodIndex]->Fill(v,ev->GetOriginalWeight());
      }
      else if (fMonitorBoostedMethod) {
         fBTrainBgdMVAHist[fMethodIndex]->Fill(v,w);
         fTrainBgdMVAHist[fMethodIndex]->Fill(v,ev->GetOriginalWeight());
      }
      sumAll += w;
      sumAllOrig += wo;
      if ( sig != (fMVAvalues->at(ievt) > method->GetSignalReferenceCut()) ) {
	 WrongDetection[ievt]=kTRUE; 
	 sumWrong+=w; 
	 sumWrongOrig+=wo;
      }
      else WrongDetection[ievt]=kFALSE;
   }
   fMethodError=sumWrong/sumAll;
   fOrigMethodError = sumWrongOrig/sumAllOrig;
   Log() << kDEBUG << "AdaBoost err (MethodErr1)= " << fMethodError<<" = wrong/all: " << sumWrong << "/" << sumAll<< " cut="<<method->GetSignalReferenceCut()<< Endl;

   // calculating the fMethodError and the fBoostWeight out of it uses the formula 
   // w = ((1-err)/err)^beta
   if (fMethodError>0 && fADABoostBeta == 1.0) {
      fBoostWeight = (1.0-fMethodError)/fMethodError;
   }
   else if (fMethodError>0 && fADABoostBeta != 1.0) {
      fBoostWeight =  TMath::Power((1.0 - fMethodError)/fMethodError, fADABoostBeta);
   }
   else fBoostWeight = 1000;

   Double_t alphaWeight = ( fBoostWeight > 0.0 ? TMath::Log(fBoostWeight) : 0.0);
   if (alphaWeight>5.) alphaWeight = 5.;
   if (alphaWeight<0.){
      //Log()<<kWARNING<<"alphaWeight is too small in AdaBoost alpha=" << alphaWeight<< Endl;
      alphaWeight = -alphaWeight;
   }
   if (fBoostType == "AdaBoost") {
      // ADA boosting, rescaling the weight of the wrong events according to the error level
      // over the entire test sample rescaling all the weights to have the same sum, but without
      // touching the original weights (changing only the boosted weight of all the events)
      // first reweight
      Double_t newSum=0., oldSum=0.;
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         ev =  Data()->GetEvent(ievt);
         oldSum += ev->GetWeight();
         //         ev->ScaleBoostWeight(TMath::Exp(-alphaWeight*((WrongDetection[ievt])? -1.0 : 1.0)));
         //ev->ScaleBoostWeight(TMath::Exp(-alphaWeight*((WrongDetection[ievt])? -1.0 : 0)));
         if (WrongDetection[ievt]) ev->ScaleBoostWeight(fBoostWeight);
         newSum += ev->GetWeight();
      }

      Double_t normWeight = oldSum/newSum;
      // bla      std::cout << "Normalize weight by (Boost)" << normWeight <<  " = " << oldSum<<"/"<<newSum<< " eventBoostFactor="<<fBoostWeight<<std::endl;
      // next normalize the weights
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         Data()->GetEvent(ievt)->ScaleBoostWeight(normWeight);
      }

   }
   else if (fBoostType == "Bagging") {
      // Bagging or Bootstrap boosting, gives new random weight for every event
      TRandom3*trandom   = new TRandom3(fRandomSeed+fMethods.size());
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         ev = Data()->GetEvent(ievt);
         ev->SetBoostWeight(trandom->Rndm());
         sumAll1+=ev->GetWeight();
      }
      // rescaling all the weights to have the same sum, but without touching the original
      // weights (changing only the boosted weight of all the events)
      Double_t Factor=sumAll/sumAll1;
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         ev = Data()->GetEvent(ievt);
         ev->ScaleBoostWeight(Factor);
      }
   }
   else if (fBoostType == "HighEdgeGauss" || 
	    fBoostType == "HighEdgeCoPara") {
      // Give events high boost weight, which are close of far away
      // from the MVA cut value
      Double_t MVACutValue = method->GetSignalReferenceCut();
      sumAll1 = 0;
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         ev = Data()->GetEvent(ievt);
	 if (fBoostType == "HighEdgeGauss")
	    ev->SetBoostWeight( TMath::Exp( -std::pow(fMVAvalues->at(ievt)-MVACutValue,2)/(0.1*fADABoostBeta) ) );
	 else if (fBoostType == "HighEdgeCoPara")
	    ev->SetBoostWeight( DataInfo().IsSignal(ev) ? TMath::Power(1.0-fMVAvalues->at(ievt),fADABoostBeta) : TMath::Power(fMVAvalues->at(ievt),fADABoostBeta) );
	 else
	    Log() << kFATAL << "Unknown event weight type!" << Endl;

         sumAll1 += ev->GetWeight();
      }
      // rescaling all the weights to have the same sum, but without
      // touching the original weights (changing only the boosted
      // weight of all the events)
      Double_t Factor=sumAll/sumAll1;
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++)
         Data()->GetEvent(ievt)->ScaleBoostWeight(Factor);
   }
   delete[] WrongDetection;
}

//_______________________________________________________________________
void TMVA::MethodBoost::CalcMethodWeight()
{
   // Calculate weight of single method.
   // This is no longer done in SingleBoost();

   MethodBase* method =  dynamic_cast<MethodBase*>(fMethods.back());
   if (!method) {
      Log() << kFATAL << "Dynamic cast to MethodBase* failed" <<Endl;
      return;
   }

   Event * ev; Float_t w;
   Double_t sumAll=0, sumWrong=0;

   // finding the MVA cut value for IsSignalLike, stored in the method
   FindMVACut();

   // finding the wrong events and calculating their total weights
   for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
      ev      = Data()->GetEvent(ievt);
      w       = ev->GetWeight();
      sumAll += w;
      if ( DataInfo().IsSignal(ev) != 
	   (fMVAvalues->at(ievt) > method->GetSignalReferenceCut()) )
	 sumWrong += w;
   }
   fMethodError=sumWrong/sumAll;

   // calculating the fMethodError and the fBoostWeight out of it uses
   // the formula
   // w = ((1-err)/err)^beta
   if (fMethodError>0 && fADABoostBeta == 1.0) {
      fBoostWeight = (1.0-fMethodError)/fMethodError;
   }
   else if (fMethodError>0 && fADABoostBeta != 1.0) {
      fBoostWeight =  TMath::Power((1.0 - fMethodError)/fMethodError, fADABoostBeta);
   }
   else fBoostWeight = 1000;

   // sanity check to avoid log() with negative argument
   if (fBoostWeight <= 0.0)  fBoostWeight = 1.0;
   
   // calculate method weight
   if      (fMethodWeightType == "ByError") fMethodWeight.push_back(TMath::Log(fBoostWeight));
   else if (fMethodWeightType == "Average") fMethodWeight.push_back(1.0);
   else if (fMethodWeightType == "ByROC")   fMethodWeight.push_back(fROC_training);
   else if (fMethodWeightType == "ByOverlap") fMethodWeight.push_back((fOverlap_integral > 0.0 ? 1.0/fOverlap_integral : 1000.0));
   else                                     fMethodWeight.push_back(0);
}

//_______________________________________________________________________
void TMVA::MethodBoost::GetHelpMessage() const
{
   // Get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "This method combines several classifier of one species in a "<<Endl;
   Log() << "single multivariate quantity via the boost algorithm." << Endl;
   Log() << "the output is a weighted sum over all individual classifiers" <<Endl;
   Log() << "By default, the AdaBoost method is employed, which gives " << Endl;
   Log() << "events that were misclassified in the previous tree a larger " << Endl;
   Log() << "weight in the training of the following classifier."<<Endl;
   Log() << "Optionally, Bagged boosting can also be applied." << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The most important parameter in the configuration is the "<<Endl;
   Log() << "number of boosts applied (Boost_Num) and the choice of boosting"<<Endl;
   Log() << "(Boost_Type), which can be set to either AdaBoost or Bagging." << Endl;
   Log() << "AdaBoosting: The most important parameters in this configuration" <<Endl;
   Log() << "is the beta parameter (Boost_AdaBoostBeta)  " << Endl;
   Log() << "When boosting a linear classifier, it is sometimes advantageous"<<Endl; 
   Log() << "to transform the MVA output non-linearly. The following options" <<Endl;
   Log() << "are available: step, log, and minmax, the default is no transform."<<Endl;
   Log() <<Endl;
   Log() << "Some classifiers are hard to boost and do not improve much in"<<Endl; 
   Log() << "their performance by boosting them, some even slightly deteriorate"<< Endl;
   Log() << "due to the boosting." <<Endl;
   Log() << "The booking of the boost method is special since it requires"<<Endl;
   Log() << "the booing of the method to be boosted and the boost itself."<<Endl;
   Log() << "This is solved by booking the method to be boosted and to add"<<Endl;
   Log() << "all Boost parameters, which all begin with \"Boost_\" to the"<<Endl;
   Log() << "options string. The factory separates the options and initiates"<<Endl;
   Log() << "the boost process. The TMVA macro directory contains the example"<<Endl;
   Log() << "macro \"Boost.C\"" <<Endl;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodBoost::CreateRanking()
{ 
   return 0;
}

//_______________________________________________________________________
Double_t TMVA::MethodBoost::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // return boosted MVA response
   Double_t mvaValue = 0;
   Double_t epsilon = TMath::Exp(-1.);
   //Double_t fact    = TMath::Exp(-1.)+TMath::Exp(1.);
   for (UInt_t i=0;i< fMethods.size(); i++){
      MethodBase* m = dynamic_cast<MethodBase*>(fMethods[i]);
      if(m==0) continue;
      Double_t val = fTmpEvent ? m->GetMvaValue(fTmpEvent) : m->GetMvaValue();
      Double_t sigcut = m->GetSignalReferenceCut();
      // default is no transform
      if (fTransformString == "linear"){

      }
      else if (fTransformString == "log"){
         if (val < sigcut) val = sigcut;

         val = TMath::Log((val-sigcut)+epsilon);
      }
      else if (fTransformString == "step" ){
         if (val < sigcut) val = -1.;
         else val = 1.;
      }
      else {
         Log() << kFATAL << "error unknown transformation " << fTransformString<<Endl;
      }
      mvaValue+=val*fMethodWeight[i];
   }
   // cannot determine error
   NoErrorCalc(err, errUpper);

   return mvaValue;
}

//_______________________________________________________________________
Double_t TMVA::MethodBoost::GetBoostROCIntegral(Bool_t singleMethod, Types::ETreeType eTT, Bool_t CalcOverlapIntergral)
{
   // Calculate the ROC integral of a single classifier or even the
   // whole boosted classifier.  The tree type (training or testing
   // sample) is specified by 'eTT'.
   //
   // If tree type kTraining is set, the original training sample is
   // used to compute the ROC integral (original weights).
   //
   // - singleMethod - if kTRUE, return ROC integral of single (last
   //                  trained) classifier; if kFALSE, return ROC
   //                  integral of full classifier
   //
   // - eTT - tree type (Types::kTraining / Types::kTesting)
   //
   // - CalcOverlapIntergral - if kTRUE, the overlap integral of the
   //                          signal/background MVA distributions
   //                          is calculated and stored in
   //                          'fOverlap_integral'

   // set data sample training / testing
   Data()->SetCurrentType(eTT);

   MethodBase* method = singleMethod ? dynamic_cast<MethodBase*>(fMethods.back()) : 0; // ToDo CoVerity flags this line as there is no prtection against a zero-pointer delivered by dynamic_cast
   // to make CoVerity happy (although, OF COURSE, the last method in the commitee
   // has to be also of type MethodBase as ANY method is... hence the dynamic_cast
   // will never by "zero" ...
   if (singleMethod && !method) {
      Log() << kFATAL << " What do you do? Your method:"
            << fMethods.back()->GetName() 
            << " seems not to be a propper TMVA method" 
            << Endl;
      std::exit(1);
   }
   Double_t err = 0.0;

   // temporary renormalize the method weights in case of evaluation
   // of full classifier.
   // save the old normalization of the methods
   std::vector<Double_t> OldMethodWeight(fMethodWeight);
   if (!singleMethod) {
      // calculate sum of weights of all methods
      Double_t AllMethodsWeight = 0;
      for (Int_t i=0; i<=fMethodIndex; i++)
         AllMethodsWeight += fMethodWeight.at(i);
      // normalize the weights of the classifiers
      if (fMethodWeightType == "LastMethod")
         fMethodWeight.back() = AllMethodsWeight = 1.0;
      if (AllMethodsWeight != 0.0) {
         for (Int_t i=0; i<=fMethodIndex; i++)
            fMethodWeight[i] /= AllMethodsWeight;
      }
   }

   // calculate MVA values
   Double_t meanS, meanB, rmsS, rmsB, xmin, xmax, nrms = 10;
   std::vector <Float_t>* mvaRes;
   if (singleMethod && eTT==Types::kTraining)
      mvaRes = fMVAvalues; // values already calculated
   else {  
      mvaRes = new std::vector <Float_t>(Data()->GetNEvents());
      for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
         Data()->GetEvent(ievt);
         (*mvaRes)[ievt] = singleMethod ? method->GetMvaValue(&err) : GetMvaValue(&err);
      }
   }

   // restore the method weights
   if (!singleMethod)
      fMethodWeight = OldMethodWeight;

   // now create histograms for calculation of the ROC integral
   Int_t signalClass = 0;
   if (DataInfo().GetClassInfo("Signal") != 0) {
      signalClass = DataInfo().GetClassInfo("Signal")->GetNumber();
   }
   gTools().ComputeStat( Data()->GetEventCollection(eTT), mvaRes,
                         meanS, meanB, rmsS, rmsB, xmin, xmax, signalClass );

   fNbins = gConfig().fVariablePlotting.fNbinsXOfROCCurve;
   xmin = TMath::Max( TMath::Min(meanS - nrms*rmsS, meanB - nrms*rmsB ), xmin );
   xmax = TMath::Min( TMath::Max(meanS + nrms*rmsS, meanB + nrms*rmsB ), xmax ) + 0.0001;

   // calculate ROC integral
   TH1* mva_s = new TH1F( "MVA_S", "MVA_S", fNbins, xmin, xmax );
   TH1* mva_b = new TH1F( "MVA_B", "MVA_B", fNbins, xmin, xmax );
   TH1 *mva_s_overlap=0, *mva_b_overlap=0;
   if (CalcOverlapIntergral) {
      mva_s_overlap = new TH1F( "MVA_S_OVERLAP", "MVA_S_OVERLAP", fNbins, xmin, xmax );
      mva_b_overlap = new TH1F( "MVA_B_OVERLAP", "MVA_B_OVERLAP", fNbins, xmin, xmax );
   }
   for (Long64_t ievt=0; ievt<GetNEvents(); ievt++) {
      const Event* ev = GetEvent(ievt);
      Float_t w = (eTT==Types::kTesting ? ev->GetWeight() : ev->GetOriginalWeight());
      if (DataInfo().IsSignal(ev))  mva_s->Fill( (*mvaRes)[ievt], w );
      else                          mva_b->Fill( (*mvaRes)[ievt], w );

      if (CalcOverlapIntergral) {
	 Float_t w_ov = ev->GetWeight();
	 if (DataInfo().IsSignal(ev))  
	    mva_s_overlap->Fill( (*mvaRes)[ievt], w_ov );
	 else
	    mva_b_overlap->Fill( (*mvaRes)[ievt], w_ov );
      }
   }
   gTools().NormHist( mva_s );
   gTools().NormHist( mva_b );
   PDF *fS = new PDF( "PDF Sig", mva_s, PDF::kSpline2 );
   PDF *fB = new PDF( "PDF Bkg", mva_b, PDF::kSpline2 );

   // calculate ROC integral from fS, fB
   Double_t ROC = MethodBase::GetROCIntegral(fS, fB);
   
   // calculate overlap integral
   if (CalcOverlapIntergral) {
      gTools().NormHist( mva_s_overlap );
      gTools().NormHist( mva_b_overlap );

      fOverlap_integral = 0.0;
      for (Int_t bin=1; bin<=mva_s_overlap->GetNbinsX(); bin++){
	 Double_t bc_s = mva_s_overlap->GetBinContent(bin);
	 Double_t bc_b = mva_b_overlap->GetBinContent(bin);
	 if (bc_s > 0.0 && bc_b > 0.0)
	    fOverlap_integral += TMath::Min(bc_s, bc_b);
      }

      delete mva_s_overlap;
      delete mva_b_overlap;
   }

   delete mva_s;
   delete mva_b;
   delete fS;
   delete fB;
   if (!(singleMethod && eTT==Types::kTraining))  delete mvaRes;

   Data()->SetCurrentType(Types::kTraining);

   return ROC;
}

void TMVA::MethodBoost::CalcMVAValues()
{
   // Calculate MVA values of current method fMethods.back() on
   // training sample

   Data()->SetCurrentType(Types::kTraining);
   MethodBase* method = dynamic_cast<MethodBase*>(fMethods.back());
   if (!method) {
      Log() << kFATAL << "dynamic cast to MethodBase* failed" <<Endl;
      return;
   }
   // calculate MVA values
   for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
      Data()->GetEvent(ievt);
      fMVAvalues->at(ievt) = method->GetMvaValue();
   }
}

