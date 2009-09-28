//////////////////////////////////////////////////////////////////////////
//
// RooStats tutorial macro #504
// 2009/08 - Nils Ruthmann, Gregory Schott
//
///////////////////////////////////////////////////////////////////////

#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooRandom.h"
#include "RooMCStudy.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooDLLSignificanceMCSModule.h"

#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/HypoTestResult.h"
#include "RooStats/UpperLimitMCSModule.h"



#include "TFile.h"
#include "TStopwatch.h"
#include "TCanvas.h"



void rs504_ProfileLikelihoodCalculator_averageSignificance(const char* fname="WS_GaussOverFlat_withSystematics.root",int ntoys=100,const char* outputplot="pll_avSign.ps"){
  using namespace RooFit;
  using namespace RooStats;
  TStopwatch t;
  t.Start();

  TFile* file =new TFile(fname);
  RooWorkspace* my_WS = (RooWorkspace*) file->Get("myWS");
  //Import the objects needed
  RooAbsPdf* model_naked=my_WS->pdf("model");
  RooAbsPdf* priorNuisance=my_WS->pdf("priorNuisance");
  const RooArgSet* paramInterestSet=my_WS->set("POI");
  RooRealVar* paramInterest= (RooRealVar*) paramInterestSet->first();
  const RooArgSet* observable=my_WS->set("observables");
  const RooArgSet* nuisanceParam=my_WS->set("parameters");
   
  //If there are nuisance parameters present, multiply their prior to the model
  RooAbsPdf* model=model_naked;
  if(priorNuisance!=0) {
    model=new RooProdPdf("constrainedModel","Model with nuisance parameters",*model_naked,*priorNuisance);
    //From now work with the product of both
  }

  //Save the default values of the parameters:
   RooArgSet* parameters=model->getVariables();
   RooArgSet* default_parameters=new RooArgSet("default_parameters");
   TIterator* it=parameters->createIterator();
   RooRealVar* currentparam=(RooRealVar*) it->Next();
   do {
     default_parameters->addClone(*currentparam,false);
     currentparam=(RooRealVar*) it->Next();
   }while(currentparam!=0);
   
  
  if(priorNuisance!=0)
    RooFormulaVar nll_nuisance("nllSyst","-TMath::Log(@0)",RooArgList(*priorNuisance));
  else
    RooFormulaVar nll_nuisance("nllSyst","0",RooArgList(*priorNuisance));
  
 
  RooRandom::randomGenerator()->SetSeed(110);

  //--------------------------------------------------------------------
  //ROOMCSTUDY 
  
  //For simplicity use RooMCStudy.
  RooMCStudy* mcs; 
  if (nuisanceParam)
     mcs = new RooMCStudy(*model,*observable,Extended(kTRUE),
				     FitOptions(Extended(kTRUE),PrintEvalErrors(-1),Minos(kFALSE)),Constrain(*nuisanceParam)) ;
  else 
     mcs = new RooMCStudy(*model,*observable,Extended(kTRUE),
				      FitOptions(Extended(kTRUE),Minos(kFALSE),PrintEvalErrors(-1))) ;

  //Adding a module which allows to compute the significance in each toy experiment
  RooDLLSignificanceMCSModule sigModule(*paramInterest,0) ;
  //If there are nuisance parameters present, they should be generated according to their pdf for every new toy experiment.
  //this is done using a MCSModule
  mcs->addModule(sigModule);
  mcs->generateAndFit(ntoys);

  TString signstring("significance_nullhypo_");
  TH1* mcssign_histo=(TH1F*)mcs->fitParDataSet().createHistogram(signstring+paramInterest->GetName());
 
  TCanvas* c2 =new TCanvas();
  c2->Divide(2,2);
  c2->cd(1);
  mcssign_histo->Draw();
  c2->cd(2);
  mcs->plotPull(*paramInterest)->Draw();
  c2->cd(3);
  if (my_WS->var("B")) { mcs->plotParam(*(my_WS->var("B")))->Draw();
     c2->cd(4);
     mcs->plotError(*(my_WS->var("B")))->Draw();
  }
  c2->Print(outputplot);
   std::cout<<"The average significance after "<<ntoys<<" toys is: "<<mcssign_histo->GetMean()<<std::endl;
  
   //file->Close();
  t.Stop();
  t.Print();
    
}
