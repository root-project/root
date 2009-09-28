//////////////////////////////////////////////////////////////////////////
//
// RooStats tutorial macro #503
// 2009/08 - Nils Ruthmann, Gregory Schott
//
///////////////////////////////////////////////////////////////////////

#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooRandom.h"
#include "RooMCStudy.h"
#include "RooDataSet.h"

#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/HypoTestResult.h"
#include "RooStats/UpperLimitMCSModule.h"



#include "TFile.h"
#include "TStopwatch.h"
#include "TCanvas.h"



void rs503_ProfileLikelihoodCalculator_averageLimit(const char* fname="WS_GaussOverFlat.root",int ntoys=1000,const char* outputplot="pll_avLimit.ps"){
  using namespace RooFit ;
  using namespace RooStats ;
  TStopwatch t;
  t.Start();

  TFile* file =new TFile(fname);
  RooWorkspace* my_WS = (RooWorkspace*) file->Get("myWS");
  std::cout<<"ws open"<<endl;
  //Import the objects needed
  RooAbsPdf* model_naked=my_WS->pdf("model");
  RooAbsPdf* priorNuisance=my_WS->pdf("priorNuisance");
  RooAbsPdf* modelBkg_naked=my_WS->pdf("modelBkg");
  const RooArgSet* paramInterestSet=my_WS->set("POI");
  RooRealVar* paramInterest= (RooRealVar*) paramInterestSet->first();
  const RooArgSet* observable=my_WS->set("observables");
  const RooArgSet* nuisanceParam=my_WS->set("parameters");
  //If there are nuisance parameters present, multiply their prior to the model
  RooAbsPdf* model=model_naked;
  RooAbsPdf* modelBkg=modelBkg_naked;
  if(priorNuisance!=0) {
     model=new RooProdPdf("constrainedModel","Model with nuisance parameters",*model_naked,*priorNuisance);
     //From now work with the product of both
     modelBkg=new RooProdPdf("constrainedBkgModel","Bkg Model with nuisance parameters",*modelBkg_naked,*priorNuisance);
  }
  else{
    std::cout<<"No Nuisance Parameters present"<<std::endl;
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
 
 

  
 
  RooRandom::randomGenerator()->SetSeed(100);

  //--------------------------------------------------------------------
  //ROOMCSTUDY 
  //For simplicity use RooMCStudy, a tool to perform several Toy MCs.
  //If there are systematics constrain them
  //Generate the background and fit for the signal to test limitsetting
  RooMCStudy* mcs = 0; 
  if(priorNuisance!=0) {
    mcs = new RooMCStudy(*modelBkg,*observable,Extended(kTRUE),FitModel(*model),
				     FitOptions(Extended(kTRUE),PrintEvalErrors(1)),Constrain(*nuisanceParam)) ;
  }
  else {
    mcs = new RooMCStudy(*modelBkg,*observable,Extended(kTRUE),FitModel(*model),
				     FitOptions(Extended(kTRUE),PrintEvalErrors(1))); 
  }

  //Adding a module which allows to compute the upper limit in every generation cycle
  UpperLimitMCSModule limitModule(paramInterestSet,0.95) ;
  mcs->addModule(limitModule) ;
  
  cout<<"start fit process"<<endl;
  mcs->generateAndFit(ntoys);

  TString limitstr("ul_");
  TH1* mcslimit_histo=(TH1F*)mcs->fitParDataSet().createHistogram(limitstr+paramInterest->GetName());
  //Make a histogram of the upperlimit vs the number of generated events
  TString limitstr2("ngen,ul_");
  TH1* mcslimitvsevt_histo=(TH1F*)mcs->fitParDataSet().createHistogram(limitstr2+paramInterest->GetName());
  TCanvas* c2 =new TCanvas();
  c2->Divide(1,2);
  c2->cd(1);
  mcslimit_histo->Draw();
  c2->cd(2);
  mcslimitvsevt_histo->Draw();
  c2->Print(outputplot);
  //file->Close();
  t.Stop();
  t.Print();

}


