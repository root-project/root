#include "RooRandom.h"
#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooStats/HybridCalculator.h"
#include "RooStats/HybridResult.h"
#include "RooStats/HybridPlot.h"

#include "TFile.h"
#include "TStopwatch.h"
#include "TCanvas.h"

using namespace RooFit;
using namespace RooStats;

void rs506_HybridCalculator_averageSignificance(const char* fname="WS_GaussOverFlat_withSystematics.root",int ntoys=2500,const char* outputplot="hc_averagesign_counting_syst.ps"){

  RooRandom::randomGenerator()->SetSeed(100);
  TStopwatch t;
  t.Start();

  TFile* file =new TFile(fname);
  RooWorkspace* my_WS = (RooWorkspace*) file->Get("myWS");
  if (!my_WS) return; 

  //Import the objects needed
  RooAbsPdf* model=my_WS->pdf("model");
  RooAbsPdf* priorNuisance=my_WS->pdf("priorNuisance");

  //const RooArgSet* paramInterestSet=my_WS->set("paramInterest");
  //RooRealVar* paramInterest=(RooRealVar*) paramInterestSet->first();
  RooAbsPdf* modelBkg=my_WS->pdf("modelBkg");
  const RooArgSet* nuisanceParam=my_WS->set("parameters");
  RooArgList observable(*(my_WS->set("observables") ) );
  
  HybridCalculator * hc=new HybridCalculator(*model,*modelBkg,observable);
  hc->SetNumberOfToys(ntoys);
  hc->SetTestStatistic(1);
  bool usepriors=false;
  if(priorNuisance!=0){
    hc->UseNuisance(kTRUE);
    hc->SetNuisancePdf(*priorNuisance);
    usepriors=true;
    nuisanceParam->Print();
    hc->SetNuisanceParameters(*nuisanceParam);
  }else{
    hc->UseNuisance(kFALSE);
  }
  
  RooRandom::randomGenerator()->SetSeed(0);
  HybridResult* hcresult=hc->Calculate(ntoys,usepriors);
  HybridPlot* hcplot = hcresult->GetPlot("HybridPlot","Toy MC Q ",100);
  double mean_sb_toys_test_stat = hcplot->GetSBmean();
  hcresult->SetDataTestStatistics(mean_sb_toys_test_stat);

  double mean_significance = hcresult->Significance();
  
  cout<<"significance:" <<mean_significance<<endl;
  hcplot = hcresult->GetPlot("HybridPlot","Toy MC -2ln Q ",100);
  
  TCanvas*c1=new TCanvas();
  c1->cd();
  hcplot->Draw(outputplot);
  c1->Print(outputplot);
  file->Close();
  t.Stop();
  t.Print();
}
