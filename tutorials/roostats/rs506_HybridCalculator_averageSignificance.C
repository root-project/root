

void rs506_HybridCalculator_averageSignificance(const char* fname="WS_counting_syst.root",int ntoys=2500,const char* outputplot="hc_averagesign_counting_syst.ps"){
  using namespace RooFit;
  using namespace RooStats;
  RooRandom::randomGenerator()->SetSeed(100);
  TStopwatch t;
  t.Start();

  TFile* file =new TFile(fname);
  RooWorkspace* my_WS = (RooWorkspace*) file->Get("my_WS");
  //Import the objects needed
  RooAbsPdf* model=my_WS->pdf("model");
  RooAbsPdf* priorNuisance=my_WS->pdf("priorNuisance");

  RooArgSet* paramInterestSet=my_WS->set("paramInterest");
  RooRealVar* paramInterest=paramInterestSet->first();
  RooAbsPdf* modelBkg=my_WS->pdf("modelBkg");
  RooArgSet* observable=my_WS->set("observable");
  RooArgSet* nuisanceParam=my_WS->set("parameters");
  
  HybridCalculator * hc=new HybridCalculator("hc","HybridCalculator",*model,*modelBkg,*observable);
  hc->SetNumberOfToys(ntoys);
  hc->SetTestStatistics(1);
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
