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


void rs505_HybridCalculator_significance(const char* fname="WS_GaussOverFlat_withSystematics.root",int ntoys=5000,const char* outputplot="hc_sign_shape_nosyst.pdf"){

  RooRandom::randomGenerator()->SetSeed(100);
  TStopwatch t;
  t.Start();
  TFile* file =new TFile(fname);
  RooWorkspace* my_WS = (RooWorkspace*) file->Get("myWS");
  if (!my_WS) return; 
  //Import the objects needed
  RooAbsPdf* model=my_WS->pdf("model");
  RooAbsData* data=my_WS->data("data");
  RooAbsPdf* priorNuisance=my_WS->pdf("priorNuisance");
  //const RooArgSet* paramInterestSet=my_WS->set("POI");
  //RooRealVar* paramInterest= (RooRealVar*) paramInterestSet->first();
  RooAbsPdf* modelBkg=my_WS->pdf("modelBkg");
  //const RooArgSet* observable=my_WS->set("observables");
  const RooArgSet* nuisanceParam=my_WS->set("parameters");
  
 
  HybridCalculator * hc=new HybridCalculator("hc","HybridCalculator",*data,*model,*modelBkg);
  hc->SetNumberOfToys(ntoys);
  bool usepriors=false;

  if(priorNuisance!=0){
    hc->UseNuisance(kTRUE);
    hc->SetNuisancePdf(*priorNuisance);
    usepriors=true;
    cout<<"The following nuisance parameters are taken into account:"<<endl;
    nuisanceParam->Print();
    hc->SetNuisanceParameters(*nuisanceParam);
  }else{
    hc->UseNuisance(kFALSE);
  }
  
  HybridResult* hcresult=hc->GetHypoTest();
  double clsb_data = hcresult->CLsplusb();
  double clb_data = hcresult->CLb();
  double cls_data = hcresult->CLs();
  double data_significance = hcresult->Significance();
  
  cout<<"CL_b:"<<clb_data<<endl;
  cout<<"CL_s:"<<cls_data<<endl;
  cout<<"CL_sb:"<<clsb_data<<endl;
  cout<<"significance:" <<data_significance<<endl;
  
  HybridPlot* hcPlot=hcresult->GetPlot("hcPlot","p Values Plot",100);
  TCanvas*c1=new TCanvas();
  c1->cd();
  hcPlot->Draw();
  c1->Print(outputplot);
  file->Close();
  t.Stop();
  t.Print();
  
}
