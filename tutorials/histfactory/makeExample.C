
void makeDataDriven()
{
  
  TFile* file = new TFile("dataDriven.root", "RECREATE");

  TH1F* FlatHist = new TH1F("FlatHist","FlatHist", 2,0,2);
  FlatHist->SetBinContent( 1, 1.0 );
  FlatHist->SetBinContent( 2, 1.0 );
    
  TH1F* Signal = new TH1F("Signal", "Signal", 2,0,2);
  Signal->SetBinContent(1, 10);
  Signal->SetBinContent(2, 80);

  // MC Background
  TH1F* Background1 = new TH1F("Background1", "Background1", 2,0,2);
  Background1->SetBinContent(1, 20);
  Background1->SetBinContent(2, 20);

  // This is the "Data-Driven" measurement
  // that represents Background2
  // Assume the extrapolation factor is 1.0
  TH1F* ControlRegion = new TH1F("ControlRegion", "ControlRegion", 2,0,2);
  ControlRegion->SetBinContent(1, 75);
  ControlRegion->SetBinContent(2, 35);


  TH1F* StatUncert = new TH1F("StatUncert", "StatUncert", 2,0,2);
  StatUncert->SetBinContent(1, .15);  // 15% uncertainty
  StatUncert->SetBinContent(2, .15);  // 15% uncertainty


  TH1F* data = new TH1F("data", "data", 2,0,2);
  data->SetBinContent(1, 90);
  data->SetBinContent(2, 110);


  file->Write();
  file->Close();


}



void makeShapeSys2DDataset()
{
  
  TFile* file = new TFile("ShapeSys2D.root", "RECREATE");
    
  TH2F* signal = new TH2F("signal", "signal", 2,0,2, 2,0,2);
  signal->SetBinContent(1, 1, 10);
  signal->SetBinContent(2, 1, 10);
  signal->SetBinContent(1, 2, 20);
  signal->SetBinContent(2, 2, 20);

  // Background 1
  TH2F* background1 = new TH2F("background1", "background1", 2,0,2, 2,0,2);
  background1->SetBinContent(1, 1, 100);
  background1->SetBinContent(2, 1, 100);
  background1->SetBinContent(1, 2, 10);
  background1->SetBinContent(2, 2, 10);

  // Background 1 Error
  TH2F* bkg1ShapeError = new TH2F("bkg1ShapeError", "bkg1ShapeError", 2,0,2, 2,0,2);
  bkg1ShapeError->SetBinContent(1, 1, .10);  // 10%
  bkg1ShapeError->SetBinContent(2, 1, .15);  // 15%
  bkg1ShapeError->SetBinContent(1, 2, .10);  // 10%
  bkg1ShapeError->SetBinContent(2, 2, .15);  // 15%


  // Background 2
  TH2F* background2 = new TH2F("background2", "background2", 2,0,2, 2,0,2);
  background2->SetBinContent(1, 1, 10);
  background2->SetBinContent(2, 1, 10);
  background2->SetBinContent(1, 2, 100);
  background2->SetBinContent(2, 2, 100);

  // Background 2 Error
  TH2F* bkg2ShapeError = new TH2F("bkg2ShapeError", "bkg2ShapeError", 2,0,2, 2,0,2);
  bkg2ShapeError->SetBinContent(1, 1, .05);  // 5%
  bkg2ShapeError->SetBinContent(2, 1, .20);  // 20%
  bkg2ShapeError->SetBinContent(1, 2, .05);  // 5%
  bkg2ShapeError->SetBinContent(2, 2, .20);  // 20%


  TH2F* data = new TH2F("data", "data", 2,0,2, 2,0,2);
  data->SetBinContent(1, 1, 122);
  data->SetBinContent(2, 1, 122);
  data->SetBinContent(1, 2, 132);
  data->SetBinContent(2, 2, 132);

  file->Write();
  file->Close();


}


void makeShapeSysDataset()
{
  
  TFile* file = new TFile("ShapeSys.root", "RECREATE");
    
  TH1F* signal = new TH1F("signal", "signal", 2,0,2);
  signal->SetBinContent(1, 20);
  signal->SetBinContent(2, 10);

  // Background 1
  TH1F* background1 = new TH1F("background1", "background1", 2,0,2);
  background1->SetBinContent(1, 100);
  background1->SetBinContent(2, 0);

  // Background 1 Error
  TH1F* bkg1ShapeError = new TH1F("bkg1ShapeError", "bkg1ShapeError", 2,0,2);
  bkg1ShapeError->SetBinContent(1, .10);  // 10%
  bkg1ShapeError->SetBinContent(2, .15);  // 15%


  // Background 2
  TH1F* background2 = new TH1F("background2", "background2", 2,0,2);
  background2->SetBinContent(1, 0);
  background2->SetBinContent(2, 100);

  // Background 2 Error
  TH1F* bkg2ShapeError = new TH1F("bkg2ShapeError", "bkg2ShapeError", 2,0,2);
  bkg2ShapeError->SetBinContent(1, .05);  // 5%
  bkg2ShapeError->SetBinContent(2, .20);  // 20%


  TH1F* data = new TH1F("data", "data", 2,0,2);
  data->SetBinContent(1, 122);
  data->SetBinContent(2, 112);

  file->Write();
  file->Close();


}


void makeStatErrorDataSet()
{
  
  TFile* file = new TFile("StatError.root", "RECREATE");

  TH1F* FlatHist = new TH1F("FlatHist","FlatHist", 2,0,2);
  FlatHist->SetBinContent( 1, 1.0 );
  FlatHist->SetBinContent( 2, 1.0 );
    
  TH1F* signal = new TH1F("signal", "signal", 2,0,2);
  signal->SetBinContent(1, 20);
  signal->SetBinContent(2, 10);

  // MC background
  TH1F* background1 = new TH1F("background1", "background1", 2,0,2);
  background1->SetBinContent(1, 100);
  background1->SetBinContent(2, 0);

  // A small statistical uncertainty
  TH1F* bkg1StatUncert = new TH1F("bkg1StatUncert", "bkg1StatUncert", 2,0,2);
  bkg1StatUncert->SetBinContent(1, .05);  // 5%  uncertainty
  bkg1StatUncert->SetBinContent(2, .10);  // 10% uncertainty

  // MC background
  TH1F* background2 = new TH1F("background2", "background2", 2,0,2);
  background2->SetBinContent(1, 0);
  background2->SetBinContent(2, 100);

  TH1F* data = new TH1F("data", "data", 2,0,2);
  data->SetBinContent(1, 122);
  data->SetBinContent(2, 112);


  file->Write();
  file->Close();


}



void makeSimpleExample(){
  TFile* example = new TFile("example.root","RECREATE");
  TH1F* data = new TH1F("data","data", 2,1,2);
  TH1F* signal = new TH1F("signal","signal histogram (pb)", 2,1,2);
  TH1F* background1 = new TH1F("background1","background 1 histogram (pb)", 2,1,2);
  TH1F* background2 = new TH1F("background2","background 2 histogram (pb)", 2,1,2);
  TH1F* statUncert = new TH1F("background1_statUncert", "statUncert", 2,1,2);

  // run with 1 pb
  data->SetBinContent(1,122);
  data->SetBinContent(2,112);

  signal->SetBinContent(1,20);
  signal->SetBinContent(2,10);

  background1->SetBinContent(1,100);
  background2->SetBinContent(2,100);

  // A small statistical uncertainty
  statUncert->SetBinContent(1, .05);  // 5% uncertainty
  statUncert->SetBinContent(2, .05);  // 5% uncertainty


  example->Write();
  example->Close();
  //////////////////////


}

makeExample(){
  makeDataDriven();
  makeShapeSys2DDataset();
  makeShapeSysDataset();
  makeStatErrorDataSet();
  makeSimpleExample();
}
