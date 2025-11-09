{
   gROOT->Reset();
   gROOT->SetStyle("Plain");
   gStyle->SetPalette(1);
   gStyle->SetOptFit(1);
   gStyle->SetStatW(0.28);
   gStyle->SetStatH(0.03);
   gStyle->SetStatX(0.9);
   gStyle->SetStatY(0.9);
   gStyle->SetOptStat(0100);
   gStyle->SetTitleSize(0.05,"XYZ");
   gStyle->SetTitleOffset(1.1,"Y");
   gStyle->SetLabelSize(0.04,"X");
   gStyle->SetLabelSize(0.04,"Y");
   gStyle->SetTitleBorderSize(0);
   gStyle->SetTitleX(0.1);

   Int_t ncols,Eye,Run,Event;
   Int_t nlines = 0;

#ifdef ClingWorkAroundProxyConfusion
  gInterpreter->GenerateDictionary("map<int,double>","map");
#endif

   //*************************************//
   TChain *chain1 = new TChain("recData");

   chain1->Add("S_1*.root");

   // chain1->Add("data/S_1*.root");
   // chain1->Add("data/S_2*.root");
   // chain1->Add("data/S_3*.root");
   // chain1->Add("data/S_4*.root");
   // chain1->Add("data/S_5*.root");

   //    chain1->Add("/data/data_augerobserver_v2/simulations/prot_qgs/S_1*.root");
   //    chain1->Add("/data/data_augerobserver_v2/simulations/prot_qgs/S_2*.root");
   //    chain1->Add("/data/data_augerobserver_v2/simulations/prot_qgs/S_3*.root");
   //    chain1->Add("/data/data_augerobserver_v2/simulations/prot_qgs/S_4*.root");
   //    chain1->Add("/data/data_augerobserver_v2/simulations/prot_qgs/S_5*.root");

   //chain1->Add("/data/data_augerobserver_v2/simulations/iron_qgs/S_1*.root");
   //chain1->Add("/data/data_augerobserver_v2/simulations/iron_qgs/S_2*.root");
   //chain1->Add("/data/data_augerobserver_v2/simulations/iron_qgs/S_3*.root");
   //chain1->Add("/data/data_augerobserver_v2/simulations/iron_qgs/S_4*.root");
   //chain1->Add("/data/data_augerobserver_v2/simulations/iron_qgs/S_5*.root");

   // chain1->Draw("event.fGenShower.fEnergy")



   TCut pixelcut = "event.fFDEvents.fFdRecPixel.fnPixel>=6";

   TCut profchisqcut = "((event.fFDEvents.fFdRecShower.fGHChi2/event.fFDEvents.fFdRecShower.fGHNdf)<6)";

   TCut hybridflagcut1 = "(abs(event.fFDEvents.fFdRecGeometry.fSDFDdT<200))";

   TCut hybridflagcut2 = "(event.fFDEvents.fFdRecGeometry.fTimeFitFDChi2/(event.fFDEvents.fFdRecPixel.fnPixel-3)<5)";

   TCut hybridflagcut3 = "(event.fFDEvents.fFdRecGeometry.fAxisDist<2000)";

   TCut chkovfraccut = "(event.fFDEvents.fFdRecShower.fChkovFrac<50)";

   TCut xtrackcut1 = "((event.fFDEvents.fFdRecShower.fXTrackMin+30)<event.fFDEvents.fFdRecShower.fXmax && event.fFDEvents.fFdRecShower.fXmax<(event.fFDEvents.fFdRecShower.fXTrackMax-30))";

   TCut xtrackcut2 = "(event.fFDEvents.fFdRecShower.fXTrackMax-event.fFDEvents.fFdRecShower.fXTrackMin>300)";

   TCut rp_cut = "event.fFDEvents.fFdRecGeometry.fRp>0";

   TCut chi0_cut ="(event.fFDEvents.fFdRecGeometry.fChi0>0)&&(event.fFDEvents.fFdRecGeometry.fChi0<180)";

   TCut fdenergycut0 = "((log10(event.fGenShower.fEnergy)>=17.25)&&(log10(event.fGenShower.fEnergy)<17.50))";
   TCut fdenergycut1 = "((log10(event.fGenShower.fEnergy)>=17.50)&&(log10(event.fGenShower.fEnergy)<17.75))";
   TCut fdenergycut2 = "((log10(event.fGenShower.fEnergy)>=17.75)&&(log10(event.fGenShower.fEnergy)<18.25))";
   TCut fdenergycut3 = "((log10(event.fGenShower.fEnergy)>=18.25)&&(log10(event.fGenShower.fEnergy)<18.75))";
   TCut fdenergycut4 = "((log10(event.fGenShower.fEnergy)>=18.75)&&(log10(event.fGenShower.fEnergy)<19.25))";
   TCut fdenergycut5 = "((log10(event.fGenShower.fEnergy)>=19.25)&&(log10(event.fGenShower.fEnergy)<=19.75))";
   TCut xmaxcut0 = "(event.fFDEvents.fFdRecShower.fXmax>0)";
   TCut thetacut = "(event.fGenShower.fAxisCoreCS.Theta()<=0.5)";

   TCut BasicAntiBias_sim = " event.fFDEvents.fFdRecShower.fXFOVMin > 0 && event.fFDEvents.fFdRecShower.fXmax > event.fFDEvents.fFdRecShower.fXFOVMin && event.fFDEvents.fFdRecShower.fXmax < event.fFDEvents.fFdRecShower.fXFOVMax";

   TCanvas* XmaxVsXlow = new TCanvas("XmaxVsXlow","XmaxVsXlow",1080,720);

   XmaxVsXlow->Draw();
   XmaxVsXlow->Divide(3,2);
   XmaxVsXlow->SetGrid();

   XmaxVsXlow->cd(1);
#ifdef ClingWorkAroundMissingDynamicScope
   TPad *XmaxVsXlow_1;
   XmaxVsXlow_1 = dynamic_cast<TPad*>( (TPad*)gROOT->FindObject("XmaxVsXlow_1") );
#endif
   XmaxVsXlow_1->SetGrid();
   auto Xlow170 = new TProfile("Xlow170","17.25<LogE<17.5",30,400,1200);
   //Xlow170->SetMinimum(500);
   //Xlow170->SetMaximum(900);
   Xlow170->SetYTitle("Xmax(g/cm^{2}) ");
   Xlow170->SetXTitle("XFOVMin(g/cm^{2}) ");
   Xlow170->SetTitle("17.25#leqLogE<17.5");

   chain1->Draw("event.fFDEvents.fFdRecShower.fXFOVMin",fdenergycut1 && xmaxcut0 && thetacut && pixelcut && profchisqcut && hybridflagcut1 && hybridflagcut2 && hybridflagcut3 && chkovfraccut && xtrackcut1 && xtrackcut2 && rp_cut && chi0_cut && BasicAntiBias_sim,"");

   XmaxVsXlow->Modified();
   XmaxVsXlow->Update();
}
