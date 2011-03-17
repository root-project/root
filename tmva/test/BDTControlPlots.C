#include <vector>
#include <string>
#include "tmvaglob.C"


// input: - Input file (result from TMVA),
//        - use of TMVA plotting TStyle
void BDTControlPlots( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );
  
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   // get all titles of the method BDT
   TList titles;
   UInt_t ninst = TMVAGlob::GetListOfTitles("Method_BDT",titles);
   if (ninst==0) {
      cout << "Could not locate directory 'Method_BDT' in file " << fin << endl;
      return;
   }
   // loop over all titles
   TIter keyIter(&titles);
   TDirectory *bdtdir;
   TKey *key;
   while ((key = TMVAGlob::NextKey(keyIter,"TDirectory"))) {
      bdtdir = (TDirectory *)key->ReadObj();
      bdtcontrolplots( bdtdir );
   }
}

void bdtcontrolplots( TDirectory *bdtdir ) {

   const Int_t nPlots = 6;

   Int_t width  = 900;
   Int_t height = 600;
   char cn[100];
   const TString titName = bdtdir->GetName();
   sprintf( cn, "cv_%s", titName.Data() );
   TCanvas *c = new TCanvas( cn,  Form( "%s Control Plots", titName.Data() ),
                             width, height ); 
   c->Divide(3,2);


   const TString titName = bdtdir->GetName();

   TString hname[nPlots]={"BoostMonitor","BoostWeight","BoostWeightVsTree","ErrFractHist","NodesBeforePruning",titName+"_FOMvsIterFrame"}

   for (Int_t i=0; i<nPlots; i++){
      Int_t color = 4; 
      TPad * cPad = (TPad*)c->cd(i+1);
      TH1 *h = (TH1*) bdtdir->Get(hname[i]);
      
      if (h){
         h->SetMaximum(h->GetMaximum()*1.3);
         h->SetMinimum( 0 );
         h->SetMarkerColor(color);
         h->SetMarkerSize( 0.7 );
         h->SetMarkerStyle( 24 );
         h->SetLineWidth(1);
         h->SetLineColor(color);
         if(hname[i]=="NodesBeforePruning")h->SetTitle("Nodes before/after pruning");
         h->Draw();
         if(hname[i]=="NodesBeforePruning"){
            TH1 *h2 = (TH1*) bdtdir->Get("NodesAfterPruning");
            h2->SetLineWidth(1);
            h2->SetLineColor(2);
            h2->Draw("same");
         }
         if(hname[i]=="BoostMonitor"){ // a plot only available in case of automatic parameter option tuning
            TGraph *g = (TGraph*) bdtdir->Get("BoostMonitorGraph");
            g->Draw("LP*");
         }
         if(hname[i]==titName+"_FOMvsIterFrame"){ // a plot only available in case of automatic parameter option tuning
            TGraph *g = (TGraph*) bdtdir->Get(titName+"_FOMvsIter");
            g->Draw();
         }
         c->Update();
      }
   }

   // write to file
   TString fname = Form( "plots/%s_ControlPlots", titName.Data() );
   TMVAGlob::imgconv( c, fname );
   
}


