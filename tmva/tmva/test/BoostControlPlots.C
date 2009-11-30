#include <vector>
#include <string>
#include "tmvaglob.C"


// input: - Input file (result from TMVA),
//        - use of TMVA plotting TStyle
// this macro is based on BDTControlPlots.C
void BoostControlPlots( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );
  
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   // get all titles of the method Boost
   TList titles;
   UInt_t ninst = TMVAGlob::GetListOfTitles("Method_Boost",titles);
   if (ninst==0) {
      cout << "Could not locate directory 'Method_Boost' in file " << fin << endl;
      return;
   }
   // loop over all titles
   TIter keyIter(&titles);
   TDirectory *boostdir;
   TKey *key;
   while ((key = TMVAGlob::NextKey(keyIter,"TDirectory"))) {
      boostdir = (TDirectory *)key->ReadObj();
      boostcontrolplots( boostdir );
   }
}

void boostcontrolplots( TDirectory *boostdir ) {

   const Int_t nPlots = 4;

   Int_t width  = 900;
   Int_t height = 600;
   char cn[100];
   const TString titName = boostdir->GetName();
   sprintf( cn, "cv_%s", titName.Data() );
   TCanvas *c = new TCanvas( cn,  Form( "%s Control Plots", titName.Data() ),
                             width, height ); 
   c->Divide(2,2);


   const TString titName = boostdir->GetName();

   TString hname[nPlots]={"Booster_BoostWeight","Booster_MethodWeight","Booster_ErrFraction","Booster_OrigErrFraction"};

   for (Int_t i=0; i<nPlots; i++){
      Int_t color = 4; 
      TPad * cPad = (TPad*)c->cd(i+1);
      TH1 *h = (TH1*) boostdir->Get(hname[i]);
      TString plotname = h->GetName();
      h->SetMaximum(h->GetMaximum()*1.3);
      h->SetMinimum( 0 );
      h->SetMarkerColor(color);
      h->SetMarkerSize( 0.7 );
      h->SetMarkerStyle( 24 );
      h->SetLineWidth(2);
      h->SetLineColor(color);
      h->Draw();
      c->Update();
   }

   // write to file
   TString fname = Form( "plots/%s_ControlPlots", titName.Data() );
   TMVAGlob::imgconv( c, fname );
   
}


