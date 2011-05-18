#include "tmvaglob.C"
#include "TControlBar.h"
#include <sstream>
#include <string>
#include <cfloat>


typedef enum { kNEV, kDISCR, kMONO, kRMS, kRMSOVMEAN } EPlotType;
typedef enum { kSEPARATE, kUNIFIED, kMONOTARGET, kMULTITARGET } EFoamType;

void PlotFoams( TString fin = "weights/TMVAClassification_PDEFoam.weights_foams.root", 
                bool useTMVAStyle=kTRUE )
{
   cout << "read file: " << fin << endl;
   TFile *file = TFile::Open(fin);

   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // create control bar
   TControlBar* cbar = new TControlBar( "vertical", "Choose cell value for plot:", 50, 50 );
   if ((gDirectory->Get("SignalFoam") && gDirectory->Get("BgFoam")) || 
       gDirectory->Get("MultiTargetRegressionFoam")) {
      TString macro = Form( "Plot(\"%s\", kNEV)", fin.Data() );
      cbar->AddButton( "Event density", macro, "Plot event density", "button" );
   } else if (gDirectory->Get("DiscrFoam")){
      TString macro = Form( "Plot(\"%s\", kDISCR)", fin.Data() );
      cbar->AddButton( "Discriminator", macro, "Plot discriminator", "button" );
   } else if (gDirectory->Get("MonoTargetRegressionFoam")){
      TString macro = Form( "Plot(\"%s\", kMONO)", fin.Data() );
      cbar->AddButton( "Target", macro, "Plot target", "button" );
   } else {
      cout << "Error: no foams found in file: " << fin << endl;
      return;
   }
   
   TString macro_rms = Form( "Plot(\"%s\", kRMS)", fin.Data() );
   cbar->AddButton( "RMS", macro_rms, "Plot RMS (Variance)", "button" );
   TString macro_rms_ov_mean = Form( "Plot(\"%s\", kRMSOVMEAN)", fin.Data() );
   cbar->AddButton( "RMS over Mean", macro_rms_ov_mean, "Plot RMS over Mean", "button" );

   cbar->Show();
   file->Close();
}

// foam plotting macro
void Plot( TString fin = "weights/TMVAClassification_PDEFoam.weights_foams.root", EPlotType pt )
{
   cout << "read file: " << fin << endl;
   TFile *file = TFile::Open(fin);

   gStyle->SetNumberContours(999);
   TMVAGlob::SetTMVAStyle();
   
   string cellval      = ""; // quantity to draw in foam projection
   string cellval_long = ""; // name of quantity to draw in foam projection

   if (pt == kNEV){
      cellval      = "cell_value";
      cellval_long = "Event density";
   }
   else if (pt == kDISCR){
      cellval      = "cell_value";
      cellval_long = "Discriminator";
   }
   else if (pt == kMONO){
      cellval      = "cell_value";
      cellval_long = "Target";
   }
   else if (pt == kRMS){
      cellval      = "rms";
      cellval_long = "RMS";
   }
   else if (pt == kRMSOVMEAN){
      cellval      = "rms_ov_mean";
      cellval_long = "RMS/Mean";
   }

   // find foams and foam type
   EFoamType ft;
   TMVA::PDEFoam *foam  = 0;
   TMVA::PDEFoam *foam2 = 0;
   string foam_capt, foam2_capt;
   if (gDirectory->Get("SignalFoam") && gDirectory->Get("BgFoam")){
      foam  = SignalFoam;
      foam2 = BgFoam;
      foam_capt  = "Signal Foam";
      foam2_capt = "Background Foam";
      ft    = kSEPARATE;
   } else if (gDirectory->Get("DiscrFoam")){
      foam = DiscrFoam;
      foam_capt = "Discriminator Foam";
      ft   = kDISCR;
   } else if (gDirectory->Get("MonoTargetRegressionFoam")){
      foam = MonoTargetRegressionFoam;
      foam_capt = "MonoTargetRegression Foam";
      ft   = kMONOTARGET;
   } else if (gDirectory->Get("MultiTargetRegressionFoam")){
      foam = MultiTargetRegressionFoam;
      foam_capt = "MultiTargetRegression Foam";
      ft   = kMULTITARGET;
   } else {
      cout << "ERROR: no Foams found in file: " << fin << endl;
      return;
   }

   Int_t kDim = foam->GetTotDim();
   cout << foam_capt << " loaded" << endl;
   cout << "Dimension of foam: " << kDim << endl;

   // ********** plot foams ********** //
   if (kDim==1){
      // draw histogram
      TH1D *hist1 = 0, *hist2 = 0;
      TCanvas *canv = new TCanvas("canv", "Foam(s)", 400, (ft==kSEPARATE) ? 800 : 400);
      if (ft==kSEPARATE)
	 canv->Divide(0,2);
      canv->cd(1);

      string var_name = foam->GetVariableName(0)->String();
      hist1 = foam->Draw1Dim(cellval.c_str(), 100);
      hist1->SetTitle((cellval_long+" of "+foam_capt+";"+var_name).c_str());
      hist1->Draw();
      hist1->SetDirectory(0);
      
      if (ft==kSEPARATE){
	 canv->cd(2);
	 string var_name2 = foam2->GetVariableName(0)->String();
	 if (ft==kSEPARATE)
	    hist2 = foam2->Draw1Dim(cellval.c_str(), 100);
	 hist2->SetTitle((cellval_long+" of "+foam2_capt+";"+var_name2).c_str());
	 hist2->Draw();
	 hist2->SetDirectory(0);
      }

      // save canvas to file
      stringstream fname  (stringstream::in | stringstream::out);
      fname << "plots/" << "foam_var_" << cellval << "_0";
      canv->Update();
      TMVAGlob::imgconv( canv, fname.str() );
   } else{ 
      // if dimension of foam > 1, draw foam projections
      TCanvas* canv=0;
      TH2D *proj=0, *proj2=0;

      // draw all possible projections (kDim*(kDim-1)/2)
      for(Int_t i=0; i<kDim; i++){
	 for (Int_t k=i+1; k<kDim; k++){

	    // set titles of canvas and foam projections
	    stringstream title (stringstream::in | stringstream::out);
	    stringstream caption (stringstream::in | stringstream::out);
	    title   << "combined_"         << i << ":" << k;
	    caption << "Foam projections " << i << ":" << k;
	    cout    << "draw projection: " << i << ":" << k << endl;
                     
	    stringstream title_proj1 (stringstream::in | stringstream::out);
	    stringstream title_proj2 (stringstream::in | stringstream::out);
	    title_proj1 << cellval_long << " of " 
			<< foam_capt << ": Projection " 
			<< foam->GetVariableName(i)->String()
			<< ":" << foam->GetVariableName(k)->String()
			<< ";" << foam->GetVariableName(i)->String()
			<< ";" << foam->GetVariableName(k)->String();
	    if (ft==kSEPARATE){
	       title_proj2 << cellval_long << " of " 
			   << foam2_capt << ": Projection " 
			   << foam2->GetVariableName(i)->String() 
			   << ":" << foam2->GetVariableName(k)->String()
			   << ";" << foam2->GetVariableName(i)->String()
			   << ";" << foam2->GetVariableName(k)->String();
	    }

	    // create canvas
	    canv = new TCanvas(title.str().c_str(), caption.str().c_str(), 
			       (Int_t)(400/(1.-0.2)), (ft==kSEPARATE ? 800 : 400));
	    if (ft==kSEPARATE){
	       canv->Divide(0,2);
	       canv->GetPad(1)->SetRightMargin(0.2);
	       canv->GetPad(2)->SetRightMargin(0.2);
	    } else {
	       canv->SetRightMargin(0.2);
	    }
	    canv->cd(1);

	    // do projections
	    proj = foam->Project2(i, k, cellval.c_str(), "kNone");
	    proj->SetTitle(title_proj1.str().c_str());
	    if (pt==kDISCR)
	       proj->GetZaxis()->SetRangeUser(-DBL_EPSILON, 1.+DBL_EPSILON);
	    proj->Draw("COLZ"); // CONT4Z
	    proj->SetDirectory(0);

	    if (ft==kSEPARATE){
	       canv->cd(2);
	       proj2 = foam2->Project2(i, k, cellval.c_str(), "kNone");
	       proj2->SetTitle(title_proj2.str().c_str());
	       proj2->Draw("COLZ"); // CONT4Z
	       proj2->SetDirectory(0);
	    }
                     
	    // save canvas to file
	    stringstream fname  (stringstream::in | stringstream::out);
	    fname << "plots/" << "foam_projection_var_" << cellval << "_" << i << ":" << k;
	    canv->Update();
	    TMVAGlob::imgconv( canv, fname.str() );
	 } // loop over all possible projections
      } // loop over all possible projections
   } // if dimension > 1

   file->Close();
}
