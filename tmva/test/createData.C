// plot the variables
#include "TROOT.h"
#include "TMath.h"
#include "TTree.h"
#include "TArrayD.h"
#include "TStyle.h"
#include "TFile.h"
#include "TRandom.h"
#include "Riostream.h"
#include "TCanvas.h"
#include "TMatrixD.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TBranch.h"
#include <vector>

void plot( TString fname = "data.root", TString var0="var0", TString var1="var1" ) 
{
   TFile* dataFile = TFile::Open( fname );

   if (!dataFile) {
      cout << "ERROR: cannot open file: " << fname << endl;
      return;
   }

   TTree *treeS = (TTree*)dataFile->Get("TreeS");
   TTree *treeB = (TTree*)dataFile->Get("TreeB");

   TCanvas* c = new TCanvas( "c", "", 0, 0, 550, 550 );

   TStyle *TMVAStyle = gROOT->GetStyle("Plain"); // our style is based on Plain
   TMVAStyle->SetOptStat(0);
   TMVAStyle->SetPadTopMargin(0.02);
   TMVAStyle->SetPadBottomMargin(0.16);
   TMVAStyle->SetPadRightMargin(0.03);
   TMVAStyle->SetPadLeftMargin(0.15);
   TMVAStyle->SetPadGridX(0);
   TMVAStyle->SetPadGridY(0);
   
   TMVAStyle->SetOptTitle(0);
   TMVAStyle->SetTitleW(.4);
   TMVAStyle->SetTitleH(.10);
   TMVAStyle->SetTitleX(.5);
   TMVAStyle->SetTitleY(.9);
   TMVAStyle->SetMarkerStyle(20);
   TMVAStyle->SetMarkerSize(1.6);
   TMVAStyle->cd();


   Float_t xmin = TMath::Min( treeS->GetMinimum( var0 ), treeB->GetMinimum( var0 ) );
   Float_t xmax = TMath::Max( treeS->GetMaximum( var0 ), treeB->GetMaximum( var0 ) );
   Float_t ymin = TMath::Min( treeS->GetMinimum( var1 ), treeB->GetMinimum( var1 ) );
   Float_t ymax = TMath::Max( treeS->GetMaximum( var1 ), treeB->GetMaximum( var1 ) );

   Int_t nbin = 500;
   TH2F* frameS = new TH2F( "DataS", "DataS", nbin, xmin, xmax, nbin, ymin, ymax );
   TH2F* frameB = new TH2F( "DataB", "DataB", nbin, xmin, xmax, nbin, ymin, ymax );

   // project trees
   treeS->Draw( Form("%s:%s>>DataS",var1.Data(),var0.Data()), "", "0" );
   treeB->Draw( Form("%s:%s>>DataB",var1.Data(),var0.Data()
), "", "0" );

   // set style
   frameS->SetMarkerSize( 0.1 );
   frameS->SetMarkerColor( 4 );

   frameB->SetMarkerSize( 0.1 );
   frameB->SetMarkerColor( 2 );

   // legend
   frameS->SetTitle( var1+" versus "+var0+" for signal and background" );
   frameS->GetXaxis()->SetTitle( var0 );
   frameS->GetYaxis()->SetTitle( var1 );

   frameS->SetLabelSize( 0.04, "X" );
   frameS->SetLabelSize( 0.04, "Y" );
   frameS->SetTitleSize( 0.05, "X" );
   frameS->SetTitleSize( 0.05, "Y" );

   // and plot
   frameS->Draw();
   frameB->Draw( "same" );  

   // Draw legend               
   TLegend *legend = new TLegend( 1 - c->GetRightMargin() - 0.32, 1 - c->GetTopMargin() - 0.12, 
                                  1 - c->GetRightMargin(), 1 - c->GetTopMargin() );
   legend->SetFillStyle( 1 );
   legend->AddEntry(frameS,"Signal","p");
   legend->AddEntry(frameB,"Background","p");
   legend->Draw("same");
   legend->SetBorderSize(1);
   legend->SetMargin( 0.3 );

}

TMatrixD* produceSqrtMat( const TMatrixD& covMat )
{
   Double_t sum = 0;
   Int_t size = covMat.GetNrows();;
   TMatrixD* sqrtMat = new TMatrixD( size, size );

   for (Int_t i=0; i< size; i++) {
      
      sum = 0;
      for (Int_t j=0;j< i; j++) sum += (*sqrtMat)(i,j) * (*sqrtMat)(i,j);

      (*sqrtMat)(i,i) = TMath::Sqrt(TMath::Abs(covMat(i,i) - sum));

      for (Int_t k=i+1 ;k<size; k++) {

         sum = 0;
         for (Int_t l=0; l<i; l++) sum += (*sqrtMat)(k,l) * (*sqrtMat)(i,l);

         (*sqrtMat)(k,i) = (covMat(k,i) - sum) / (*sqrtMat)(i,i);

      }
   }
   return sqrtMat;
}

void getGaussRnd( TArrayD& v, const TMatrixD& sqrtMat, TRandom& R ) 
{
   // generate "size" correlated Gaussian random numbers

   // sanity check
   const Int_t size = sqrtMat.GetNrows();
   if (size != v.GetSize()) 
      cout << "<getGaussRnd> too short input vector: " << size << " " << v.GetSize() << endl;

   Double_t* tmpVec = new Double_t[size];

   for (Int_t i=0; i<size; i++) {
      Double_t x, y, z;
      y = R.Rndm();
      z = R.Rndm();
      x = 2*TMath::Pi()*z;
      tmpVec[i] = TMath::Sin(x) * TMath::Sqrt(-2.0*TMath::Log(y));
   }

   for (Int_t i=0; i<size; i++) {
      v[i] = 0;
      for (Int_t j=0; j<=i; j++) v[i] += sqrtMat(i,j) * tmpVec[j];
   }

   delete[] tmpVec;
}

// create the data
void create_lin_Nvar_withFriend(Int_t N = 2000)
{
   const Int_t nvar  = 4;
   const Int_t nvar2 = 1;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar-nvar2; ivar++) {
     cout << "Creating branch var" << ivar+1 << " in signal tree" << endl;
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
   TTree* treeSF = new TTree( "TreeSF", "TreeS", 1 );   
   TTree* treeBF = new TTree( "TreeBF", "TreeB", 1 );   
   for (Int_t ivar=nvar-nvar2; ivar<nvar; ivar++) {
      treeSF->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeBF->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }

      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }

   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree  = (itype==0) ? treeS : treeB;
      TTree* treeF = (itype==0) ? treeSF : treeBF;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         
         tree->Fill();
         treeF->Fill();
      }
   }

//    treeS->AddFriend(treeSF);
//    treeB->AddFriend(treeBF);

   // write trees
   treeS->Write();
   treeB->Write();
   treeSF->Write();
   treeBF->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;


}


// create the tree
TTree* makeTree_lin_Nvar( TString treeName, TString treeTitle, Float_t* x, Float_t* dx, const Int_t nvar, Int_t N )
{
   Float_t xvar[nvar];

   // create tree
   TTree* tree = new TTree(treeName, treeTitle, 1);

   for (Int_t ivar=0; ivar<nvar; ivar++) {
      tree->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
      
   TRandom R( 100 );
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMat = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMat)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMat)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMat)(jvar,ivar) = (*covMat)(ivar,jvar);
      }
   }
   //cout << "covariance matrix: " << endl;
   //covMat->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMat = produceSqrtMat( *covMat );

   // event loop
   for (Int_t i=0; i<N; i++) {

      if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
      getGaussRnd( *v, *sqrtMat, R );

      for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         
      tree->Fill();
   }

   // write trees
//   tree->Write();

   tree->Show(0);

   cout << "created tree: " << tree->GetName() << endl;
   return tree;
}


// create the data
TTree* makeTree_circ(TString treeName, TString treeTitle, Int_t nvar = 2, Int_t N  = 6000, Float_t radius = 1.0, Bool_t distort = false)
{
   Int_t Nn = 0;
   Float_t xvar[nvar]; //variable array size does not work in interactive mode
 
   // create signal and background trees
   TTree* tree = new TTree( treeName, treeTitle, 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      tree->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
      
   TRandom R( 100 );
   //Float_t phimin = -30, phimax = 130;
   Float_t phimin = -70, phimax = 130;
   Float_t phisig = 5;
   Float_t rsig = 0.1;
   Float_t fnmin = -(radius+4.0*rsig);
   Float_t fnmax = +(radius+4.0*rsig);
   Float_t dfn = fnmax-fnmin;

   // event loop
   for (Int_t i=0; i<N; i++) {
      Double_t r1=R.Rndm(),r2=R.Rndm(), r3; 
      r3= r1>r2? r1 :r2;
      Float_t phi;
      if (distort) phi = r3*(phimax - phimin) + phimin;
      else  phi = R.Rndm()*(phimax - phimin) + phimin;
      phi += R.Gaus()*phisig;
      
      Float_t r = radius;
      r += R.Gaus()*rsig;

      xvar[0] = r*cos(TMath::DegToRad()*phi);
      xvar[1] = r*sin(TMath::DegToRad()*phi);

      for( Int_t j = 2; j<nvar; ++j )
	 xvar[j] = dfn*R.Rndm()+fnmin;
         
      tree->Fill();
   }

   for (Int_t i=0; i<Nn; i++) {

      xvar[0] = dfn*R.Rndm()+fnmin;
      xvar[1] = dfn*R.Rndm()+fnmin;

      for( Int_t j = 2; j<nvar; ++j )
	 xvar[j] = dfn*R.Rndm()+fnmin;
         
         
      tree->Fill();
   }

   tree->Show(0);
   // write trees
   cout << "created tree: " << tree->GetName() << endl;
   return tree;
}



// create the data
void create_lin_Nvar_2(Int_t N = 50000)
{
   const int nvar = 4;
   
   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );


   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };

   // create signal and background trees
   TTree* treeS = makeTree_lin_Nvar( "TreeS", "Signal tree", xS, dx, nvar, N );
   TTree* treeB = makeTree_lin_Nvar( "TreeB", "Background tree", xB, dx, nvar, N );

   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(0);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;
}

	


// create the data
void create_lin_Nvar(Int_t N = 50000)
{
   const Int_t nvar = 4;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;
}

// create the category data
// type = 1 (offset) or 2 (variable = -99)
void create_lin_Nvar_categories(Int_t N = 10000, Int_t type = 2)  
{
   const Int_t nvar = 4;
   Float_t xvar[nvar];
   Float_t eta;

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }

   // add category variable
   treeS->Branch( "eta", &eta, "eta/F" );
   treeB->Branch( "eta", &eta, "eta/F" );
      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.0;
   rho[1*3] = 0.0;
   rho[1*4] = 0.0;
   rho[2*3] = 0.0;
   rho[2*4] = 0.0;
   rho[3*4] = 0.0;
   if (type != 1) {
      rho[1*2] = 0.6;
      rho[1*3] = 0.7;
      rho[1*4] = 0.9;
      rho[2*3] = 0.8;
      rho[2*4] = 0.9;
      rho[3*4] = 0.93;
   }

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );

         eta = 2.5*2*(R.Rndm() - 0.5);
         Float_t offset = 0;
         if (type == 1) offset = TMath::Abs(eta) > 1.3 ? 0.8 : -0.8;
         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar] + offset;
         if (type != 1 && TMath::Abs(eta) > 1.3) xvar[nvar-1] = -5;

         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;
}


// create the data
void create_lin_Nvar_weighted(Int_t N = 10000, int WeightedSignal=0, int WeightedBkg=1, Float_t BackgroundContamination=0, Int_t seed=100)
{
   const Int_t nvar = 4;
   Float_t xvar[nvar];
   Float_t weight;

   
   cout << endl << endl << endl;
   cout << "please use .L createData.C++ if you want to run this MC geneation" <<endl;
   cout << "otherwise you will wait for ages!!! " << endl;
   cout << endl << endl << endl;


   // output flie
   TString fileName;
   if (BackgroundContamination) fileName = Form("linCorGauss%d_weighted+background.root",seed);
   else                         fileName = Form("linCorGauss%d_weighted.root",seed);
   
   TFile* dataFile = TFile::Open( fileName.Data(), "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
   if (WeightedSignal||BackgroundContamination>0||1) treeS->Branch( "weight", &weight,"weight/F" );
   if (WeightedBkg)    treeB->Branch( "weight", &weight,"weight/F" );
      
   TRandom R( seed );
   Float_t xS[nvar] = {  0.2,  0.3,  0.4,  0.8 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.4, -0.5 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      Int_t i=0;
      do {
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         //         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = R.Uniform()*10.-5.;
         
         //         weight = 0.5 / (TMath::Gaus( (xvar[nvar-1]-x[nvar-1]), 0, 1.1) );
         // weight = TMath::Gaus(0.675,0,1) / (TMath::Gaus( (xvar[nvar-1]-x[nvar-1]), 0, 1.) );
         weight = 0.8 / (TMath::Gaus( ((*v)[nvar-1]), 0, 1.09) );
         Double_t tmp=R.Uniform()/0.00034;
         if (itype==0 && !WeightedSignal) {
            weight = 1;
            tree->Fill();
            i++;
         } else if (itype==1 && !WeightedBkg) {
            weight = 1;
            tree->Fill();
            i++;
         }
         else {
            if (tmp < weight){
               weight = 1./weight;
               tree->Fill();
               if (i%10 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
               i++;
            }
         }
      } while (i<N);
   }


   if (BackgroundContamination > 0){  // add "background contamination" in the Signal (which later is again "subtracted" with 
            // using (statistically indepentent) background events with negative weight)
      Float_t*  x=xB;
      TMatrixD* m = sqrtMatB;
      TTree* tree = treeS;
      for (Int_t i=0; i<N*BackgroundContamination*2; i++) {
         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );
         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];

         // add weights
         if (i%2) weight = 1;
         else weight = -1;
         
         tree->Fill();
      }
   }



   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   TH1F *h[4];   
   TH1F *hw[4];
   for (Int_t  i=0;i<4;i++){
      char buffer[5];
      sprintf(buffer,"h%d",i);
      h[i]= new TH1F(buffer,"",100,-5,5);
      sprintf(buffer,"hw%d",i);
      hw[i] = new TH1F(buffer,"",100,-5,5);
      hw[i]->SetLineColor(3);
   }

   for (int ie=0;ie<treeS->GetEntries();ie++){
      treeS->GetEntry(ie);
      for (Int_t  i=0;i<4;i++){
         h[i]->Fill(xvar[i]);
         hw[i]->Fill(xvar[i],weight);
      }
   }

   TCanvas *c = new TCanvas("c","",800,800);
   c->Divide(2,2);

   for (Int_t  i=0;i<4;i++){
      c->cd(i+1);
      h[i]->Draw();
      hw[i]->Draw("same");
   }


   //   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;
}



// create the data
void create_lin_Nvar_Arr(Int_t N = 1000)
{
   const Int_t nvar = 4;
   std::vector<float>* xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      xvar[ivar] = new std::vector<float>();
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), "vector<float>", &xvar[ivar], 64000, 1 );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), "vector<float>", &xvar[ivar], 64000, 1 );
   }

   TRandom R( 100 );
   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%100 == 0) cout << "... event: " << i << " (" << N << ")" << endl;

         Int_t aSize = (Int_t)(gRandom->Rndm()*10); // size of array varies between events
         for (Int_t ivar=0; ivar<nvar; ivar++) {
            xvar[ivar]->clear();
            xvar[ivar]->reserve(aSize);
         }
         for(Int_t iA = 0; iA<aSize; iA++) {
            //for (Int_t ivar=0; ivar<nvar; ivar++) {
               getGaussRnd( *v, *m, R );
               for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar]->push_back((*v)[ivar] + x[ivar]);
               //}
         }
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   //plot();
}



// create the data
void create_lin_Nvar_double()
{
   Int_t N = 10000;
   const Int_t nvar = 4;
   Double_t xvar[nvar];
   Double_t xvarD[nvar];
   Float_t  xvarF[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      if (ivar<2) {
         treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvarD[ivar], TString(Form( "var%i/D", ivar+1 )).Data() );
         treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvarD[ivar], TString(Form( "var%i/D", ivar+1 )).Data() );
      }
      else {
         treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvarF[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
         treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvarF[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      }
   }
      
   TRandom R( 100 );
   Double_t xS[nvar] = {  0.2,  0.3,  0.5,  0.6 };
   Double_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Double_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Double_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Double_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         for (Int_t ivar=0; ivar<nvar; ivar++) {
            if (ivar<2) xvarD[ivar] = xvar[ivar];
            else        xvarF[ivar] = xvar[ivar];
         }
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();
}

// create the data
void create_lin_Nvar_discrete()
{
   Int_t N = 10000;
   const Int_t nvar = 4;
   Float_t xvar[nvar];
   Int_t   xvarI[2];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar-2; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
   for (Int_t ivar=0; ivar<2; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar+nvar-2+1 )).Data(), &xvarI[ivar], TString(Form( "var%i/I", ivar+nvar-2+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+nvar-2+1 )).Data(), &xvarI[ivar], TString(Form( "var%i/I", ivar+nvar-2+1 )).Data() );
   }
      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.2,  0.3,  1,  2 };
   Float_t xB[nvar] = { -0.2, -0.3,  0,  0 };
   Float_t dx[nvar] = {  1.0,  1.0, 1, 2 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;
   // no correlations
   for (int i=0; i<20; i++) rho[i] = 0;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];

         xvarI[0] =  TMath::Nint(xvar[nvar-2]);
         xvarI[1] =  TMath::Nint(xvar[nvar-1]);
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();
}

// create the data
void create_ManyVars()
{
   Int_t N = 20000;
   const Int_t nvar = 20;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }
      
   Float_t xS[nvar];
   Float_t xB[nvar];
   Float_t dx[nvar];
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      xS[ivar] = 0 + ivar*0.05;
      xB[ivar] = 0 - ivar*0.05;
      dx[ivar] = 1;
   }

   xS[0] =   0.2;
   xB[0] =  -0.2;
   dx[0] =   1.0;
   xS[1] =   0.3;
   xB[1] =  -0.3;
   dx[1] =   1.0;
   xS[2] =   0.4;
   xB[2] =  -0.4;
   dx[2] =  1.0 ;
   xS[3] =   0.8 ;
   xB[3] =  -0.5 ;
   dx[3] =   1.0 ;
//   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   TRandom R( 100 );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t* x = (itype == 0) ? xS : xB; 

      // event loop
      TTree* tree = (itype == 0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         for (Int_t ivar=0; ivar<nvar; ivar++) {
            if (ivar == 1500 && itype!=10) xvar[ivar] = 1;
            else                           xvar[ivar] = x[ivar] + R.Gaus()*dx[ivar];
         }
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   plot();
   cout << "created data file: " << dataFile->GetName() << endl;
}

// create the data
void create_lin_NvarObsolete()
{
   Int_t N = 20000;
   const Int_t nvar = 20;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }
      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.5,  0.5,  0.0,  0.0,  0.0,  0.0 };
   Float_t xB[nvar] = { -0.5, -0.5, -0.0, -0.0, -0.0, -0.0 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[50];
   for (Int_t i=0; i<50; i++) rho[i] = 0;
   rho[1*2] = 0.3;
   rho[1*3] = 0.0;
   rho[1*4] = 0.0;
   rho[2*3] = 0.0;
   rho[2*4] = 0.0;
   rho[3*4] = 0.0;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();
}

// create the data
void create_lin(Int_t N = 2000)
{
   const Int_t nvar = 2;
   Double_t xvar[nvar];
   Float_t weight;

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/D", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/D", ivar )).Data() );
   }
   treeS->Branch( "weight", &weight, "weight/F" );
   treeB->Branch( "weight", &weight, "weight/F" );
      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.0,  0.0 };
   Float_t xB[nvar] = { -0.0, -0.0 };
   Float_t dx[nvar] = {  1.0,  1.0 };
   TArrayD* v = new TArrayD( 2 );
   Float_t rhoS =  0.21;
   Float_t rhoB =  0.0;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rhoS*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rhoB*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );
         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];

         // add weights
         if (itype == 0) weight = 1.0; // this is the signal weight
         else            weight = 2.0; // this is the background weight
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();
}

void create_fullcirc(Int_t nmax  = 20000,  Bool_t distort=false)
{
  TFile* dataFile = TFile::Open( "circledata.root", "RECREATE" );
   int nvar = 2;
   int nsig = 0, nbgd=0;
   Float_t weight=1;
   Float_t xvar[100];
   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar)).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar)).Data() );
   }
   treeS->Branch("weight", &weight, "weight/F");
   treeB->Branch("weight", &weight, "weight/F");

   TRandom R( 100 );
   do {
      for (Int_t ivar=0; ivar<nvar; ivar++) { xvar[ivar]=2.*R.Rndm()-1.;}
      Float_t xout = xvar[0]*xvar[0]+xvar[1]*xvar[1];
      if (nsig<10) cout << "xout = " << xout<<endl;
      if (xout < 0.3  || (xout >0.3 && xout<0.5 && R.Rndm() > xout)) {
         if (distort && xvar[0] < 0 && R.Rndm()>0.1) continue; 
         treeS->Fill();
         nsig++;
      }
      else {
         if (distort && xvar[0] > 0 && R.Rndm()>0.1) continue; 
         treeB->Fill();
         nbgd++;
      }
   } while ( nsig < nmax || nbgd < nmax);

   dataFile->Write();
   dataFile->Close();
   
} 

// create the data
void create_circ(Int_t N  = 6000, Bool_t distort = false)
{
   Int_t Nn = 0;
   const Int_t nvar = 2;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }
//    TTree *treeB  = treeS->CloneTree();
//    for (Int_t ivar=0; ivar<nvar; ivar++) {
//       treeS->SetBranchAddress( Form( "var%i", ivar ), &xvar[ivar] );
//       treeB->SetBranchAddress( Form( "var%i", ivar ), &xvar[ivar] );
//    }
//    treeB->SetName ( "TreeB" );
//    treeB->SetTitle( "TreeB" );
      
   TRandom R( 100 );
   //Float_t phimin = -30, phimax = 130;
   Float_t phimin = -70, phimax = 130;
   Float_t phisig = 5;
   Float_t rS = 1.1;
   Float_t rB = 0.75;
   Float_t rsig = 0.1;
   Float_t fnmin = -(rS+4.0*rsig);
   Float_t fnmax = +(rS+4.0*rsig);
   Float_t dfn = fnmax-fnmin;
   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {
	 Double_t r1=R.Rndm(),r2=R.Rndm(), r3; 
	 if (itype==0) r3= r1>r2? r1 :r2;
	 else r3= r2;
	 Float_t phi;
	 if (distort) phi = r3*(phimax - phimin) + phimin;
	 else  phi = R.Rndm()*(phimax - phimin) + phimin;
         phi += R.Gaus()*phisig;
      
         Float_t r = (itype==0) ? rS : rB;
         r += R.Gaus()*rsig;

         xvar[0] = r*cos(TMath::DegToRad()*phi);
         xvar[1] = r*sin(TMath::DegToRad()*phi);
         
         tree->Fill();
      }

      for (Int_t i=0; i<Nn; i++) {

         xvar[0] = dfn*R.Rndm()+fnmin;
         xvar[1] = dfn*R.Rndm()+fnmin;
         
         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();
}


void create_schachbrett(Int_t nEvents = 20000) {

   const Int_t nvar = 2;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }

   Int_t   nSeed   = 12345;
   TRandom *m_rand = new TRandom(nSeed);
   Double_t sigma=0.3;
   Double_t meanX;
   Double_t meanY;
   Int_t xtype=1, ytype=1;
   Int_t iev=0;
   Int_t m_nDim = 2; // actually the boundary, there is a "bump" for every interger value
                     // between in the Inteval [-m_nDim,m_nDim]
   while (iev < nEvents){
      xtype=1;
      for (Int_t i=-m_nDim; i <=  m_nDim; i++){
         ytype  =  1;
         for (Int_t j=-m_nDim; j <=  m_nDim; j++){
            meanX=Double_t(i);
            meanY=Double_t(j);
            xvar[0]=m_rand->Gaus(meanY,sigma);
            xvar[1]=m_rand->Gaus(meanX,sigma);
            Int_t type   = xtype*ytype;
            TTree* tree = (type==1) ? treeS : treeB;
            tree->Fill();
            iev++;
            ytype *= -1;
         }
         xtype *= -1;
      }
   }


   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();

}


void create_schachbrett_5D(Int_t nEvents = 200000) {
   const Int_t nvar = 5;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }

   Int_t   nSeed   = 12345;
   TRandom *m_rand = new TRandom(nSeed);
   Double_t sigma=0.3;
   Int_t itype[nvar];
   Int_t iev=0;
   Int_t m_nDim = 2; // actually the boundary, there is a "bump" for every interger value
                     // between in the Inteval [-m_nDim,m_nDim]

   int idx[nvar];
   while (iev < nEvents){
      itype[0]=1;
      for (idx[0]=-m_nDim; idx[0] <=  m_nDim; idx[0]++){
         itype[1]=1;
         for (idx[1]=-m_nDim; idx[1] <=  m_nDim; idx[1]++){
            itype[2]=1;
            for (idx[2]=-m_nDim; idx[2] <=  m_nDim; idx[2]++){
               itype[3]=1;
               for (idx[3]=-m_nDim; idx[3] <=  m_nDim; idx[3]++){
                  itype[4]=1;
                  for (idx[4]=-m_nDim; idx[4] <=  m_nDim; idx[4]++){
                     Int_t type   = itype[0]; 
                     for (Int_t i=0;i<nvar;i++){
                        xvar[i]=m_rand->Gaus(Double_t(idx[i]),sigma);
                        if (i>0) type *= itype[i];
                     }
                     TTree* tree = (type==1) ? treeS : treeB;
                     tree->Fill();
                     iev++;
                     itype[4] *= -1;
                  }
                  itype[3] *= -1;
               }
               itype[2] *= -1;
            }
            itype[1] *= -1;
         }
         itype[0] *= -1;
      }
   }
            
   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();

}


void create_schachbrett_4D(Int_t nEvents = 200000) {

   const Int_t nvar = 4;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }

   Int_t   nSeed   = 12345;
   TRandom *m_rand = new TRandom(nSeed);
   Double_t sigma=0.3;
   Int_t itype[nvar];
   Int_t iev=0;
   Int_t m_nDim = 2; // actually the boundary, there is a "bump" for every interger value
                     // between in the Inteval [-m_nDim,m_nDim]

   int idx[nvar];
   while (iev < nEvents){
      itype[0]=1;
      for (idx[0]=-m_nDim; idx[0] <=  m_nDim; idx[0]++){
         itype[1]=1;
         for (idx[1]=-m_nDim; idx[1] <=  m_nDim; idx[1]++){
            itype[2]=1;
            for (idx[2]=-m_nDim; idx[2] <=  m_nDim; idx[2]++){
               itype[3]=1;
               for (idx[3]=-m_nDim; idx[3] <=  m_nDim; idx[3]++){
                  Int_t type   = itype[0]; 
                  for (Int_t i=0;i<nvar;i++){
                     xvar[i]=m_rand->Gaus(Double_t(idx[i]),sigma);
                     if (i>0) type *= itype[i];
                  }
                  TTree* tree = (type==1) ? treeS : treeB;
                  tree->Fill();
                  iev++;
                  itype[3] *= -1;
               }
               itype[2] *= -1;
            }
            itype[1] *= -1;
         }
         itype[0] *= -1;
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();

}


void create_schachbrett_3D(Int_t nEvents = 20000) {

   const Int_t nvar = 3;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }

   Int_t   nSeed   = 12345;
   TRandom *m_rand = new TRandom(nSeed);
   Double_t sigma=0.3;
   Int_t itype[nvar];
   Int_t iev=0;
   Int_t m_nDim = 2; // actually the boundary, there is a "bump" for every interger value
                     // between in the Inteval [-m_nDim,m_nDim]

   int idx[nvar];
   while (iev < nEvents){
      itype[0]=1;
      for (idx[0]=-m_nDim; idx[0] <=  m_nDim; idx[0]++){
         itype[1]=1;
         for (idx[1]=-m_nDim; idx[1] <=  m_nDim; idx[1]++){
            itype[2]=1;
            for (idx[2]=-m_nDim; idx[2] <=  m_nDim; idx[2]++){
               Int_t type   = itype[0]; 
               for (Int_t i=0;i<nvar;i++){
                  xvar[i]=m_rand->Gaus(Double_t(idx[i]),sigma);
                  if (i>0) type *= itype[i];
               }
               TTree* tree = (type==1) ? treeS : treeB;
               tree->Fill();
               iev++;
               itype[2] *= -1;
            }
            itype[1] *= -1;
         }
         itype[0] *= -1;
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();

}


void create_schachbrett_2D(Int_t nEvents = 100000, Int_t nbumps=2) {

   const Int_t nvar = 2;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }

   Int_t   nSeed   = 345;
   TRandom *m_rand = new TRandom(nSeed);
   Double_t sigma=0.35;
   Int_t itype[nvar];
   Int_t iev=0;
   Int_t m_nDim = nbumps; // actually the boundary, there is a "bump" for every interger value
                     // between in the Inteval [-m_nDim,m_nDim]

   int idx[nvar];
   while (iev < nEvents){
      itype[0]=1;
      for (idx[0]=-m_nDim; idx[0] <=  m_nDim; idx[0]++){
         itype[1]=1;
         for (idx[1]=-m_nDim; idx[1] <=  m_nDim; idx[1]++){
            Int_t type   = itype[0]; 
            for (Int_t i=0;i<nvar;i++){
               xvar[i]=m_rand->Gaus(Double_t(idx[i]),sigma);
               if (i>0) type *= itype[i];
            }
            TTree* tree = (type==1) ? treeS : treeB;
            tree->Fill();
            iev++;
            itype[1] *= -1;
         }
         itype[0] *= -1;
      }
   }
   
   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot();

}



void create_3Bumps(Int_t nEvents = 5000) {
   // signal is clustered around (1,0) and (-1,0) where one is two times(1,0) 
   // bkg                        (0,0)
   


   const Int_t nvar = 2;
   Float_t xvar[nvar];

   // output flie
   TString filename = "data_3Bumps.root";
   TFile* dataFile = TFile::Open( filename, "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar )).Data() );
   }

   Int_t   nSeed   = 12345;
   TRandom *m_rand = new TRandom(nSeed);
   Double_t sigma=0.2;
   Int_t type;
   Int_t iev=0;
   Double_t Centers[nvar][6] = {{-1,0,0,0,1,1},{0,0,0,0,0,0}}; // 


   while (iev < nEvents){
      for (int idx=0; idx<6; idx++){
         if (idx==1 || idx==2 || idx==3) type = 0;
         else type=1;
         for (Int_t ivar=0;ivar<nvar;ivar++){
            xvar[ivar]=m_rand->Gaus(Centers[ivar][idx],sigma);
         }
         TTree* tree = (type==1) ? treeS : treeB;
         tree->Fill();
         iev++;
      }
   }
   
   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;

   plot(filename);

}

void createOnionData(Int_t nmax = 50000){
   // output file
   TFile* dataFile = TFile::Open( "oniondata.root", "RECREATE" );
   int nvar = 4;
   int nsig = 0, nbgd=0;
   Float_t xvar[100];
   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeS->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
      treeB->Branch( TString(Form( "var%i", ivar+1 )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar+1 )).Data() );
   }
   
   TRandom R( 100 );
   do {
      for (Int_t ivar=0; ivar<nvar; ivar++) { xvar[ivar]=R.Rndm();}
      Float_t xout = sin(2.*acos(-1.)*(xvar[0]*xvar[1]*xvar[2]*xvar[3]+xvar[0]*xvar[1]));
      if (nsig<100) cout << "xout = " << xout<<endl;
      Int_t i = (Int_t) ((1.+xout)*4.99);
      if (i%2 == 0 && nsig < nmax) {
	 treeS->Fill();
	 nsig++;
      }
      if (i%2 != 0 && nbgd < nmax){
	 treeB->Fill();
	 nbgd++;
      }
   } while ( nsig < nmax || nbgd < nmax);

   dataFile->Write();
   dataFile->Close();
}

void create_multiclassdata(Int_t nmax  = 20000)
{
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );
   int ncls = 3;
   int nvar = 4;
   int ndat = 0;
   Int_t cls;
   Float_t thecls;
   Float_t weight=1;
   Float_t xcls[100];
   Float_t xmean[3][4] = {
      { 0.   ,  0.3,  0.5, 0.9 }, 
      { -0.2 , -0.3,  0.5, 0.4 }, 
      { 0.2  ,  0.1, -0.1, 0.7 }} ;

   Float_t xvar[100];
   // create tree using class flag stored in int variable cls
   TTree* treeR = new TTree( "TreeR", "TreeR", 1 );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      treeR->Branch( TString(Form( "var%i", ivar )).Data(), &xvar[ivar], TString(Form( "var%i/F", ivar)).Data() );
   }
   for (Int_t icls=0; icls<ncls; icls++) {
      treeR->Branch(TString(Form( "cls%i", icls )).Data(), &xcls[icls], TString(Form( "cls%i/F", icls)).Data() );
   }

   treeR->Branch("cls", &thecls, "cls/F");
   treeR->Branch("weight", &weight, "weight/F");
   
   TRandom R( 100 );
   do {
      for (Int_t icls=0; icls<ncls; icls++) xcls[icls]=0.;
      cls = R.Integer(ncls);
      thecls = cls;
      xcls[cls]=1.;
      for (Int_t ivar=0; ivar<nvar; ivar++) { 
         xvar[ivar]=R.Gaus(xmean[cls][ivar],1.);
      }
      
      if (ndat<30) cout << "cls=" << cls <<" xvar = " << xvar[0]<<" " <<xvar[1]<<" " << xvar[2]<<" " <<xvar[3]<<endl;
      
      treeR->Fill();
      ndat++;
   } while ( ndat < nmax );

   dataFile->Write();
   dataFile->Close();
   
} 






// create the data
void create_array_with_different_lengths(Int_t N = 100)
{
   const Int_t nvar = 4;
   Int_t nvarCurrent = 4;
   Float_t xvar[nvar];

   // output flie
   TFile* dataFile = TFile::Open( "data.root", "RECREATE" );

   // create signal and background trees
   TTree* treeS = new TTree( "TreeS", "TreeS", 1 );   
   TTree* treeB = new TTree( "TreeB", "TreeB", 1 );   
   treeS->Branch( "arrSize", &nvarCurrent, "arrSize/I" );
   treeS->Branch( "arr", xvar, "arr[arrSize]/F" );
   treeB->Branch( "arrSize", &nvarCurrent, "arrSize/I" );
   treeB->Branch( "arr", xvar, "arr[arrSize]/F" );
      
   TRandom R( 100 );
   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t dx[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   TArrayD* v = new TArrayD( nvar );
   Float_t rho[20];
   rho[1*2] = 0.4;
   rho[1*3] = 0.6;
   rho[1*4] = 0.9;
   rho[2*3] = 0.7;
   rho[2*4] = 0.8;
   rho[3*4] = 0.93;

   // create covariance matrix
   TMatrixD* covMatS = new TMatrixD( nvar, nvar );
   TMatrixD* covMatB = new TMatrixD( nvar, nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      (*covMatS)(ivar,ivar) = dx[ivar]*dx[ivar];
      (*covMatB)(ivar,ivar) = dx[ivar]*dx[ivar];
      for (Int_t jvar=ivar+1; jvar<nvar; jvar++) {
         (*covMatS)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatS)(jvar,ivar) = (*covMatS)(ivar,jvar);

         (*covMatB)(ivar,jvar) = rho[(ivar+1)*(jvar+1)]*dx[ivar]*dx[jvar];
         (*covMatB)(jvar,ivar) = (*covMatB)(ivar,jvar);
      }
   }
   cout << "signal covariance matrix: " << endl;
   covMatS->Print();
   cout << "background covariance matrix: " << endl;
   covMatB->Print();

   // produce the square-root matrix
   TMatrixD* sqrtMatS = produceSqrtMat( *covMatS );
   TMatrixD* sqrtMatB = produceSqrtMat( *covMatB );

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      Float_t*  x;
      TMatrixD* m;
      if (itype == 0) { x = xS; m = sqrtMatS; cout << "- produce signal" << endl; }
      else            { x = xB; m = sqrtMatB; cout << "- produce background" << endl; }

      // event loop
      TTree* tree = (itype==0) ? treeS : treeB;
      for (Int_t i=0; i<N; i++) {

         if (i%1000 == 0) cout << "... event: " << i << " (" << N << ")" << endl;
         getGaussRnd( *v, *m, R );

         for (Int_t ivar=0; ivar<nvar; ivar++) xvar[ivar] = (*v)[ivar] + x[ivar];
         

	 nvarCurrent = (i%4)+1;

         tree->Fill();
      }
   }

   // write trees
   treeS->Write();
   treeB->Write();

   treeS->Show(0);
   treeB->Show(1);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;
}



// create the data
void create_MultipleBackground(Int_t N = 50000)
{
   const int nvar = 4;
   
   // output flie
   TFile* dataFile = TFile::Open( "tmva_example_multiple_background.root", "RECREATE" );


   Float_t xS[nvar] = {  0.2,  0.3,  0.5,  0.9 };
   Float_t xB0[nvar] = { -0.2, -0.3, -0.5, -0.6 };
   Float_t xB1[nvar] = { -0.2, 0.3, 0.5, -0.6 };
   Float_t dx0[nvar] = {  1.0,  1.0, 1.0, 1.0 };
   Float_t dx1[nvar] = {  -1.0,  -1.0, -1.0, -1.0 };

   // create signal and background trees
   TTree* treeS = makeTree_lin_Nvar( "TreeS", "Signal tree", xS, dx0, nvar, N );
   TTree* treeB0 = makeTree_lin_Nvar( "TreeB0", "Background 0", xB0, dx0, nvar, N );
   TTree* treeB1 = makeTree_lin_Nvar( "TreeB1", "Background 1", xB1, dx1, nvar, N );
   TTree* treeB2 = makeTree_circ( "TreeB2", "Background 2", nvar, N, 1.5, true);

   treeS->Write();
   treeB0->Write();
   treeB1->Write();
   treeB2->Write();

   //treeS->Show(0);
   //treeB0->Show(0);
   //treeB1->Show(0);
   //treeB2->Show(0);

   dataFile->Close();
   cout << "created data file: " << dataFile->GetName() << endl;
}
