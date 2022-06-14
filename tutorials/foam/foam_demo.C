/// \file
/// \ingroup tutorial_FOAM
/// \notebook -nodraw
/// Demonstrate the TFoam class.
///
///  To run this macro type from CINT command line
///
/// ~~~{.cpp}
///  root [0] gSystem->Load("libFoam.so")
///  root [1] .x foam_demo.C+
/// ~~~
///
/// \macro_code
///
/// \author Stascek Jadach


#include "Riostream.h"
#include "TFile.h"
#include "TFoam.h"
#include "TH1.h"
#include "TMath.h"
#include "TFoamIntegrand.h"
#include "TRandom3.h"

class TFDISTR: public TFoamIntegrand {
public:
   TFDISTR(){};
   Double_t Density(int nDim, Double_t *Xarg){
   // Integrand for mFOAM
   Double_t Fun1,Fun2,R1,R2;
   Double_t pos1=1e0/3e0;
   Double_t pos2=2e0/3e0;
   Double_t Gam1= 0.100e0;  // as in JPC
   Double_t Gam2= 0.100e0;  // as in JPC
   Double_t sPi = sqrt(TMath::Pi());
   Double_t xn1=1e0;
   Double_t xn2=1e0;
   int i;
   R1=0;
   R2=0;
   for(i = 0 ; i<nDim ; i++){
      R1=R1+(Xarg[i] -pos1)*(Xarg[i] -pos1);
      R2=R2+(Xarg[i] -pos2)*(Xarg[i] -pos2);
      xn1=xn1*Gam1*sPi;
      xn2=xn2*Gam2*sPi;
   }
   R1   = sqrt(R1);
   R2   = sqrt(R2);
   Fun1 = exp(-(R1*R1)/(Gam1*Gam1))/xn1;  // Gaussian delta-like profile
   Fun2 = exp(-(R2*R2)/(Gam2*Gam2))/xn2;  // Gaussian delta-like profile
   return 0.5e0*(Fun1+ Fun2);
}
  ClassDef(TFDISTR,1) //Class of testing functions for FOAM
};
ClassImp(TFDISTR)

Int_t foam_demo()
{
   TFile RootFile("foam_demo.root","RECREATE","histograms");
   long   loop;
   Double_t MCresult,MCerror,MCwt;
   //-----------------------------------------
   long NevTot   =     50000;   // Total MC statistics
   Int_t  kDim   =         2;   // total dimension
   Int_t  nCells   =     500;   // Number of Cells
   Int_t  nSampl   =     200;   // Number of MC events per cell in build-up
   Int_t  nBin     =       8;   // Number of bins in build-up
   Int_t  OptRej   =       1;   // Wted events for OptRej=0; wt=1 for OptRej=1 (default)
   Int_t  OptDrive =       2;   // (D=2) Option, type of Drive =0,1,2 for TrueVol,Sigma,WtMax
   Int_t  EvPerBin =      25;   // Maximum events (equiv.) per bin in buid-up
   Int_t  Chat     =       1;   // Chat level
   //-----------------------------------------
   TRandom *PseRan   = new TRandom3();  // Create random number generator
   TFoam   *FoamX    = new TFoam("FoamX");   // Create Simulator
   TFoamIntegrand    *rho= new TFDISTR();
   PseRan->SetSeed(4357);
   //-----------------------------------------
   cout<<"*****   Demonstration Program for Foam version "<<FoamX->GetVersion()<<"    *****"<<endl;
   FoamX->SetkDim(        kDim);      // Mandatory!!!
   FoamX->SetnCells(      nCells);    // optional
   FoamX->SetnSampl(      nSampl);    // optional
   FoamX->SetnBin(        nBin);      // optional
   FoamX->SetOptRej(      OptRej);    // optional
   FoamX->SetOptDrive(    OptDrive);  // optional
   FoamX->SetEvPerBin(    EvPerBin);  // optional
   FoamX->SetChat(        Chat);      // optional
   //-----------------------------------------
   FoamX->SetRho(rho);
   FoamX->SetPseRan(PseRan);
   FoamX->Initialize(); // Initialize simulator
   FoamX->Write("FoamX");     // Writing Foam on the disk, TESTING PERSISTENCY!!!
   //-----------------------------------------
   long nCalls=FoamX->GetnCalls();
   cout << "====== Initialization done, entering MC loop" << endl;
   //-----------------------------------------
   /*cout<<" About to start MC loop: ";  cin.getline(question,20);*/
   Double_t *MCvect =new Double_t[kDim]; // vector generated in the MC run
   //-----------------------------------------
   TH1D  *hst_Wt = new TH1D("hst_Wt" ,  "Main weight of Foam",25,0,1.25);
   hst_Wt->Sumw2();
   //-----------------------------------------
   for(loop=0; loop<NevTot; loop++){
   /*===============================*/
      FoamX->MakeEvent();           // generate MC event
   /*===============================*/
      FoamX->GetMCvect( MCvect);
      MCwt=FoamX->GetMCwt();
      hst_Wt->Fill(MCwt,1.0);
      if(loop<15){
         cout<<"MCwt= "<<MCwt<<",  ";
         cout<<"MCvect= ";
         for ( Int_t k=0 ; k<kDim ; k++) cout<<MCvect[k]<<" "; cout<< endl;
      }
      if( ((loop)%100000)==0 ){
         cout<<"   loop= "<<loop<<endl;
      }
   }

   //-----------------------------------------

   cout << "====== Events generated, entering Finalize" << endl;

   hst_Wt->Print("all");
   Double_t eps = 0.0005;
   Double_t Effic, WtMax, AveWt, Sigma;
   Double_t IntNorm, Errel;
   FoamX->Finalize(   IntNorm, Errel);     // final printout
   FoamX->GetIntegMC( MCresult, MCerror);  // get MC intnegral
   FoamX->GetWtParams(eps, AveWt, WtMax, Sigma); // get MC wt parameters
   Effic=0; if(WtMax>0) Effic=AveWt/WtMax;
   cout << "================================================================" << endl;
   cout << " MCresult= " << MCresult << " +- " << MCerror << " RelErr= "<< MCerror/MCresult << endl;
   cout << " Dispersion/<wt>= " << Sigma/AveWt << endl;
   cout << "      <wt>/WtMax= " << Effic <<",    for epsilon = "<<eps << endl;
   cout << " nCalls (initialization only) =   " << nCalls << endl;
   cout << "================================================================" << endl;

   delete [] MCvect;
   //
   RootFile.ls();
   RootFile.Write();
   RootFile.Close();
   cout << "***** End of Demonstration Program  *****" << endl;

   return 0;
} 


