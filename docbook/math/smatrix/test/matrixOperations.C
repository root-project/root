#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TFile.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TMath.h"
#include "TFrame.h"

#include <string>
#include <iostream>

int fillCol=20;
std::string systemName; 
bool drawSingleGraph = true; 

int topX=10;
int topY=50;

const Int_t N=10;


void matrixOperations_do(std::string type = "", bool clhep=false, bool drawSingleGraph = false ); 

void matrixOperations(std::string type = "",bool clhep=false, bool drawSingleGraph = false ) { 

  matrixOperations_do(type,clhep,drawSingleGraph); 
//   matrixOperations_do("slc3_ia32_gcc323"); 
//   matrixOperations_do("win32_vc71"); 

}


void DrawData(char *  title, TGraphErrors * h1, TGraphErrors * h2, TGraphErrors * h3 = 0, TGraphErrors * h4 = 0, TGraphErrors * h5 = 0, TGraphErrors * h6 = 0) {  

  
   gPad->SetFillColor(fillCol);
   gPad->SetGrid();
   gPad->SetLogy();
   gPad->SetLogx();
   //gStyle->SetFillColor(30);

   TMultiGraph * mg = new TMultiGraph();

h1->SetLineColor(3);
h1->SetMarkerColor(3);
h1->SetMarkerStyle(20);
h1->SetLineWidth(2);
mg->Add(h1);

h2->SetLineColor(4);
h2->SetMarkerColor(4);
h2->SetMarkerStyle(21);
h2->SetLineWidth(2);
 mg->Add(h2);

 if (h3 != 0 ) { 
h3->SetLineColor(2);
h3->SetMarkerColor(2);
h3->SetMarkerStyle(22);
h3->SetLineWidth(2);
 mg->Add(h3);
 }
if (h4!= 0) { 
   h4->SetLineColor(5);
   h4->SetMarkerColor(5);
   h4->SetMarkerStyle(23);
   h4->SetLineWidth(2);
 mg->Add(h4);
 }
if (h5!= 0) { 
   h5->SetLineColor(6);
   h5->SetMarkerColor(6);
   h5->SetMarkerStyle(29);
   h5->SetLineWidth(2);
 mg->Add(h5);
 }
if (h6!= 0) { 
   h6->SetLineColor(7);
   h6->SetMarkerColor(7);
   h6->SetMarkerStyle(3);
   h6->SetLineWidth(2);
 mg->Add(h6);
 }


TGraphErrors * hd = h1; 
// hd->SetLineWidth(2);
//hd->GetXaxis()->SetLabelSize(0);
//hd->GetXaxis()->SetNdivisions(510);
    hd->GetYaxis()->SetTitleSize(0.05);
    hd->GetXaxis()->SetTitle("Matrix size ");
    hd->GetYaxis()->SetTitle("CPU Time ");
    hd->GetYaxis()->SetTitleOffset(0.7);
    hd->SetTitle("");
//    h1->SetMaximum(20);
//   h1->SetMinimum(0.0);

    gStyle->SetOptStat(0);

 
    mg->Draw("alp");

    // ned to do after drawing
    mg->GetXaxis()->SetLimits(1.8,32);
    mg->GetXaxis()->SetTitle("Matrix size ");
    mg->GetXaxis()->SetMoreLogLabels(1);
    mg->GetXaxis()->SetNoExponent(1);
    mg->GetYaxis()->SetTitle("CPU Time ");
    mg->GetYaxis()->SetTitleOffset(1.25);

    TLegend * tleg = new TLegend(0.78, 0.25, 0.97 ,0.45);
    tleg->AddEntry(h1, "SMatrix", "p");
    tleg->AddEntry(h2, "TMatrix", "p");
    if (h3 != 0) tleg->AddEntry(h3, "SMatrix_sym", "p");
    if (h4 != 0) tleg->AddEntry(h4, "TMatrix_sym", "p");
    if (h5 != 0) tleg->AddEntry(h5, "HepMatrix", "p");
    if (h6 != 0) tleg->AddEntry(h6, "HepMatrix_sym", "p");
    tleg->Draw();

    { 
       TPaveText *pt1 = new TPaveText(0.78,0.15,0.97,0.2,"brNDC");
       pt1->AddText(systemName.c_str());
       pt1->SetBorderSize(1);
       pt1->Draw();
       pt1->SetFillColor(0);
    }

    { 
       TPaveText *pt1 = new TPaveText(0.3,0.91,0.7,0.98,"brNDC");
       pt1->AddText(title);
       pt1->SetBorderSize(1);
       pt1->Draw();
       pt1->SetFillColor(0);
    }
    
}

void GetData(std::string s,double * x, double * y, double * ey) {
   std::string fileName;
   if (systemName != "") 
      fileName="testOperations_" + systemName + ".root";
   else 
      fileName="testOperations.root";

   TFile * f = new TFile(fileName.c_str());
   TProfile * h1 = (TProfile * ) f->Get(s.c_str() );
   if (h1 ==0) { 
      std::cout << "Profile " << s << " not found !!! " << std::endl;
      return;
   } 
   for (int i = 0; i < N; ++i) { 
      y[i] = h1->GetBinContent(int(x[i] + 0.1) ); 
      ey[i] = h1->GetBinError(int(x[i] + 0.1) ); 
   }
}



void matrixOperations_do(std::string type, bool clhep, bool drawSingleGraph) { 


   systemName = type;
   std::string cName = "c1_" + type; 
   std::string cTitle = "Matrix operations " + type;

   TCanvas * c1 = new TCanvas(cName.c_str(),cTitle.c_str(),topX,topY,800*sqrt(2),800);
   topX+=20;
   topY+=20;

   c1->Divide(3,2);

   const int nb = N;
   //double x[N] = { 2.,3.,4.,5.,7.,10,15,20,50,75,100};
   double x[N] = { 2.,3.,4.,5.,6,7.,10,15,20,30};
   //  timings
   double smat[N];  //  = { 1., 2., 3.,  4,5,10,100,300,1000 };
   double tmat[N]; // = {  1.4, 2.4, 3.4, 5,6,20,200,500,2000 };  
   double cmat[N]; // = {  2., 3., 4., 5,8,10,300,800,5000 };  
   double ymat[N]; // = {  2., 3., 4., 5,8,10,300,800,5000 };  
   double wmat[N]; // = {  2., 3., 4., 5,8,10,300,800,5000 };  
   double zmat[N]; // = {  2., 3., 4., 5,8,10,300,800,5000 };  
   double es[N];
   double et[N];
   double ec[N];
   double ey[N];
   double ew[N];
   double ez[N];


   c1->cd(1);
   
   GetData("SMatrix_dot",x,smat,es);
   GetData("TMatrix_dot",x,tmat,et);
   if (clhep) GetData("HepMatrix_dot",x,cmat,ec); 
   TGraphErrors * g10 = new TGraphErrors(nb,x, smat,0, es);
   TGraphErrors * g20 = new TGraphErrors(nb,x, tmat,0, et);
   TGraphErrors * g30 = 0; 
   TGraphErrors * g40 = 0; 
   TGraphErrors * g50 = 0; 
   if (clhep)   g50 = new TGraphErrors(nb,x, cmat,0, ec);
   DrawData("#vec{v} #upoint #vec{w}",g10,g20,g30,g40,g50);

   c1->cd(2);
   GetData("SMatrix_M*V+",x,smat,es);
   GetData("TMatrix_M*V+",x,tmat,et);
   GetData("SMatrix_sym_M*V+",x,ymat,ey);
   GetData("TMatrix_sym_M*V+",x,wmat,ew);
   if (clhep) GetData("HepMatrix_M*V+",x,cmat,ec);
   if (clhep) GetData("HepMatrix_sym_M*V+",x,zmat,ez);
   TGraphErrors * g11 = new TGraphErrors(nb,x, smat,0,es);
   TGraphErrors * g21 = new TGraphErrors(nb,x, tmat,0,et);
   TGraphErrors * g31 = new TGraphErrors(nb,x, ymat,0,ey);
   TGraphErrors * g41 = new TGraphErrors(nb,x, wmat,0,ew);
   TGraphErrors * g51 = 0; 
   TGraphErrors * g61 = 0; 
   if (clhep)     g51 = new TGraphErrors(nb,x, cmat,0, ec);
   if (clhep)     g61 = new TGraphErrors(nb,x, zmat,0, ez);
   DrawData("M #upoint #vec{v} + #vec{w}",g11,g21,g31,g41,g51,g61);

   c1->cd(3);
   GetData("SMatrix_prod",x,smat,es);
   GetData("TMatrix_prod",x,tmat,et);
   GetData("SMatrix_sym_prod",x,ymat,ey);
   GetData("TMatrix_sym_prod",x,wmat,ew);
   if (clhep) { 
     GetData("HepMatrix_M*V+",x,cmat,ec);
     GetData("HepMatrix_sym_M*V+",x,zmat,ez);
   }
   TGraphErrors * g12 = new TGraphErrors(nb,x, smat,0,es);
   TGraphErrors * g22 = new TGraphErrors(nb,x, tmat,0,et);
   TGraphErrors * g32 = new TGraphErrors(nb,x, ymat,0,ey);
   TGraphErrors * g42 = new TGraphErrors(nb,x, wmat,0,ew);
   TGraphErrors * g52 = 0; 
   TGraphErrors * g62 = 0; 
   if (clhep)     g52 = new TGraphErrors(nb,x, cmat,0, ec);
   if (clhep)     g62 = new TGraphErrors(nb,x, zmat,0, ez);
   DrawData("v^{T} * M * v",g12,g22,g32,g42,g52,g62);


   c1->cd(4);
   GetData("SMatrix_M*M",x,smat,es);
   GetData("TMatrix_M*M",x,tmat,et);
   GetData("SMatrix_sym_M*M",x,ymat,ey);
   GetData("TMatrix_sym_M*M",x,wmat,ew);
   if (clhep) { 
     GetData("HepMatrix_M*M",x,cmat,ec);
     GetData("HepMatrix_sym_M*M",x,zmat,ez);
   }
   TGraphErrors * g14 = new TGraphErrors(nb,x, smat,0,es);
   TGraphErrors * g24 = new TGraphErrors(nb,x, tmat,0,et);
   TGraphErrors * g34 = new TGraphErrors(nb,x, ymat,0,ey);
   TGraphErrors * g44 = new TGraphErrors(nb,x, wmat,0,ew);
   TGraphErrors * g54 = 0; 
   TGraphErrors * g64 = 0; 
   if (clhep)     g54 = new TGraphErrors(nb,x, cmat,0, ec);
   if (clhep)     g64 = new TGraphErrors(nb,x, zmat,0, ez);
   DrawData("A * B + C",g14,g24,g34,g44,g54,g64);

   c1->cd(5);
   GetData("SMatrix_At*M*A",x,smat,es);
   GetData("TMatrix_At*M*A",x,tmat,et);
   GetData("SMatrix_sym_At*M*A",x,ymat,ey);
   GetData("TMatrix_sym_At*M*A",x,wmat,ew);
   if (clhep) { 
     GetData("HepMatrix_At*M*A",x,cmat,ec);
     GetData("HepMatrix_sym_At*M*A",x,zmat,ez);
   }
   TGraphErrors * g15 = new TGraphErrors(nb,x, smat,0,es);
   TGraphErrors * g25 = new TGraphErrors(nb,x, tmat,0,et);
   TGraphErrors * g35 = new TGraphErrors(nb,x, ymat,0,ey);
   TGraphErrors * g45 = new TGraphErrors(nb,x, wmat,0,ew);
   TGraphErrors * g55 = 0; 
   TGraphErrors * g65 = 0; 
   if (clhep)     g55 = new TGraphErrors(nb,x, cmat,0, ec);
   if (clhep)     g65 = new TGraphErrors(nb,x, zmat,0, ez);
   DrawData("A * M * A^{T}",g15,g25,g35,g45,g55,g65);

   c1->cd(6);
   GetData("SMatrix_inv",x,smat,es);
   GetData("TMatrix_inv",x,tmat,et);
   GetData("SMatrix_sym_inv",x,ymat,ey);
   GetData("TMatrix_sym_inv",x,wmat,ew);
   if (clhep) { 
     GetData("HepMatrix_inv",x,cmat,ec);
     GetData("HepMatrix_sym_inv",x,zmat,ez);
   }
   TGraphErrors * g16 = new TGraphErrors(nb,x, smat,0,es);
   TGraphErrors * g26 = new TGraphErrors(nb,x, tmat,0,et);
   TGraphErrors * g36 = new TGraphErrors(nb,x, ymat,0,ey);
   TGraphErrors * g46 = new TGraphErrors(nb,x, wmat,0,ew);
   TGraphErrors * g56 = 0; 
   TGraphErrors * g66 = 0; 
   if (clhep)     g56 = new TGraphErrors(nb,x, cmat,0, ec);
   if (clhep)     g66 = new TGraphErrors(nb,x, zmat,0, ez);
   DrawData("A^{-1}",g16,g26,g36,g46,g56,g66);
   

   // TCanvas::Update() draws the frame, after which one can change it
   c1->Update();
   c1->SetFillColor(fillCol);
   c1->GetFrame()->SetFillColor(21);
    c1->GetFrame()->SetBorderSize(12);
   c1->Modified();

   if (drawSingleGraph) { 
     std::string c2Name = "c2_" + type; 
     TCanvas * c2 = new TCanvas(c2Name.c_str(),"Matrix Operations",200,10,700,600);
     DrawData("A * M * A^{T}",g15,g25,g35,g45,g55,g65);
     c2->SetRightMargin(0.028);
     c2->Update();
   }


}
