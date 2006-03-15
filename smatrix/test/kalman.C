#include "Riostream.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TPaveLabel.h"
#include "TText.h"
#include "TLine.h"
#include "Math/SMatrix.h"
   
using namespace ROOT::Math; 
void kalman_do(int machine,int sym, int cut);
const int nx = 9;
const int ny = 7;
const Int_t n=nx*ny;

void kalman(int sym=1,int cut =6) {

  cout << "loading lib smatrix" << std::endl; 
  gSystem->Load("libSmatrix");

  kalman_do(0,sym,cut);
}   

void read_data(char *  machine, double * s, double * ss, double * t) { 

  std::string fileName = "kalman_" + std::string(machine) + ".root";
  TFile * file = new TFile(fileName.c_str() ); 
  SMatrix<double,9,7,ROOT::Math::MatRepStd<double,9,7> > *ms; 
  SMatrix<double,9,7,ROOT::Math::MatRepStd<double,9,7> > *mss; 
  SMatrix<double,9,7,ROOT::Math::MatRepStd<double,9,7> > *mt; 
  file->GetObject("SMatrix",ms);   
  file->GetObject("SMatrix_sym",mss);   
  file->GetObject("TMatrix",mt); 
  for (int i=0; i<n; ++i){
    s[i]  = ms->apply(i);
    ss[i] = mss->apply(i);
    t[i]  = mt->apply(i);
  }

  file->Close(); 
  delete file; 
   
}
  


void kalman_do(int machine,int sym, int cut) {
   //testing SMatrix[nx,ny]  2<=nx<=10,   2<=ny<<8
   //sym =0  shade cases where SMatrix is faster than TMatrix
   //sym =1  shade cases where SMatrix_Sym is faster than TMatrix
   //machine=1, Windows VC++7.1 (pcbrun3)
   //machine=2, Windows VC++8.0 (axel) (table to be filled by Axel)
   //machine=3, slc3 gcc3.2.3 (pcbrun)
   //machine=4, fc3, amd64 gcc3.4.3 (venus)
   //machine=5, solaris CC5.2 (refsol9)
                  
      
   TCanvas *c1 = 0;
   char tmachine[50];
   
   double s[n]; 
   double ss[n]; 
   double t[n]; 
 
  if (machine ==0) {
      //Linnux slc3 gcc 3.2.3
      sprintf(tmachine,"%s","slc3_ia32/gcc3.2.3");
      c1 = new TCanvas("kalmanslc3","slc3_ia32/gcc3.2.3",70,70,800,650);
      read_data("ref",s,ss,t);
   }
   if (machine ==1) {
      //Windows/VC++7.1
      sprintf(tmachine,"%s","Windows/VC++7.1");
      c1 = new TCanvas("kalmanvc7","windows/vc7.1",10,10,800,650);
      read_data("pcphsft15_win",s,ss,t);
   }
   if (machine ==2) {
      //windows/VC++8.0
      sprintf(tmachine,"%s","Windows/VC++8.0");
      c1 = new TCanvas("kalmanvc8","Windows/VC++8.0",30,30,800,650);
      //read_data("pcphsft15_win2",s,ss,t);
   }
   if (machine ==3) {
      //slc3-amd64/gcc3.4.3
      sprintf(tmachine,"%s","SLC3-amd64/gcc3.4");
      c1 = new TCanvas("kalmanslc3-amd64","slc3-amd64/gcc3.4",50,50,800,650);
      read_data("slc3-amd64",s,ss,t);
   }
   if (machine ==4) {
      //amd64 fc3/gcc3.4.3
     sprintf(tmachine,"%s","amd64 FC3/gcc3.4.3");
     c1 = new TCanvas("kalmanamd64","amd64 fc3/gcc3.4.3",70,70,800,650);
     //read_data("slc3-amd64",s,ss,t);
   }
   if (machine ==5) {
      //solaris CC5.2
      sprintf(tmachine,"%s","Solaris/CC5.2");
      c1 = new TCanvas("kalmansol","solaris/cc5.2",90,90,800,650);
      read_data("refsol9",s,ss,t);
   }
   if (machine ==7) {
      //Linnux slc3 gcc 3.2.3
      sprintf(tmachine,"%s","New slc3_ia32/gcc3.2.3");
      c1 = new TCanvas("kalmanslc3_new","New slc3_ia32/gcc3.2.3",70,70,800,650);
      read_data("pcphsft19_new",s,ss,t);
   }
  if (machine ==8) {
      //MACOS powerPC
      sprintf(tmachine,"%s","MacOS/gcc 4.0");
      c1 = new TCanvas("kalmanMac","MacOS/gcc 4.0",10,10,800,650);
      read_data("sealg5",s,ss,t);
   }
 
   c1->SetHighLightColor(19);
   int i,j;
   double xmin = 0.1;
   double xmax = 0.9;
   double ymin = 0.2;
   double ymax = 0.9;
   double dx = (xmax-xmin)/nx;
   double dy = (ymax-ymin)/ny;
   TPaveLabel *pl=0;
   TBox box;
   if (sym == 0) box.SetFillColor(kBlack);
   else          box.SetFillColor(kBlue);
   box.SetFillStyle(3002);
   for (i=0;i<nx;i++) {
     for (j=0;j<ny;j++) {
         if (sym == 0) {
            if (s[ny*i+j] > t[ny*i+j]) continue;
            box.DrawBox(xmin+i*dx,ymax-(j+1)*dy,xmin+(i+1)*dx,ymax-j*dy);
            pl = new TPaveLabel(xmin+5*dx,0.025,xmax,0.075,"SMatrix better than TMatrix","brNDC");
         } else {
            if (ss[ny*i+j] > t[ny*i+j]) continue;
            box.DrawBox(xmin+i*dx,ymax-(j+1)*dy,xmin+(i+1)*dx,ymax-j*dy);
            pl = new TPaveLabel(xmin+5*dx,0.025,xmax,0.075,"SMatrix_Sym better than TMatrix","brNDC");
         }
         pl->SetFillStyle(box.GetFillStyle());
         pl->SetFillColor(box.GetFillColor());
         pl->Draw();
      }
   }


   TLine line;
   TText tss,ts,tt;
   tss.SetTextColor(kBlue);
   tss.SetTextSize(0.031);
   ts.SetTextColor(kBlack);
   ts.SetTextSize(0.031);
   tt.SetTextColor(kRed);
   tt.SetTextSize(0.031);
   char text[10];
   ts.SetTextAlign(22);
   for (i=0;i<=nx;i++) {
      line.DrawLine(xmin+i*dx,ymin,xmin+i*dx,ymax);
      if(i==nx) continue;
      sprintf(text,"%d",i+2);
      ts.DrawText(xmin+(i+0.5)*dx,ymax+0.1*dy,text);
   }
   ts.SetTextAlign(32);
   for (i=0;i<=ny;i++) {
      line.DrawLine(xmin,ymax-i*dy,xmax,ymax-i*dy);
      if(i==ny) continue;
      sprintf(text,"%d",i+2);
      ts.DrawText(xmin-0.1*dx,ymax-(i+0.5)*dy,text);
   }
   tss.SetTextAlign(22);
   ts.SetTextAlign(22);
   tt.SetTextAlign(22);
   double sums1  = 0; 
   double sumss1 = 0; 
   double sumt1  = 0; 
   double sums2  = 0; 
   double sumss2 = 0; 
   double sumt2  = 0; 
   for (i=0;i<nx;i++) {
     for (j=0;j<ny;j++) {
         sprintf(text,"%6.2f",ss[ny*i+j]);
         tss.DrawText(xmin+(i+0.5)*dx,ymax -(j+0.22)*dy,text);
         sprintf(text,"%6.2f",s[ny*i+j]);
         ts.DrawText(xmin+(i+0.5)*dx,ymax -(j+0.5)*dy,text);
         sprintf(text,"%6.2f",t[ny*i+j]);
         tt.DrawText(xmin+(i+0.5)*dx,ymax -(j+0.78)*dy,text);
	 if ( i <=cut-2 && j <=cut-2) { 
	    sums1  += s[ny*i+j];
	    sumss1 += ss[ny*i+j];
	    sumt1  += t[ny*i+j];
	 }
	 else { 
	    sums2  += s[ny*i+j];
	    sumss2 += ss[ny*i+j];
	    sumt2  += t[ny*i+j];
	 }
      }
   }
   tss.DrawText(xmin+0.5*dx,0.05,"SMatrix_Sym");
   ts.DrawText (xmin+2.5*dx,0.05,"SMatrix");
   tt.DrawText (xmin+4*dx,0.05,"TMatrix");
   ts.SetTextSize(0.05);
   char title[100];
   sprintf(title,"TestKalman [nx,ny] : %s",tmachine);
   ts.DrawText(0.5,0.96,title);


   // summary boxes 

   double ylow = 0.082;

   tt.SetTextAlign(22);
   tss.SetTextColor(kBlue);
   tss.SetTextSize(0.031);
   ts.SetTextColor(kBlack);
   ts.SetTextSize(0.031);
   tt.SetTextColor(kRed);
   tt.SetTextSize(0.031);

   TText tl;
   tl.SetTextColor(kBlack);
   tl.SetTextSize(0.04);
   tt.SetTextAlign(22);

   i = 2;
   sprintf(text,"N1,N2 <= %d",cut);
   tl.DrawText (xmin+i*dx-0.15,ylow+0.04,text);
   if (sym == 0) { 
     if (sums1 <= sumt1) 
       box.DrawBox(xmin+i*dx,ylow,xmin+(i+1)*dx,ylow+dy);
   }  
   else {  
     if (sumss1 <= sumt1) 
       box.DrawBox(xmin+i*dx,ylow,xmin+(i+1)*dx,ylow+dy);
   }
   sprintf(text,"%6.2f",sumss1);
   tss.DrawText(xmin+(i+0.5)*dx,ylow+0.078,text);
   sprintf(text,"%6.2f",sums1);
   ts.DrawText(xmin+(i+0.5)*dx,ylow+0.05,text);
   sprintf(text,"%6.2f",sumt1);
   tt.DrawText(xmin+(i+0.5)*dx,ylow+0.022,text);


   i = 5; 
   sprintf(text,"N1,N2 >  %d",cut);
   tl.DrawText (xmin+i*dx-0.15,ylow+0.04,text);
   if (sym == 0) { 
     if (sums2 <= sumt2) 
       box.DrawBox(xmin+i*dx,ylow,xmin+(i+1)*dx,ylow+dy);
   }  
   else {  
     if (sumss2 <= sumt2) 
       box.DrawBox(xmin+i*dx,ylow,xmin+(i+1)*dx,ylow+dy);
   }
   sprintf(text,"%6.2f",sumss2);
   tss.DrawText(xmin+(i+0.5)*dx,ylow+0.078,text);
   sprintf(text,"%6.2f",sums2);
   ts.DrawText(xmin+(i+0.5)*dx,ylow+0.05,text);
   sprintf(text,"%6.2f",sumt2);
   tt.DrawText(xmin+(i+0.5)*dx,ylow+0.022,text);

   i= 8; 
   tl.DrawText (xmin+i*dx-0.15,ylow+0.04,"All N1,N2 ");
   if (sym == 0) { 
     if (sums1+sums2 <= sumt1+sumt2) 
       box.DrawBox(xmin+i*dx,ylow,xmin+(i+1)*dx,ylow+dy);
   }  
   else {  
     if (sumss1+sumss2 <= sumt1+sumt2) 
       box.DrawBox(xmin+i*dx,ylow,xmin+(i+1)*dx,ylow+dy);
   }
   sprintf(text,"%6.2f",sumss1+sumss2);
   tss.DrawText(xmin+(i+0.5)*dx,ylow+0.078,text);
   sprintf(text,"%6.2f",sums1+sums2);
   ts.DrawText(xmin+(i+0.5)*dx,ylow+0.05,text);
   sprintf(text,"%6.2f",sumt1+sumt2);
   tt.DrawText(xmin+(i+0.5)*dx,ylow+0.022,text);
}
        
