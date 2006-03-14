void kalman(int sym=1,int cut =6) {

  cout << "loading lib smatrix" << std::endl; 
   gSystem->Load("libSmatrix");
   using namespace ROOT::Math; 

   kalman_do(0,sym,cut);
//    kalman_do(1,sym,cut);
//    kalman_do(2,sym,cut);
//    kalman_do(3,sym,cut);
//    kalman_do(4,sym,cut);
//    kalman_do(5,sym,cut);
//    kalman_do(6,sym,cut);
//    kalman_do(7,sym,cut);
//    kalman_do(8,sym,cut);
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
  for (int i=0; i<63; ++i){
    s[i]  = ms->apply(i);
    ss[i] = mss->apply(i);
    t[i]  = mt->apply(i);
  }

  file->Close(); 
  delete file; 
   
}
  


void kalman_do(int machine,int sym, int cut) {
   //testing SMatrix[nx,ny]  2<=nx<=10,   2<=ny<<8
   const int nx = 9;
   const int ny = 7;
   const Int_t n=nx*ny;
   //sym =0  shade cases where SMatrix is faster than TMatrix
   //sym =1  shade cases where SMatrix_Sym is faster than TMatrix
   //machine=1, Windows VC++7.1 (pcbrun3)
   //       nx =     2     3     4     5     6     7       8      9     10
//    double ss1[n] = {0.62, 0.82, 1.40, 2.86, 4.54, 18.98,  6.87, 38.78, 51.90,
//                     0.98, 1.15, 3.43, 6.28, 8.59, 26.90, 14.33, 48.22, 72.52,     
//                     0.81, 1.80, 2.31, 4.21, 6.21, 23.05,  8.95, 42.73, 60.00,
//                     2.70, 4.90, 6.56, 9.91,13.34, 37.63, 23.00, 66.04, 98.55,
//                     5.47, 6.90, 9.45,13.54,17.03, 43.51, 27.43, 74.07,117.24,
//                    17.70,21.39,25.32,31.74,38.34, 67.62, 51.78,103.70,152.29,
//                     9.38, 9.87,10.85,13.74,17.36, 39.60, 19.34, 68.86, 82.40};
//    double s1[n]  = {0.55, 0.72, 1.38, 2.51, 3.64,  5.17,  3.36,  9.69, 10.87,            
//                     0.96, 1.11, 3.77, 5.83, 8.09, 12.00, 11.44, 17.82, 22.13,
//                     1.20, 1.47, 2.02, 3.56, 4.76,  6.38,  4.98, 10.46, 14.03,
//                     2.17, 4.47, 7.10, 9.51,12.34, 19.07, 19.66, 26.38, 32.73,
//                     5.36, 6.92,10.03,11.77,15.71, 23.98, 24.16, 32.34, 37.20,
//                     6.94, 8.84,12.09,14.77,21.23, 27.45, 28.25, 38.18, 46.52,
//                     5.83, 7.20, 8.04,10.09,12.50, 14.24, 13.07, 18.92, 23.24};
//    double t1[n]  = {1.86, 2.31, 2.76, 3.08, 4.17,  4.53,  4.82,  6.10,  6.56,
//                     2.49, 2.75, 3.78, 4.16, 5.10,  5.65,  6.31,  8.24,  9.80,
//                     2.84, 3.41, 3.72, 5.49, 6.49,  8.13,  8.92, 10.32, 12.46,
//                     3.60, 4.76, 5.68, 6.17, 8.51,  9.70, 11.46, 12.86, 14.85,
//                     5.64, 6.25, 7.15, 9.25,10.36, 12.79, 12.42, 15.59, 18.02,
//                    21.40,22.08,24.27,25.60,29.57, 28.56, 30.92, 35.55, 38.61,
//                    26.90,28.28,31.56,30.95,33.38, 34.52, 34.94, 38.65, 42.15};
//    //machine=2, Windows VC++8.0 (axel) (table to be filled by Axel)
//    double ss2[n] = {0.62, 0.82, 1.40, 2.86, 4.54, 18.98,  6.87, 38.78, 51.90,
//                     0.98, 1.15, 3.43, 6.28, 8.59, 26.90, 14.33, 48.22, 72.52,     
//                     0.81, 1.80, 2.31, 4.21, 6.21, 23.05,  8.95, 42.73, 60.00,
//                     2.70, 4.90, 6.56, 9.91,13.34, 37.63, 23.00, 66.04, 98.55,
//                     5.47, 6.90, 9.45,13.54,17.03, 43.51, 27.43, 74.07,117.24,
//                    17.70,21.39,25.32,31.74,38.34, 67.62, 51.78,103.70,152.29,
//                     9.38, 9.87,10.85,13.74,17.36, 39.60, 19.34, 68.86, 82.40};
//    double s2[n]  = {0.55, 0.72, 1.38, 2.51, 3.64,  5.17,  3.36,  9.69, 10.87,            
//                     0.96, 1.11, 3.77, 5.83, 8.09, 12.00, 11.44, 17.82, 22.13,
//                     1.20, 1.47, 2.02, 3.56, 4.76,  6.38,  4.98, 10.46, 14.03,
//                     2.17, 4.47, 7.10, 9.51,12.34, 19.07, 19.66, 26.38, 32.73,
//                     5.36, 6.92,10.03,11.77,15.71, 23.98, 24.16, 32.34, 37.20,
//                     6.94, 8.84,12.09,14.77,21.23, 27.45, 28.25, 38.18, 46.52,
//                     5.83, 7.20, 8.04,10.09,12.50, 14.24, 13.07, 18.92, 23.24};
//    double t2[n]  = {1.86, 2.31, 2.76, 3.08, 4.17,  4.53,  4.82,  6.10,  6.56,
//                     2.49, 2.75, 3.78, 4.16, 5.10,  5.65,  6.31,  8.24,  9.80,
//                     2.84, 3.41, 3.72, 5.49, 6.49,  8.13,  8.92, 10.32, 12.46,
//                     3.60, 4.76, 5.68, 6.17, 8.51,  9.70, 11.46, 12.86, 14.85,
//                     5.64, 6.25, 7.15, 9.25,10.36, 12.79, 12.42, 15.59, 18.02,
//                    21.40,22.08,24.27,25.60,29.57, 28.56, 30.92, 35.55, 38.61,
//                    26.90,28.28,31.56,30.95,33.38, 34.52, 34.94, 38.65, 42.15};
//    //machine=3, slc3 gcc3.2.3 (pcbrun)
//    double ss3[n] = {0.53, 0.85, 0.94, 1.84, 2.63, 3.61, 3.27, 5.61, 7.35,
//                     1.16, 1.72, 2.27, 3.91, 5.49, 7.08, 7.49,10.64,13.59,
//                     0.99, 1.39, 1.69, 2.67, 3.91, 5.10, 5.04, 7.54, 9.94,
//                     2.39, 3.27, 4.21, 6.37, 9.03,10.89,11.96,17.11,21.24,
//                     3.49, 4.74, 5.96, 9.28,11.55,13.53,15.16,20.88,25.82,
//                     8.03, 9.52,10.54,13.30,17.43,19.05,20.96,27.54,34.34,
//                     9.07, 9.55,10.69,12.15,13.81,15.59,15.47,19.88,22.83};
//    double s3[n]  = {0.48, 0.78, 0.81, 1.71, 2.62, 3.13, 2.38, 5.74, 8.26,
//                     0.95, 1.54, 2.04, 3.54, 5.07, 6.17, 6.17,10.57,14.71,
//                     0.79, 1.13, 1.31, 2.26, 3.70, 4.12, 3.46, 6.87,10.56,
//                     1.87, 2.87, 3.68, 5.73, 8.60, 9.59,10.04,16.73,21.62,
//                     2.79, 4.13, 5.15, 8.09,11.14,11.97,13.23,20.67,26.94,
//                     6.70, 8.34, 9.31,12.19,15.22,16.78,18.45,26.01,31.41,
//                     7.65, 8.15, 9.09,10.00,12.51,12.97,11.95,17.06,21.67};
//    double t3[n]  = {1.59, 1.94, 2.24, 2.37, 3.02, 3.24, 3.47, 3.88, 4.15,
//                     2.15, 2.10, 2.53, 2.74, 3.75, 3.82, 4.62, 5.68, 6.52,
//                     2.34, 2.71, 2.89, 3.69, 4.08, 5.67, 5.98, 7.02, 7.12,
//                     2.78, 3.20, 3.80, 4.17, 5.74, 6.44, 7.69, 7.76, 8.93,
//                     4.93, 5.63, 6.10, 7.00, 7.40, 8.80, 8.87, 9.94,10.47,
//                    17.59,17.28,18.21,19.32,20.96,19.80,20.57,21.08,22.94,
//                    18.33,20.35,22.51,22.42,22.54,23.48,24.07,25.22,27.35};
//    //machine=4, fc3, amd64 gcc3.4.3 (venus)
//    double ss4[n] = {0.31, 0.54, 0.71, 1.40, 2.27, 4.02, 3.62, 6.15, 8.58,
//                     0.56, 0.94, 1.23, 2.02, 3.08, 5.19, 5.58, 7.61,10.33,
//                     1.16, 1.47, 1.70, 2.73, 3.89, 5.80, 5.40, 8.38,11.46,
//                     2.72, 3.23, 5.83, 4.81, 6.28, 8.77, 8.53,12.04,15.51,
//                     6.21, 7.13, 7.59, 8.90,10.65,13.11,13.13,16.28,20.54,
//                     6.52, 7.97, 9.06,11.40,13.62,14.38,16.92,22.22,28.20,
//                     7.47, 8.69, 9.82,11.78,13.86,16.66,13.54,22.12,26.08};
//    double s4[n]  = {0.21, 0.35, 0.45, 0.84, 1.07, 1.68, 1.79, 2.73, 3.38,
//                     0.34, 0.54, 0.73, 1.14, 1.59, 2.95, 5.24, 6.18, 4.58,
//                     0.45, 0.62, 0.81, 1.36, 1.69, 2.49, 2.62, 3.85, 4.76,
//                     0.88, 1.14, 1.45, 2.01, 2.75, 3.71, 4.23, 5.41, 6.96,
//                     1.69, 2.09, 2.52, 3.20, 3.95, 5.04, 5.95, 6.93, 8.66,
//                     3.54, 4.09, 4.79, 5.70, 6.71, 7.89, 8.92,10.70,12.75,
//                     4.60, 4.88, 5.44, 6.20, 6.95, 8.10, 8.45,10.85,11.79};
//    double t4[n]  = {1.22, 1.36, 1.53, 1.76, 2.24, 2.45, 3.11, 3.47, 3.87,
//                     1.33, 1.66, 1.91, 2.08, 2.67, 3.06, 4.12, 4.76, 5.26,
//                     1.60, 1.92, 2.11, 2.44, 3.23, 3.86, 5.19, 5.83, 6.61,
//                     1.94, 2.30, 2.67, 2.99, 3.98, 4.90, 6.34, 7.37, 8.20,
//                     3.19, 3.63, 4.10, 4.92, 5.52, 6.06, 8.25, 9.33,10.25,
//                     9.03, 9.59,10.66,11.77,12.57,12.49,15.40,16.44,17.74,
//                    11.72,12.21,13.70,14.77,15.89,16.99,19.39,21.41,22.66};
//    //machine=5, solaris CC5.2 (refsol9)
//    double ss5[n] = {2.45, 5.51, 4.26,15.82, 25.25, 40.27, 42.73, 94.16,138.53,
//                     5.41,10.34,12.62,24.90, 36.27, 54.10, 60.15,118.23,167.91,
//                     5.35,10.56,12.70,24.86, 37.09, 54.70, 59.86,121.30,174.78,
//                    12.12,19.86,24.81,40.63, 57.22, 84.16, 88.55,160.13,221.61,
//                    19.15,28.59,36.06,55.63, 73.65,105.00,111.06,191.30,261.49,
//                    35.95,48.49,56.20,77.75,101.31,126.66,145.51,237.68,311.50,
//                    37.21,47.07,54.24,74.43, 94.59,126.02,136.83,237.63,314.84};
//    double s5[n]  = {1.81, 3.75, 4.97,10.46, 16.61, 29.36, 39.94, 75.31,112.76,
//                     3.87, 7.26, 9.74,17.51, 25.73, 41.54, 54.70, 96.36,139.11
//                     4.02, 7.63, 9.87,17.43, 25.58, 41.03, 53.86, 99.57,142.33,
//                     8.24,13.61,18.84,29.47, 41.06, 61.65, 79.13,132.15,183.37,
//                    13.52,20.20,26.38,40.31, 54.47, 75.92, 98.20,156.40,216.39,
//                    24.75,34.47,42.32,56.66, 75.96, 97.87,127.02,196.06,259.64,
//                    28.56,35.47,42.50,54.81, 69.00,104.14,127.13,202.57,272.12};
//    double t5[n]  = {4.26, 4.96, 5.59, 6.38,  7.83,  8.92, 10.07, 11.34, 12.70,
//                     5.00, 5.67, 6.67, 7.85, 10.01, 11.22, 12.76, 17.09, 19.13,
//                     6.11, 7.02, 8.23, 9.76, 12.04, 16.64, 18.76, 21.29, 24.09,
//                     7.40, 8.77,10.50,12.37, 17.75, 20.21, 22.81, 26.12, 29.50,
//                    10.94,12.97,15.03,20.08, 23.58, 26.56, 29.94, 33.62, 37.89,
//                    38.37,40.33,45.42,48.58, 52.47, 56.01, 60.74, 65.36, 70.51,
//                    46.43,49.06,54.27,58.21, 62.56, 67.12, 70.47, 77.05, 82.62};
                  
      
   TCanvas *c1 = 0;
   char tmachine[50];
   
   //double *ss=0, *s=0, *t=0;
   double s[63]; 
   double ss[63]; 
   double t[63]; 
 
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

   int i = 2;
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
        
