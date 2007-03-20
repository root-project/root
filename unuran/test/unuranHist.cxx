#include "TUnuran.h"
#include "TUnuranEmpDist.h"

#include "TH1.h"
#include "TMath.h"
#include "TF1.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TRandom3.h"

#include "TStopwatch.h"
#include <iterator> 

#include <iostream>


int unuranHist() {

   int nbin = 100;
   double xmin = -5; 
   double xmax = 20;

   int n = 100000; // number of events generated and of reference histo
   int ns = 1000;   // number of events from starting histo

   // h0 is reference histo
   TH1D * h0 = new TH1D("h0","Landau ref data",nbin,xmin,xmax);
   h0->FillRandom("landau",n); 
   
   // h1 is low statistic histo from where we generate numbers
   TH1D * h1 = new TH1D("h1","Landau source data",nbin,xmin,xmax);
   h1->SetBuffer(ns);


   h1->FillRandom("landau",ns); 

//    const double * bf = h1->GetBuffer();
//    std::ostream_iterator<double> oi(std::cout," ,  ");
//    std::copy(bf, bf+ 2*h1->GetBufferLength() + 2, oi); 
//    std::cout << std::endl << std::endl; 
 
   int iret = 0;

   std::cout << "Test Using UnBinned data\n " << std::endl;


   TUnuran unr; 
   TUnuranEmpDist dist(h1);

   if (!unr.Init(dist)) return -1;

   TStopwatch w; 
   w.Start(); 
   TH1D * h2 = new TH1D("h2","Landau from Unuran unbin generation",nbin,xmin,xmax);
   for (int i = 0; i < n; ++i) { 
      h2->Fill( unr.Sample() );
   }
   w.Stop(); 
   double time = w.CpuTime()*1.E9/n; 
   std::cout << "Time using Unuran  " << unr.MethodName() << "   \t=\t " << time << "\tns/call" << std::endl;

   w.Start();
   TH1D * h3 = new TH1D("h3","h3",nbin,xmin,xmax);
   for (int i = 0; i < n; ++i) { 
      h3->Fill( h1->GetRandom() );
   }
   w.Stop(); 
   time = w.CpuTime()*1.E9/n;
   std::cout << "Time using TH1::GetRandom()  \t\t=\t " <<  time << "\tns/call" << std::endl;


   std::cout << "\nTest quality UnuRan  " << unr.MethodName() << "\t:\t";
   double prob = h1->Chi2Test(h2,"UUP");
   if (prob < 1.E-6 ) { 
      std::cerr << "Chi2 Test failed for UNURAN method " <<  unr.MethodName() << std::endl;
      iret = -1;
   }

   std::cout << "Test quality TH1::GetRandom \t\t:\t"; 
   h1->Chi2Test(h3,"UUP");
   std::cout << "Comparison UnuRan-TH1::GetRandom \t:\t"; 
   h2->Chi2Test(h3,"UUP");



   TCanvas * c1 = new TCanvas("c1_unuranEmp","Empirical distribution",10,10,800,800); 
   c1->Divide(1,2);
   c1->cd(1);
   h2->SetLineColor(kBlue);
   h2->Draw();
   h0->Draw("same");
   h3->SetLineColor(kRed);
   h3->Draw("same");



   // generate using binned data from h0
   std::cout << "\nTest Using Binned data\n " << std::endl;

   TUnuranEmpDist dist2(h0,false);


   if (!unr.Init(dist2) ) return -1;



   w.Start(); 
   TH1D * h4 = new TH1D("h4","Landa from Unuran binned generation",nbin,xmin,xmax);
   for (int i = 0; i < n; ++i) { 
      h4->Fill( unr.Sample() );
   }
   w.Stop(); 
   time = w.CpuTime()*1.E9/n; 
   std::cout << "Time using Unuran  " << unr.MethodName() << "   \t=\t " << time << "\tns/call" << std::endl;

   std::cout << "\nTest quality UNURAN  " << unr.MethodName() << "\t:\t"; 
   double prob2 = h0->Chi2Test(h4,"UUP");
   if (prob2 < 1.E-6) { 
      std::cerr << "Chi2 Test failed for UNURAN method " <<  unr.MethodName() << std::endl;
      iret = -2;
   }

   c1->cd(2);
   h4->SetLineColor(kBlue);
   h4->Draw();
   h0->Draw("same");


   return iret; 
}
#ifndef __CINT__
int main(int argc, char **argv)
{
   int iret = 0; 
   if (argc > 1) { 
      TApplication theApp("App",&argc,argv);
      iret =  unuranHist();
      theApp.Run();
   } 
   else 
      iret =  unuranHist();
   
   if (iret != 0) 
      std::cerr <<"\n\nUnuRan Empirical Distribution Test:\t  Failed !!!!!!!\n" << std::endl;
   else 
      std::cout << "\n\nUnuRan  Empirical Distribution Test:\t OK\n" << std::endl;
   return iret; 
}
#endif
