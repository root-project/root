
#include "TStopwatch.h"
#include "TUnuran.h"

#include "TH3.h"
#include "TF3.h"
#include "TCanvas.h"

#include "TRandom.h"
#include "TSystem.h"
#include "TApplication.h"

// #include "Math/ProbFunc.h"
// #include "Math/DistFunc.h"


#include <iostream> 

using std::cout; 
using std::endl; 


double gaus3d(double *x, double *p) { 

   double sigma_x = p[0]; 
   double sigma_y = p[1];
   double sigma_z = p[2];
   double rho = p[2]; 
   double u = x[0] / sigma_x ;
   double v = x[1] / sigma_y ;
   double w = x[2] / sigma_z ;
   double c = 1 - rho*rho ;
   double result = (1 / (2 * TMath::Pi() * sigma_x * sigma_y * sigma_z * sqrt(c))) 
      * exp (-(u * u - 2 * rho * u * v + v * v + w*w) / (2 * c));
   return result;
}



void unuranMulti3D() { 

   gRandom->SetSeed(0);

   gSystem->Load("libMathCore");
   gSystem->Load("libUnuran");

   // simple test of unuran

   

   TH3D * h1 = new TH3D("h1","gaussian 3D distribution",50,-10,10,50,-10,10,50,-10,10);
   TH3D * h2 = new TH3D("h2","gaussian 3D distribution",50,-10,10,50,-10,10,50,-10,10);

   

   TF3 * f = new TF3("g3d",gaus3d,-10,10,-10,10,-10,10,3); 
   double par[3] = {1,1,0.5}; 
   f->SetParameters(par); 

   f->SetNpx(100);
   f->SetNpy(100);
   f->SetNpz(100);
   std::cout << " npoints" << f->GetNpx() << "  "  << f->GetNpy() <<  "  "  << f->GetNpz() << std::endl;

   TCanvas * c1 = new TCanvas("c1_unuranMulti","Multidimensional distribution",10,10,500,500); 
   c1->Divide(1,2);

//    cout << fc->Eval(-11) << "  " <<    fc->Eval(1) << endl;

   TUnuranDistrMulti dist(f); 
   TUnuran unr(gRandom);
   //std::string method = "method=vnrou";
   std::string method = "method=hitro;use_boundingrectangle=false "; 
   if ( !  unr.Init(dist,method,0) ) { 
      cout << "Error initializing unuran" << endl;
      return;
   } 


   TStopwatch w; 
   w.Start(); 

   double x[3]; 
   for (int i = 0; i < 1000000; ++i) {  
      unr.SampleMulti(x);
//      cout << x[0] << " " << x[1] << endl;
      h1->Fill(x[0],x[1],x[2]);
   }

   w.Stop(); 
   cout << w.CpuTime() << endl;
   c1->cd(1); 
   h1->Draw();
//   h1->Fit("gaus");


   w.Start();
   for (int i = 0; i < 1000000; ++i) { 
      f->GetRandom3(x[0],x[1],x[2]);
      h2->Fill(x[0],x[1],x[2]); 
   }

   w.Stop(); 
   cout << w.CpuTime() << endl;

   
   c1->cd(2);
   h2->Draw("same");
   //  h2->Fit("gaus");

   std::cout << " chi2 test h1 vs h2 " << std::endl;
   h1->Chi2Test(h2,"P");
   

}

#ifndef __CINT__
int main(int argc, char **argv)
{
   TApplication theApp("App", &argc, argv);
   unuranMulti3D();
   theApp.Run();
   return 0;
}
#endif

