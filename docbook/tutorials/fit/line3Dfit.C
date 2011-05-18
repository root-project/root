//Fitting of a TGraph2D with a 3D straight line
//
// run this macro by doing: 
// 
// root>.x line3Dfit.C+
//

//Author: L. Moneta
//

   
#include <TMath.h>
#include <TGraph2D.h>
#include <TRandom2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TF2.h>
#include <TH1.h>
#include <TVirtualFitter.h>
#include <TPolyLine3D.h>
#include <Math/Vector3D.h>

using namespace ROOT::Math;


// define the parameteric line equation 
void line(double t, double *p, double &x, double &y, double &z) { 
   // a parameteric line is define from 6 parameters but 4 are independent
   // x0,y0,z0,z1,y1,z1 which are the coordinates of two points on the line
   // can choose z0 = 0 if line not parallel to x-y plane and z1 = 1; 
   x = p[0] + p[1]*t; 
   y = p[2] + p[3]*t;
   z = t; 
} 

// calculate distance line-point 
double distance2(double x,double y,double z, double *p) { 
   // distance line point is D= | (xp-x0) cross  ux | 
   // where ux is direction of line and x0 is a point in the line (like t = 0) 
   XYZVector xp(x,y,z); 
   XYZVector x0(p[0], p[2], 0. ); 
   XYZVector x1(p[0] + p[1], p[2] + p[3], 1. ); 
   XYZVector u = (x1-x0).Unit(); 
   double d2 = ((xp-x0).Cross(u)) .Mag2(); 
   return d2; 
}
bool first = true; 


// function to be minimized 
void SumDistance2(int &, double *, double & sum, double * par, int ) { 
   // the TGraph must be a global variable
   TGraph2D * gr = dynamic_cast<TGraph2D*>( (TVirtualFitter::GetFitter())->GetObjectFit() );
   assert(gr != 0);
   double * x = gr->GetX();
   double * y = gr->GetY();
   double * z = gr->GetZ();
   int npoints = gr->GetN();
   sum = 0;
   for (int i  = 0; i < npoints; ++i) { 
      double d = distance2(x[i],y[i],z[i],par); 
      sum += d;
#ifdef DEBUG
      if (first) std::cout << "point " << i << "\t" 
                           << x[i] << "\t" 
                           << y[i] << "\t" 
                           << z[i] << "\t" 
                           << std::sqrt(d) << std::endl; 
#endif
   }
   if (first) 
      std::cout << "Total sum2 = " << sum << std::endl;
   first = false;
}

void line3Dfit()
{
   gStyle->SetOptStat(0);
   gStyle->SetOptFit();


   //double e = 0.1;
   Int_t nd = 10000;


//    double xmin = 0; double ymin = 0;
//    double xmax = 10; double ymax = 10;

   TGraph2D * gr = new TGraph2D();

   // Fill the 2D graph
   double p0[4] = {10,20,1,2};

   // generate graph with the 3d points
   for (Int_t N=0; N<nd; N++) {
      double x,y,z = 0;
      // Generate a random number 
      double t = gRandom->Uniform(0,10);
      line(t,p0,x,y,z);
      double err = 1;
    // do a gaussian smearing around the points in all coordinates
      x += gRandom->Gaus(0,err);  
      y += gRandom->Gaus(0,err);  
      z += gRandom->Gaus(0,err);  
      gr->SetPoint(N,x,y,z);
      //dt->SetPointError(N,0,0,err);
   }
   // fit the graph now 
   
   TVirtualFitter *min = TVirtualFitter::Fitter(0,4);
   min->SetObjectFit(gr);
   min->SetFCN(SumDistance2);
   
  
   Double_t arglist[10];
   arglist[0] = 3;
   min->ExecuteCommand("SET PRINT",arglist,1);
  
   double pStart[4] = {1,1,1,1};
   min->SetParameter(0,"x0",pStart[0],0.01,0,0);
   min->SetParameter(1,"Ax",pStart[1],0.01,0,0);
   min->SetParameter(2,"y0",pStart[2],0.01,0,0);
   min->SetParameter(3,"Ay",pStart[3],0.01,0,0);
    
   arglist[0] = 1000; // number of function calls 
   arglist[1] = 0.001; // tolerance 
   min->ExecuteCommand("MIGRAD",arglist,2);

  //if (minos) min->ExecuteCommand("MINOS",arglist,0);
   int nvpar,nparx; 
   double amin,edm, errdef;
   min->GetStats(amin,edm,errdef,nvpar,nparx);
   min->PrintResults(1,amin);
   gr->Draw("p0");

   // get fit parameters
   double parFit[4];
   for (int i = 0; i <4; ++i) 
      parFit[i] = min->GetParameter(i);
   
   // draw the fitted line
   int n = 1000;
   double t0 = 0;
   double dt = 10;
   TPolyLine3D *l = new TPolyLine3D(n);
   for (int i = 0; i <n;++i) {
      double t = t0+ dt*i/n;
      double x,y,z;
      line(t,parFit,x,y,z);
      l->SetPoint(i,x,y,z);
   }
   l->SetLineColor(kRed);
   l->Draw("same");
   
   // draw original line
   TPolyLine3D *l0 = new TPolyLine3D(n);
   for (int i = 0; i <n;++i) {
      double t = t0+ dt*i/n;
      double x,y,z;
      line(t,p0,x,y,z);
      l0->SetPoint(i,x,y,z);
   }
   l0->SetLineColor(kBlue);
   l0->Draw("same");
   
}

int main() { 
   line3Dfit();
}
