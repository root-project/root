#include "TGraph.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "Math/Interpolator.h"
#include <iostream>

#include <cmath>


void interpolate( const  ROOT::Math::Interpolator & itp ) { 

  std::cout << "x[i]     y[i]     deriv[i]     deriv2[i]    integral[i] \n" << std::endl; 
  // print result of interpolation
  const Int_t n = 51;
  Int_t i = 0;
  Float_t xcoord[n], ycoord[n];
  for ( double xi = 0; xi < 10; xi += 0.2) { 
    xcoord[i] = xi;
    ycoord[i] = itp.Eval(xi);
    double dyi = itp.Deriv(xi); 
    double dy2i = itp.Deriv2(xi);
    double igyi = itp.Integ(0, xi); 
    std::cout << xcoord[i]  << "  " << ycoord[i] << "  " << dyi << "  " << dy2i << "  " <<  igyi << std::endl;
    i++; 
  }

  TGraph *gr = new TGraph(n,xcoord,ycoord);
  gr->SetMarkerColor(kBlue);
  gr->SetMarkerStyle(21);
  gr->Draw("CP");

  return;
}


void testInterpolation() {

  // Create data
  int n = 10; 
  Float_t xorig[10], yorig[10];
  std::vector<double> x(n); 
  std::vector<double> y(n); 
  for (int i = 0; i < n; ++i) {  
    x[i] = i + 0.5 * std::sin(i+0.0); 
    y[i] = i + std::cos(i * i+0.0); 
    xorig[i] = x[i];
    yorig[i] = y[i];
  } 

  TCanvas *c1 = new TCanvas("c1","Original (red), Linear (upper left), Polynomial (upper right), Akima (lower left) and Akima Periodic (lower right) Interpolation",200,10,1400,1000);
  c1->Divide(2,2);
  c1->cd(1);

  TGraph *grorig = new TGraph(n,xorig,yorig);
  grorig->SetMarkerColor(kRed);
  grorig->SetMarkerStyle(21);
  grorig->GetYaxis()->SetRange(0,40);
  grorig->Draw("ACP"); 

  std::cout << "Linear Interpolation: " << std::endl;
  ROOT::Math::Interpolator itp1(x, y, ROOT::Math::Interpolation::LINEAR); 
  interpolate(itp1);


  c1->cd(2);
  grorig->Draw("ACP"); 

  std::cout << "Polynomial Interpolation: " << std::endl;
  ROOT::Math::Interpolator itp2(x, y, ROOT::Math::Interpolation::POLYNOMIAL); 
  interpolate(itp2);


  c1->cd(3);
  grorig->Draw("ACP"); 

  std::cout << "Akima Periodic Interpolation: " << std::endl;
  ROOT::Math::Interpolator itp3(x, y, ROOT::Math::Interpolation::AKIMA); 
  interpolate(itp3);



  c1->cd(4);
  grorig->Draw("ACP"); 

  std::cout << "Akima Interpolation: " << std::endl;
  ROOT::Math::Interpolator itp4(x, y, ROOT::Math::Interpolation::AKIMA_PERIODIC); 
  interpolate(itp4);


}


int main() {

  testInterpolation();
  return 0;

}
