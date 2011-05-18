// @(#)root/minuit2:$Id$
// Author: L. Moneta    12/2005  
/**
   test of a pure minimization passing a user FCN class directly to the 
   TFitterMinuit

*/
#include "TH1.h"
#include "TF1.h"
#include "TRandom3.h"
#include "TVirtualFitter.h"
#include "TStyle.h"
#include "Minuit2/FCNBase.h"
#include "TFitterMinuit.h"
#include "TSystem.h"
#include "TStopwatch.h"

#include <vector>
#include <iostream>

class MyFCN : public ROOT::Minuit2::FCNBase { 

public: 

  MyFCN(double a = 100, double b = 1) : fA(a), fB(b) {}

  double operator() (const std::vector<double> & x) const {
    // Rosebrock function
    return  fA*(x[1] - x[0]*x[0])*(x[1] - x[0]*x[0]) + fB*(1 - x[0])*(1 - x[0]);
  } 
  
  double Up() const { return 1.; }

private: 

  double fA;
  double fB; 

};

int testMinimize() { 

  gSystem->Load("libMinuit2");

  TFitterMinuit * minuit = new TFitterMinuit();

  MyFCN fcn;
  minuit->SetMinuitFCN(&fcn);
  // starting values 
  double startX = -1.2; 
  double startY = 1.0;
  // if not limited (vhigh <= vlow) 
  minuit->SetParameter(0,"x",startX,0.1,0,0);
  minuit->SetParameter(1,"y",startY,0.1,0,0);
  minuit->SetPrintLevel(3);
  // create Minimizer (default is Migrad)
  minuit->CreateMinimizer();
  int iret = minuit->Minimize();
  if (iret != 0) { 
     return iret; 
  }

  std::cout << "\nTest performances........\n\n"; 

  // test performances 
  int nMin = 10000; 
  TStopwatch w; 
  w.Start();
  for (int i = 0; i < nMin; ++i) { 
     minuit->Clear();
     // reset -everything
     //minuit->SetMinuitFCN(&fcn); 
     minuit->SetParameter(0,"x",startX,0.1,0,0);
     minuit->SetParameter(1,"y",startY,0.1,0,0);
     minuit->SetPrintLevel(0);
     // create Minimizer (default is Migrad)
     minuit->CreateMinimizer();
     iret = minuit->Minimize();
     if (iret != 0) { 
        std::cout << "Minimization failed - exit " ; 
        return iret; 
     }
  }
  w.Stop();
  std::cout << "\nTime: \t" << w.RealTime() << " , " << w.CpuTime() << std::endl;  
  return 0; 
}

#ifndef __CINT__
int main() { 
  int iret = testMinimize();
  if (iret != 0) { 
    std::cerr << "ERROR: Minimize test failed !" << std::endl;
    return iret;
  }
  return 0;
}
#endif
