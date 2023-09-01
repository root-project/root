// program to test distribution sampling

#include "Math/Factory.h"
#include "Math/DistSampler.h"
#include "TF1.h"
#include "TH1.h"

#include "TStopwatch.h"
#include "TRandom.h"
#include "TError.h"
#include "TCanvas.h"
// double Pdf(double x) {
// }

using namespace ROOT::Math;


int testCont1D(int n)  {

   // test gaussian

   DistSampler * sampler = Factory::CreateDistSampler("Unuran");
   if (!sampler) return -1;
   TF1 * f = new TF1("pdf","gaus");
   f->SetParameters(1,0,1);

   TH1D * h1 = new TH1D("h1","h1",100,-3,3);
   TH1D * hr = new TH1D("hr","h2",100,-3,3);

   sampler->SetFunction(*f,1);
   bool ret = sampler->Init("AUTO");
   if (!ret)      return -1;


   TStopwatch w;
   w.Start();
   for (int i = 0; i < n; ++i) {
      h1->Fill(sampler->Sample1D() );
   }
   w.Stop();
   double c = 1.E9/double(n);
   std::cout << "Unuran sampling - (ns)/call = " << c*w.RealTime() << "   " << c*w.CpuTime() << std::endl;
   new TCanvas("Continous test");
   h1->SetLineColor(kBlue);
   h1->Draw();

   // generate ref histogram
   w.Start();
   for (int i = 0; i < n; ++i) {
      hr->Fill(gRandom->Gaus(0,1) );
   }
   w.Stop();
   std::cout << "TRandom::Gauss sampling - (ns)/call = " << c*w.RealTime() << "   " << c*w.CpuTime() << std::endl;

   hr->Draw("SAME");

   // do a Chi2 test
   // switch off printing of  info messages from chi2 test
   gErrorIgnoreLevel = 1001;
   double prob = h1->Chi2Test(hr,"UU");
   if (prob < 1.E-6) {
      std::cerr << "Chi2 test of generated histogram failed" << std::endl;
      return -2;
   }
   gErrorIgnoreLevel = 0;

   return 0;
}


int testDisc1D(int n)  {

   // test a discrete distribution


   DistSampler * sampler = Factory::CreateDistSampler("Unuran");
   if (!sampler) return -1;
   TF1 * f = new TF1("pdf","TMath::Poisson(x,[0])");
   double mu = 10;
   f->SetParameter(0,mu);

   TH1D * h1 = new TH1D("h2","h1",50,0,50);
   TH1D * hr = new TH1D("hd","h2",50,0,50);

   sampler->SetFunction(*f,1);
   sampler->SetMode(mu);
   sampler->SetArea(1);
   bool ret = sampler->Init("DARI");
   if (!ret)      return -1;


   TStopwatch w;
   w.Start();
   for (int i = 0; i < n; ++i) {
      h1->Fill(sampler->Sample1D() );
   }
   w.Stop();
   double c = 1.E9/double(n);
   std::cout << "Unuran sampling - (ns)/call = " << c*w.RealTime() << "   " << c*w.CpuTime() << std::endl;
   new TCanvas("Discrete test");
   h1->SetLineColor(kBlue);
   h1->Draw();

   // generate ref histogram
   w.Start();
   for (int i = 0; i < n; ++i) {
      hr->Fill(gRandom->Poisson(mu) );
   }
   w.Stop();
   std::cout << "TRandom::Poisson sampling - (ns)/call = " << c*w.RealTime() << "   " << c*w.CpuTime() << std::endl;

   hr->Draw("SAME");

   // do a Chi2 test
   // switch off printing of  info messages from chi2 test
   gErrorIgnoreLevel = 1001;
   double prob = h1->Chi2Test(hr,"UU");
   if (prob < 1.E-6) {
      std::cerr << "Chi2 test of generated histogram failed" << std::endl;
      return -2;
   }
   gErrorIgnoreLevel = 0;

   return 0;
}


int testDistSampler(int n = 10000) {
   int iret = 0;
   iret |= testCont1D(n);
   iret |= testDisc1D(n);
   return iret;
}
int main() {
   int iret = testDistSampler();
   if (iret)  std::cerr << "\ntestDistSampler: ....  FAILED!" << std::endl;
   else std::cerr << "\ntestDistSampler: ....  OK" << std::endl;
   return iret;
}

