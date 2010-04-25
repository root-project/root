// program to test distribution sampling

#include "Math/Factory.h"
#include "Math/DistSampler.h"
#include "TF1.h"
#include "TH1.h"

#include "TStopwatch.h"
#include "TRandom.h"
#include "TError.h"
// double Pdf(double x) { 
// }

using namespace ROOT::Math; 

int testDistSampler(int n = 10000) { 


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

int main() { 
   int iret = testDistSampler(); 
   if (iret)  std::cerr << "\ntestDistSampler: ....  FAILED!" << std::endl;
   else std::cerr << "\ntestDistSampler: ....  OK" << std::endl;
   return iret;
}

