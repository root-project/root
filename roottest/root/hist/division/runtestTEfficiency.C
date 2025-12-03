#include "TEfficiency.h"

bool testTEfficiency_vs_TGA(int nexp /* =1000 */,TEfficiency::EStatOption statOpt /* = TEfficiency::kBUniform */,bool mode /* = true */)
{ 
   gRandom->SetSeed(111);

   bool ok = true; 
   for (int i = 0; i < nexp; ++i) {

      //if (i>0 && i%500==0) cout << i << endl;
 
      // loop on the experiment
      double n = int(std::abs( gRandom->BreitWigner(0,5) ) ) + 1;
      double cut = ROOT::Math::beta_quantile(gRandom->Rndm(), 0.5, 0.5); 
      double k = int( cut * n); 
      TH1D * h1 = new TH1D("h1","h1",1,0,1);
      TH1D * h2 = new TH1D("h2","h2",1,0,1);
      h1->SetDirectory(0);
      h2->SetDirectory(0);
      h1->SetBinContent(1,k);      
      h2->SetBinContent(1,n);


      TGraphAsymmErrors * g = new TGraphAsymmErrors();
      if (statOpt== TEfficiency::kBUniform && mode) g->BayesDivide(h1,h2);
      else if (statOpt== TEfficiency::kBUniform && !mode) g->Divide(h1,h2,"cl=0.683 b(1,1)");
      else if (statOpt== TEfficiency::kBJeffrey &&  mode) g->Divide(h1,h2,"cl=0.683 b(0.5,0.5) mode");
      else if (statOpt== TEfficiency::kBJeffrey &&  !mode) g->Divide(h1,h2,"cl=0.683 b(0.5,0.5)");
      else if (statOpt== TEfficiency::kFCP ) g->Divide(h1,h2,"cl=0.683 cp");
      else if (statOpt== TEfficiency::kFAC ) g->Divide(h1,h2,"cl=0.683 ac");
      else if (statOpt== TEfficiency::kFFC ) g->Divide(h1,h2,"cl=0.683 fc");
      else if (statOpt== TEfficiency::kFWilson ) g->Divide(h1,h2,"cl=0.683 w");
      else if (statOpt== TEfficiency::kFNormal ) g->Divide(h1,h2,"cl=0.683 n");
      else { 
         cout << "invalid statistic options - exit " << endl;
         return false;
      }
      double eff =  g->GetY()[0];
      double eu  =  g->GetEYhigh()[0];
      double el  =  g->GetEYlow()[0];


      
      TEfficiency *  e = new TEfficiency(*h1,*h2);
   // eff->SetPosteriorMode(false);
      e->SetStatisticOption(statOpt);
      e->SetPosteriorMode(mode);
      if (mode) e->SetShortestInterval();
      e->SetConfidenceLevel(0.683);

      double eff2 = e->GetEfficiency(1);
      double el2 = e->GetEfficiencyErrorLow(1);
      double eu2 = e->GetEfficiencyErrorUp(1);

      double tol = 1.E-12;
      if (!TMath::AreEqualAbs(eff2, eff, 1.E-14)) { cerr << "Different efficiency " << eff2 << "  vs  " << eff << endl; ok=false;}
      if (!TMath::AreEqualAbs(el2, el, 1.E-14))  { cerr << "Different low error " << el2 << "  vs  " << el << endl; ok = false; }
      if (!TMath::AreEqualAbs(eu2, eu, 1.E-14))  { cerr << "Different up  error " << eu2 << "  vs " << eu << endl; ok = false; }
      if (!ok) { 
         cerr << "Iteration " << i << ":\t Error for (k,n) " << int(k) << " , " << int(n) << endl;
         break;
      }
      delete e;

      delete h1;
      delete h2;
      delete g;

   }

   if (ok) cout << "Comparison TEfficiency-TGraphAsymError :  OK for nevt = "  << nexp << std::endl;
   else  cout << "Comparison TEfficiency-TGraphAsymError :  FAILED ! "  << std::endl;

   return ok;
}

bool testNormal()
{
  float tol = 1e-3;
  bool ok = true;

  // test the 95% confidence intervals
  // taken from: http://www.measuringusability.com/wald.htm
  //
  // format: (k,n) -> lower bound, upper bound
  // (0,0) -> 0, 1
  // (3,7) -> 0.062, 0.795
  // (0,8) -> 0, 0
  // (3,12) -> 0.005, 0.495
  // (2,14) -> 0, 0.326
  // (5,18) -> 0.071, 0.485
  // (15,30) -> 0.321, 0.679
  // (10,10) -> 1, 1

  const int max = 8;
  Double_t k[max] = {0,3,0,3,2,5,15,10};
  Double_t n[max] = {0,7,8,12,14,18,30,10};
  Double_t low[max] = {0, 0.062, 0, 0.005, 0, 0.071, 0.321, 1};
  Double_t up[max] = {1, 0.795, 0, 0.495, 0.326, 0.485, 0.679, 1};

  Double_t alpha, beta;
  for(int i = 0; i < max; ++i)
  {
    alpha = k[i] + 0.5;
    beta = n[i] - k[i] + 0.5;
    
    if(fabs(TEfficiency::Normal(n[i],k[i],0.95,true) - up[i]) > tol)
    {
      cerr << "different upper bound for Normal interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::Normal(n[i],k[i],0.95,true) << " expecting: " << up[i] << endl;
      ok = false;
    }
    if(fabs(TEfficiency::Normal(n[i],k[i],0.95,false) - low[i]) > tol)
    {
      cerr << "different lower bound for Normal interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::Normal(n[i],k[i],0.95,false) << " expecting: " << low[i] << endl;
      ok = false;
    }
  }

  cout << "confidence interval for Normal ";
  (ok) ? cout << "OK" : cout << "FAILED"; cout << endl;
  
  return ok;
}

bool testWilson()
{
  float tol = 1e-3;
  bool ok = true;

  // test the 95% confidence intervals
  // taken from: http://www.measuringusability.com/wald.htm
  //
  // format: (k,n) -> lower bound, upper bound
  // (0,0) -> 0, 1
  // (3,7) -> 0.158, 0.750
  // (0,8) -> 0, 0.324
  // (3,12) -> 0.089, 0.532
  // (2,14) -> 0.040, 0.399
  // (5,18) -> 0.125, 0.509
  // (15,30) -> 0.332, 0.669
  // (10,10) -> 0.722, 1.000

  const int max = 8;
  Double_t k[max] = {0,3,0,3,2,5,15,10};
  Double_t n[max] = {0,7,8,12,14,18,30,10};
  Double_t low[max] = {0, 0.158, 0, 0.089, 0.040, 0.125, 0.332, 0.722};
  Double_t up[max] = {1, 0.750, 0.324, 0.532, 0.399, 0.509, 0.669, 1};

  Double_t alpha, beta;
  for(int i = 0; i < max; ++i)
  {
    alpha = k[i] + 0.5;
    beta = n[i] - k[i] + 0.5;
    
    if(fabs(TEfficiency::Wilson(n[i],k[i],0.95,true) - up[i]) > tol)
    {
      cerr << "different upper bound for Wilson interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::Wilson(n[i],k[i],0.95,true) << " expecting: " << up[i] << endl;
      ok = false;
    }
    if(fabs(TEfficiency::Wilson(n[i],k[i],0.95,false) - low[i]) > tol)
    {
      cerr << "different lower bound for Wilson interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::Wilson(n[i],k[i],0.95,false) << " expecting: " << low[i] << endl;
      ok = false;
    }
  }

  cout << "confidence interval for Wilson ";
  (ok) ? cout << "OK" : cout << "FAILED"; cout << endl;
  
  return ok;
}

bool testFeldmanCousins()
{
  float tol = 1e-3;
  bool ok = true;

  // test the 95% confidence intervals
  // taken from: http://people.na.infn.it/~lista/cgi/binomial/binomial.pl
  //
  // format: (k,n) -> lower bound, upper bound
  // (0,0) -> 0, 1
  // (3,7) -> 0.129, 0.775
  // (0,8) -> 0, 0.321
  // (3,12) -> 0.072, 0.548
  // (2,14) -> 0.026, 0.418
  // (5,18) -> 0.106, 0.531
  // (15,30) -> 0.324, 0.676
  // (10,10) -> 0.733, 1.000

  const int max = 8;
  Double_t k[max] = {0,3,0,3,2,5,15,10};
  Double_t n[max] = {0,7,8,12,14,18,30,10};
  Double_t low[max] = {0, 0.129, 0, 0.072, 0.026, 0.106, 0.324, 0.733};
  Double_t up[max] = {1, 0.775, 0.321, 0.548, 0.418, 0.531, 0.676, 1};

  Double_t alpha, beta;
  for(int i = 0; i < max; ++i)
  {
    alpha = k[i] + 0.5;
    beta = n[i] - k[i] + 0.5;
    
    if(fabs(TEfficiency::FeldmanCousins(n[i],k[i],0.95,true) - up[i]) > tol)
    {
      cerr << "different upper bound for Feldman-Cousins interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::FeldmanCousins(n[i],k[i],0.95,true) << " expecting: " << up[i] << endl;
      ok = false;
    }
    if(fabs(TEfficiency::FeldmanCousins(n[i],k[i],0.95,false) - low[i]) > tol)
    {
      cerr << "different lower bound for Feldman-Cousins interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::FeldmanCousins(n[i],k[i],0.95,false) << " expecting: " << low[i] << endl;
      ok = false;
    }
  }

  cout << "confidence interval for Feldman-Cousins ";
  (ok) ? cout << "OK" : cout << "FAILED"; cout << endl;
  
  return ok;
}

bool testClopperPearson()
{
  float tol = 1e-3;
  bool ok = true;

  // test the 95% confidence intervals
  // taken from: http://people.na.infn.it/~lista/cgi/binomial/binomial.pl
  //
  // format: (k,n) -> lower bound, upper bound
  // (0,0) -> 0, 1
  // (3,7) -> 0.099, 0.816
  // (0,8) -> 0, 0.369
  // (3,12) -> 0.055, 0.572
  // (2,14) -> 0.018, 0.428
  // (5,18) -> 0.097, 0.535
  // (15,30) -> 0.313, 0.687
  // (10,10) -> 0.692, 1.000

  const int max = 8;
  Double_t k[max] = {0,3,0,3,2,5,15,10};
  Double_t n[max] = {0,7,8,12,14,18,30,10};
  Double_t low[max] = {0, 0.099, 0, 0.055, 0.018, 0.097, 0.313, 0.692};
  Double_t up[max] = {1, 0.816, 0.369, 0.572, 0.428, 0.535, 0.687, 1};

  Double_t alpha, beta;
  for(int i = 0; i < max; ++i)
  {
    alpha = k[i] + 0.5;
    beta = n[i] - k[i] + 0.5;
    
    if(fabs(TEfficiency::ClopperPearson(n[i],k[i],0.95,true) - up[i]) > tol)
    {
      cerr << "different upper bound for Clopper-Pearson interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::ClopperPearson(n[i],k[i],0.95,true) << " expecting: " << up[i] << endl;
      ok = false;
    }
    if(fabs(TEfficiency::ClopperPearson(n[i],k[i],0.95,false) - low[i]) > tol)
    {
      cerr << "different lower bound for Clopper=Pearson interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::ClopperPearson(n[i],k[i],0.95,false) << " expecting: " << low[i] << endl;
      ok = false;
    }
  }

  cout << "confidence interval for Clopper-Pearson ";
  (ok) ? cout << "OK" : cout << "FAILED"; cout << endl;
  
  return ok;
}

bool testJeffreyPrior()
{
  float tol = 1e-3;
  bool ok = true;

  // test the 95% confidence intervals
  // taken from:
  // "Interval Estimation for a Binomial Proportion" Brown, Cai, DasGupta
  // Table 5
  //
  // format: (k,n) -> lower bound, upper bound
  // (0,0) -> 0.002, 0.998
  // (3,7) -> 0.139, 0.766
  // (0,8) -> 0, 0.262
  // (3,12) -> 0.076, 0.529
  // (2,14) -> 0.031, 0.385
  // (5,18) -> 0.115, 0.506
  // (15,30) -> 0.328, 0.672
  // (10,10) -> 0.783, 1.000
  //
  // alpha = k + 0.5
  // beta = n - k + 0.5

  const int max = 8;
  Double_t k[max] = {0,3,0,3,2,5,15,10};
  Double_t n[max] = {0,7,8,12,14,18,30,10};
  Double_t low[max] = {0.002, 0.139, 0, 0.076, 0.031, 0.115, 0.328, 0.783};
  Double_t up[max] = {0.998, 0.766, 0.262, 0.529, 0.385, 0.506, 0.672, 1};

  Double_t alpha, beta;
  for(int i = 0; i < max; ++i)
  {
    alpha = k[i] + 0.5;
    beta = n[i] - k[i] + 0.5;
    
    if(fabs(TEfficiency::BetaCentralInterval(0.95,alpha,beta,true) - up[i]) > tol)
    {
      cerr << "different upper bound for Jeffrey interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::BetaCentralInterval(0.95,alpha,beta,true) << " expecting: " << up[i] << endl;
      ok = false;
    }
    if(fabs(TEfficiency::BetaCentralInterval(0.95,alpha,beta,false) - low[i]) > tol)
    {
      cerr << "different lower bound for Jeffrey interval (" << k[i] << "," << n[i] << ")" << endl;
      cerr << "got: " << TEfficiency::BetaCentralInterval(0.95,alpha,beta,false) << " expecting: " << low[i] << endl;
      ok = false;
    }
  }

  cout << "confidence interval for Jeffrey prior ";
  (ok) ? cout << "OK" : cout << "FAILED"; cout << endl;
  
  return ok;
}

void runtestTEfficiency()
{
  // check consistency between TEfficiency and TGraphAsymmErrors
  cout << "uniform prior with mode: "; testTEfficiency_vs_TGA(1000,TEfficiency::kBUniform,true);
  cout << "uniform prior with mean: "; testTEfficiency_vs_TGA(1000,TEfficiency::kBUniform,false);
  cout << "Jeffrey prior with mode: "; testTEfficiency_vs_TGA(1000,TEfficiency::kBJeffrey,true);
  cout << "Jeffrey prior with mean: "; testTEfficiency_vs_TGA(1000,TEfficiency::kBJeffrey,false);
  cout << "Clopper-Pearson: "; testTEfficiency_vs_TGA(1000,TEfficiency::kFCP,false);
  cout << "Agresti-Coull: "; testTEfficiency_vs_TGA(1000,TEfficiency::kFAC,false);
  cout << "Feldman-Cousin: "; testTEfficiency_vs_TGA(1000,TEfficiency::kFFC,false);
  cout << "Wilson: "; testTEfficiency_vs_TGA(1000,TEfficiency::kFWilson,false);
  cout << "Normal: "; testTEfficiency_vs_TGA(1000,TEfficiency::kFNormal,false);

  // check confidence intervals for a few points
  testClopperPearson();
  testNormal();
  testWilson();
  testFeldmanCousins();
  testJeffreyPrior();
}

