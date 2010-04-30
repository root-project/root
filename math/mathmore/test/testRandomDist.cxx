#include "Math/Random.h"
#include "Math/GSLRndmEngines.h"
#include "Math/DistFunc.h"
#include "TStopwatch.h"
#include "TRandom3.h"
#include "TRandom2.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TMath.h"
#include <iostream>
#include <cmath>
#include <typeinfo>
#define HAVE_UNURAN
#ifdef HAVE_UNURAN
#include "UnuRanDist.h"
#endif

#ifdef HAVE_CLHEP
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonT.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandGaussT.h"
#include "CLHEP/Random/RandBinomial.h"
#include "CLHEP/Random/JamesRandom.h"
#endif


#ifndef PI
#define PI       3.14159265358979323846264338328      /* pi */
#endif



#ifndef NEVT
#define NEVT 1000000
#endif

//#define TEST_TIME

using namespace ROOT::Math;
#ifdef HAVE_CLHEP
using namespace CLHEP;
#endif

static bool fillHist = false;



void testDiff(TH1D & h1, TH1D & h2, const std::string & name="") { 
  
  double chi2; 
  int ndf; 
  if (h1.GetEntries() == 0 && h2.GetEntries() == 0) return; 
  int igood = 0; 
  double prob = h1.Chi2TestX(&h2,chi2,ndf,igood,"UU"); 
  std::cout << "\nTest " << name << " chi2 = " << chi2 << " ndf " << ndf << " prob = " << prob;
  if (igood) std::cout << " \t\t" << " igood = " << igood;
  std::cout << std::endl; 

  std::string cname="c1_" + name; 
  std::string ctitle="Test of " + name; 
  TCanvas *c1 = new TCanvas(cname.c_str(), ctitle.c_str(),200,10,800,600);
  h1.DrawCopy();
  h2.DrawCopy("Esame");
  c1->Update();


}

template <class R> 
std::string findName( const R & r) { 

  std::string type = typeid(r).name(); 
  if (type.find("GSL") != std::string::npos ) 
    return "ROOT::Math::Random";
  else if (type.find("TRandom") != std::string::npos )
    return "TRandom           "; 
  else if (type.find("UnuRan") != std::string::npos )
    return "UnuRan            "; 
  else if (type.find("Rand") != std::string::npos )
    return "CLHEP             "; 
  
  return   "?????????         ";
}


template <class R> 
void testPoisson( R & r,double mu,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    int n = r.Poisson(mu );
    if (fillHist)
      h.Fill( double(n) );
  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Poisson - mu = " << mu << "\t\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testPoisson(r,mu,h);
}



// Knuth Algorith for Poisson (used also in GSL) 
template <class R> 
unsigned int genPoisson( R & r, double mu) {

  // algorithm described by Knuth Vol 2. 2nd edition pag. 132
  // for generating poisson deviates when mu is large

  unsigned int k = 0; 

  while (mu > 20) { 

    const double alpha = 0.875;  // 7/8
    unsigned int m = static_cast<unsigned int> ((alpha*mu) ); 

    // generate xg according to a Gamma distribution of m
    double sqm = std::sqrt( 2.*m -1.);
    double pi = TMath::Pi();
    double x,y,v;
    do { 
      do { 
	y = std::tan( pi * r.Rndm() );
	x = sqm * y + m - 1.;
      }
      while (x <= 0); 
      v = r.Rndm(); 
    }
    while ( v > (1 + y * y) * std::exp ( (m - 1) * std::log (x / (m - 1)) - sqm * y));

    // x is now distributed according to a gamma of m 
      
    if ( x >= mu ) 
      return k + r.Binomial( m-1, mu/x); 

    else { 
    // continue the loop decresing mu
      mu -= x; 
      k += m; 
    }
  }
  // for lower values of mu use rejection method from exponential
  double expmu = TMath::Exp(-mu);
  double pir = 1.0; 
  do { 
    pir *= r.Rndm();
    k++;
  }
  while (pir > expmu); 
  return k -1; 
   
} 



// Numerical Receip algorithm  for Poisson (used also in CLHEP) 
template <class R> 
unsigned int genPoisson2( R & r, double mu) {

  //double om = getOldMean();




  if( mu < 12.0 ) {

    double expmu = TMath::Exp(-mu);
    double pir = 1.0; 
    unsigned int k = 0; 
    do { 
      pir *= r.Rndm();
      k++;
    }
    while (pir > expmu); 
    return k-1; 
  }
  // for large mu values (should care for values larger than 2E9) 
  else {
    
    double em, t, y;
    double sq, alxm, g;
    double pi = TMath::Pi();

    sq = std::sqrt(2.0*mu);
    alxm = std::log(mu);
    g = mu*alxm - TMath::LnGamma(mu + 1.0);
    
    do {
      do {
	y = std::tan(pi*r.Rndm());
	em = sq*y + mu;
      } while( em < 0.0 );

      em = std::floor(em);
      t = 0.9*(1.0 + y*y)* std::exp(em*alxm - TMath::LnGamma(em + 1.0) - g);
    } while( r.Rndm() > t );

    return static_cast<unsigned int> (em);

  }

}


template <class R> 
void testPoisson2( R & r,double mu,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) {
    //    int n = genPoisson2(r,mu);
    int n = r.PoissonD(mu);
    if (fillHist)
      h.Fill( double(n) );
  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Poisson \t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testPoisson2(r,mu,h);
}

#ifdef HAVE_CLHEP
template<class R>
void testPoissonCLHEP( R & r, double mu,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  //  r.SetSeed(0);
  for (int i = 0; i < n; ++i) {
    //int n = RandPoisson::shoot(mu + RandFlat::shoot());
    int n = static_cast<unsigned int> ( r(mu) ) ;
    if (fillHist)
      h.Fill( double(n) );
  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Poisson - mu = " << mu << "\t\t" << findName(r) <<"\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testPoissonCLHEP(r,mu,h);
}

template<class R>
void testGausCLHEP( R & r,double mu,double sigma,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  int n1 = 100;
  int n2 = n/n1;
  w.Start();
  for (int i = 0; i < n2; ++i) { 
     for (int j = 0; j < n1; ++j) { 
    double x = r(mu,sigma );
    if (fillHist)
      h.Fill( x );

  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Gaussian - mu,sigma = " << mu << " , " << sigma << "\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testGausCLHEP(r,mu,sigma,h);
}

template <class R> 
void testFlatCLHEP( R & r,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  w.Start();
  //r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = r();
    if (fillHist)
      h.Fill( x );

  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Flat - [0,1]           \t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testFlatCLHEP(r,h);
}


#endif



template <class R> 
void testFlat( R & r,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = r.Rndm();
    if (fillHist)
      h.Fill( x );

  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Flat - [0,1]       \t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testFlat(r,h);
}



template <class R> 
void testGaus( R & r,double mu,double sigma,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = r.Gaus(mu,sigma );
    if (fillHist)
      h.Fill( x );

  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Gaussian - mu,sigma = " << mu << " , " << sigma << "\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testGaus(r,mu,sigma,h);
}



template <class R> 
void testLandau( R & r,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = r.Landau();
    if (fillHist)
      h.Fill( x );

  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Landau " << "\t\t\t\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testLandau(r,h);
}




template <class R> 
void testBreitWigner( R & r,double mu,double gamma,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = r.BreitWigner(mu,gamma );
    if (fillHist)
      h.Fill( x );
  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Breit-Wigner - m,g = " << mu << " , " << gamma << "\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testBreitWigner(r,mu,gamma,h);
}



template <class R> 
void testBinomial( R & r,int ntot,double p,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = double( r.Binomial(ntot,p ) );
    if (fillHist)
      h.Fill( x );
  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Binomial - ntot,p = " << ntot << " , " << p << "\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testBinomial(r,ntot,p,h);
}


template<class R> 
void testBinomialCLHEP( R & r,int ntot,double p,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  //r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = double( r(ntot,p ) );
    if (fillHist)
      h.Fill( x );
  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Binomial - ntot,p = " << ntot << " , " << p << "\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testBinomialCLHEP(r,ntot,p,h);
}


template <class R> 
void testMultinomial( R & r,int ntot, TH1D & h1, TH1D & h2) { 

   // generates the p distribution
   const int nbins = h1.GetNbinsX();
   std::vector<double> p(nbins);
   double psum = 0;
   for (int i = 0; i < nbins; ++i) { 
      double x1 = h1.GetBinLowEdge(i+1);
      double x2 = x1 + h1.GetBinWidth(i+1);
      p[i] = ROOT::Math::normal_cdf(x2) -  ROOT::Math::normal_cdf(x1);
      psum += p[i];
   }
   //std::cout << " psum  = " << psum << std::endl;
   // generate the multinomial 
   TStopwatch w; 
   int n = NEVT/10;
   if (fillHist) n = 1;

   for (int i = 0; i < n; ++i) { 
      std::vector<unsigned int> multDist = r.Multinomial(ntot,p);
      if (fillHist) { 
         for (int j = 0; j < nbins; ++j) h1.SetBinContent(j+1,multDist[j]); 
      }
   }
   w.Stop();

   if (!fillHist) {
      std::cout << "Multinomial - nb, ntot = " << nbins << " , " << ntot  << "\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/n << " \t" 
                << w.CpuTime()*1.0E9/n 
                << "\t(ns/call)" << std::endl;   
   }

   // check now time using poisson distribution

   w.Start();
   for (int i = 0; i < n; ++i) { 
      for (int j = 0; j < nbins; ++j) { 
         double y  = r.Poisson(ntot*p[j]);
         if (fillHist) h2.SetBinContent(j+1,y);
      }
   }
   w.Stop();
   if (!fillHist) {
      std::cout << "Multi-Poisson - nb, ntot = " << nbins << " , " << ntot  << "\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/n << " \t" 
                << w.CpuTime()*1.0E9/n 
                << "\t(ns/call)" << std::endl;   
   }

   if (fillHist) { fillHist=false; return; }  

   // fill histogram the second pass
   fillHist = true; 
   testMultinomial(r,ntot,h1,h2);

}


template <class R> 
void testExp( R & r,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  for (int i = 0; i < n; ++i) { 
    double x = r.Exp(1.);
    if (fillHist)
      h.Fill( x );
  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  
  std::cout << "Exponential " << "\t\t\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testExp(r,h);
}


template <class R> 
void testCircle( R & r,TH1D & h) { 

  TStopwatch w; 

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  double x,y; 
  for (int i = 0; i < n; ++i) { 
    r.Circle(x,y,1.0);
    if (fillHist)
      h.Fill( std::atan2(x,y) );

  }
  w.Stop();
  if (fillHist) { fillHist=false; return; }  

  std::cout << "Circle " << "\t\t\t\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   
  // fill histogram the second pass
  fillHist = true; 
  testCircle(r,h);

}


template <class R> 
void testSphere( R & r,TH1D & h1, TH1D & h2 ) { 


#ifdef PLOT_SPHERE
  TH2D hxy("hxy","xy",100,-1.1,1.1,100,-1.1,1.1);
  TH3D h3d("h3d","sphere",100,-1.1,1.1,100,-1.1,1.1,100,-1.1,1.1);
  TH1D hz("hz","z",100,-1.1,1.1);
#endif

  TStopwatch w; 
  

  int n = NEVT;
  // estimate PI
  w.Start();
  r.SetSeed(0);
  double x,y,z; 
  for (int i = 0; i < n; ++i) { 
    r.Sphere(x,y,z,1.0);
    if (fillHist) { 

      h1.Fill( std::atan2(x,y) );
      h2.Fill( std::atan2( std::sqrt(x*x+y*y), z ) );

#ifdef PLOT_SPHERE
      hxy.Fill(x,y);
      hz.Fill(z);
      h3d.Fill(x,y,z);
#endif

    }

  }

  w.Stop();

#ifdef PLOT_SPHERE
  if (fillHist) { 
    TCanvas *c1 = new TCanvas("c1_xyz","sphere",220,20,800,900);
    c1->Divide(2,2);
    c1->cd(1);
    hxy.DrawCopy();
    c1->cd(2);
    hz.DrawCopy();
    c1->cd(3);
    h3d.DrawCopy();
    c1->Update();
  }
#endif  

  if (fillHist) { fillHist=false; return; }  

  std::cout << "Sphere " << "\t\t\t\t"<< findName(r) << "\tTime = " << w.RealTime()*1.0E9/NEVT << " \t" 
	    << w.CpuTime()*1.0E9/NEVT 
	    << "\t(ns/call)" << std::endl;   

  // fill histogram the second pass
  fillHist = true; 
  testSphere(r,h1,h2);

}


int testRandomDist() {

  std::cout << "***************************************************\n"; 
  std::cout << " TEST RANDOM DISTRIBUTIONS   NEVT = " << NEVT << std::endl;
  std::cout << "***************************************************\n\n"; 



  Random<GSLRngMT>         r;
  TRandom3                 tr;
#ifdef HAVE_UNURAN
  UnuRanDist               ur; 
#else 
  TRandom2                 ur;
#endif

  // flat 
  double xmin = 0; 
  double xmax = 1;
  int nch = 1000;
  TH1D hf1("hf1","FLAT ROOT",nch,xmin,xmax);
  TH1D hf2("hf2","Flat GSL",nch,xmin,xmax);

  testFlat(r,hf1);
  testFlat(tr,hf2);
  testDiff(hf1,hf2,"Flat ROOT-GSL");

#ifdef HAVE_CLHEP
  HepJamesRandom eng; 
  RandFlat crf(eng);
  TH1D hf3("hf3","Flat CLHEP",nch,xmin,xmax);
  testFlatCLHEP(crf,hf3);
  testDiff(hf3,hf1,"Flat CLHEP-GSL");
#endif



  // Poisson 
  std::cout << std::endl; 

  double mu = 25; 
  xmin = std::floor(std::max(0.0,mu-5*std::sqrt(mu) ) );
  xmax = std::floor( mu+5*std::sqrt(mu) );
  nch = std::min( int(xmax-xmin),1000);
  TH1D hp1("hp1","Poisson ROOT",nch,xmin,xmax);
  TH1D hp2("hp2","Poisson GSL",nch,xmin,xmax);
  TH1D hp3("hp3","Poisson UNR",nch,xmin,xmax);

  testPoisson(r,mu,hp1);
  testPoisson(tr,mu,hp2);
  testPoisson(ur,mu,hp3);
#ifdef HAVE_CLHEP
  RandPoissonT crp(eng);
  TH1D hp4("hp4","Poisson CLHEP",nch,xmin,xmax);
  testPoissonCLHEP(crp,mu,hp4);
#endif
  //testPoisson2(tr,mu,h2);
  // test differences 
  testDiff(hp1,hp2,"Poisson ROOT-GSL");
  testDiff(hp1,hp3,"Poisson ROOT-UNR");
#ifdef HAVE_CLHEP
  testDiff(hp1,hp4,"Poisson ROOT-CLHEP");
#endif

  // Gaussian
  std::cout << std::endl; 

  TH1D hg1("hg1","Gaussian ROOT",nch,xmin,xmax);
  TH1D hg2("hg2","Gaussian GSL",nch,xmin,xmax);
  TH1D hg3("hg3","Gaussian UNURAN",nch,xmin,xmax);


  testGaus(r,mu,sqrt(mu),hg1);
  testGaus(tr,mu,sqrt(mu),hg2);
 
  testGaus(ur,mu,sqrt(mu),hg3);
#ifdef HAVE_CLHEP
  RandGauss crg(eng);
  TH1D hg4("hg4","Gauss CLHEP",nch,xmin,xmax);
  testGausCLHEP(crg,mu,sqrt(mu),hg4);
  RandGaussQ crgQ(eng);
  testGausCLHEP(crgQ,mu,sqrt(mu),hg4);
  RandGaussT crgT(eng);
  testGausCLHEP(crgT,mu,sqrt(mu),hg4);
#endif


  testDiff(hg1,hg2,"Gaussian ROOT-GSL");
  testDiff(hg1,hg3,"Gaussian ROOT_UNR");

  // Landau
  std::cout << std::endl; 

  TH1D hl1("hl1","Landau ROOT",300,0,50);
  TH1D hl2("hl2","Landau  GSL",300,0,50);

  testLandau(r,hl1);
  testLandau(tr,hl2);
  testDiff(hl1,hl2,"Landau");

  // Breit Wigner
  std::cout << std::endl; 

  TH1D hbw1("hbw1","BreitWigner ROOT",nch,xmin,xmax);
  TH1D hbw2("hbw2","BreitWigner GSL",nch,xmin,xmax);


  testBreitWigner(r,mu,sqrt(mu),hbw1);
  testBreitWigner(tr,mu,sqrt(mu),hbw2);
  testDiff(hbw1,hbw2,"Breit-Wigner");

  // binomial
  std::cout << std::endl; 

  int ntot = 100;
  double p =0.1;
  xmin = 0;
  xmax = ntot+1;
  nch = std::min(1000,ntot+1);
  TH1D hb1("hb1","Binomial ROOT",nch,xmin,xmax);
  TH1D hb2("hb2","Binomial GSL",nch,xmin,xmax);
  TH1D hb3("hb3","Binomial UNR",nch,xmin,xmax);


  testBinomial(r,ntot,p,hb1);
  testBinomial(tr,ntot,p,hb2);
  testBinomial(ur,ntot,p,hb3);
#ifdef HAVE_CLHEP
  RandBinomial crb(eng);
  TH1D hb4("hb4","Binomial CLHEP",nch,xmin,xmax);
  testBinomialCLHEP(crb,ntot,p,hp4);
#endif


  testDiff(hb1,hb2,"Binomial ROOT-GSL");
  testDiff(hb1,hb3,"Binomial ROOT-UNR");

  // multinomial
  std::cout << std::endl; 

  TH1D hmt1("hmt1","Multinomial GSL",30,-3,3);
  TH1D hmt2("hmt2","Multi-Poisson GSL",30,-3,3);
  TH1D hmt3("hmt3","Gaus",30,-3,3);
  ntot = 10000; 
  testMultinomial(r,ntot,hmt1,hmt2);
  hmt3.FillRandom("gaus",ntot);
  testDiff(hmt1,hmt2,"Multinomial-Poisson");
  testDiff(hmt1,hmt3,"Multinomial-gaus");
  

  // exponential
  std::cout << std::endl; 

  TH1D he1("he1","Exp  ROOT",300,0,20);
  TH1D he2("he2","Exp  GSL",300,0,20);

  testExp(r,he1);
  testExp(tr,he2);
  testDiff(he1,he2,"Exponential");

  // circle
  std::cout << std::endl; 

  TH1D hc1("hc1","Circle  ROOT",300,-PI,PI);
  TH1D hc2("hc2","Circle  GSL",300,-PI,PI);

  testCircle(r,hc1);
  testCircle(tr,hc2);
  testDiff(hc1,hc2,"Circle");


  // sphere
  std::cout << std::endl; 

  TH1D hs1("hs1","Sphere-Phi ROOT",300,-PI,PI);
  TH1D hs2("hs2","Sphere-Phi  GSL ",300,-PI,PI);
  TH1D hs3("hs3","Sphere-Theta ROOT",300,0,PI);
  TH1D hs4("hs4","Sphere-Theta  GSL ",300,0,PI);

  testSphere(r,hs1,hs3);
  testSphere(tr,hs2,hs4);
  testDiff(hs1,hs2,"Sphere-phi");
  testDiff(hs3,hs4,"Sphere-theta");


  return 0;

}

int main() { 
  return testRandomDist();
}
