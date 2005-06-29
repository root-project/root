// @(#)root/test:$Name:  $:$Id: StressVector.cxx,v 1.1 2005/06/28 18:54:24 brun Exp $
// Author: Lorenzo Moneta   06/2005 
///////////////////////////////////////////////////////////////////////////////////
//
//  MathCore Benchmark test suite
//  ==============================
//
//  This program performs tests of ROOT::Math 4D LorentzVectors comparing with TLorentzVector
//  The time performing various vector operations on a collection of vectors is measured. 
//  The benchmarked operations are: 
//      - vector construction from 4 values
//      - construction using a setter method
//      - simple addition of all the vector pairs in the collection 
//      - calculation of deltaR = phi**2 + eta**2 of all vector pairs in the collection            
//      -  two simple analysis: 
//         - the first requires some cut (on pt and eta) and on the invariant mass 
//           of the selected pairs
//         - the second requires just some cut in pt, eta and delta R on all the 
//           vector pair
//      - conversion between XYZTVectors to PtRhoEtaPhi based vectors
//
//   The two analysis demostrates, especially in the second case, the advantage of using 
//   vector based on cylindrical coordinate, given the fact that the time spent in the conversion is 
//   much less than the time spent in the analysis routine. 
//
//   To run the program do: 
//   stressVector          : run standard test with collection of 1000 vectors
//   stressVector  10000   : run with a collection of 10000 vectors
//
///////////////////////////////////////////////////////////////////////////////////



#include <iostream>
#include <algorithm>

#include <assert.h>
#include <map>

#include "TStopwatch.h"


#include "TRandom3.h"
#include "TLorentzVector.h"


#include "MathCore/Vector4D.h"
#include "MathCore/VectorUtil.h"

#include "limits"



using namespace ROOT::Math;


class VectorTest { 

private: 

// global data variables 
  std::vector<double> dataX; 
  std::vector<double> dataY;  
  std::vector<double> dataZ;  
  std::vector<double> dataE; 

  size_t n2Loop ;
  size_t nGen;


public: 
  
  VectorTest(int n1, int n2) : 
    n2Loop(n1),
    nGen(n2)
  {}
    

  

  void print(TStopwatch & time, std::string s) { 
    int pr = std::cout.precision(8);
    std::cout << s << "\t" << " time = " << time.RealTime() << "\t(sec)\t" 
      //    << time.CpuTime() 
	      << std::endl;
    std::cout.precision(pr);
  }

  void genData() { 
    int n = nGen;

  // generate n -4 momentum quantities 
    TRandom3 r;
    for (int i = 0; i < n ; ++ i) { 
 
      double phi = r.Rndm()*3.1415926535897931; 
      double eta = r.Uniform(-5.,5.); 
      double pt = r.Exp(10.);
      double m = r.Uniform(0,10.); 
      if ( i%50 == 0 ) 
	m = r.BreitWigner(1.,0.01); 

      double E = sqrt( m*m + pt*pt*cosh(eta)*cosh(eta) );
    
    // fill vectors 
    
      PtEtaPhiEVector q( pt, eta, phi, E); 
      dataX.push_back( q.x() ); 
      dataY.push_back( q.y() ); 
      dataZ.push_back( q.z() ); 
      dataE.push_back( q.t() ); 

    }
  }



  template <class V> 
  void testCreate( std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
    
    int n = dataX.size(); 
    dataV.resize(n); 
    tim.Start();
    for (int i = 0; i < n; ++i) { 
      dataV[i] = new V( dataX[i], dataY[i], dataZ[i], dataE[i] ); 
    }
    tim.Stop();
    t += tim.RealTime();
    print(tim,s);
  }


  template <class V> 
  void testCreate2( std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
    
    int n = dataX.size(); 
    dataV.resize(n); 
    tim.Start();
    for (int i = 0; i < n; ++i) { 
      dataV[i] = new V(); 
      dataV[i]->SetXYZT(dataX[i], dataY[i], dataZ[i], dataE[i] );
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
  }



  template <class V> 
  void clear(  std::vector<V *> & dataV ) { 
    for (unsigned int i = 0; i < dataV.size(); ++i) { 
      V * p = dataV[i]; 
      delete p; 
    }
    dataV.clear(); 
  }

template <class V>
double testAddition( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = std::min(n2Loop, dataV.size() );
  double tot = 0;
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    for (unsigned int j = i +1; j < n; ++j) {
      V & v2 = *(dataV[j]); 
      V v3 = v1 + v2;
      tot += v3.E();
    }
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  



template< class V> 
bool cutPtEtaAndMass(const V & v) { 
 double pt = v.Pt();
 double mass = v.M();
 double eta = v.Eta();
 return ( pt > 5. && fabs(mass - 1.) < 0.2 && fabs(eta) < 3. );
}


template< class V> 
bool cutPtEta(const V & v,double ptMin, double etaMax) { 
 double pt = v.Pt();
 double eta = v.Eta();
 return ( pt > ptMin && fabs(eta) < etaMax );
}



template <class V>
double testDeltaR( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = std::min(n2Loop, dataV.size() );
  tim.Start(); 
  double tot = 0;
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    for (unsigned int j = i +1; j < n; ++j) {
      V & v2 = *(dataV[j]); 
      double delta = VectorUtil::DeltaR(v1,v2);
      tot += delta;
    }
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot;
}  


template <class V>
int testAnalysis( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  int nsel = 0;  
  int nsel2 = 0; 
  double deltaMax = 1.;
  double ptMin = 1.;
  double etaMax = 3.;
  
  unsigned int n = std::min(n2Loop, dataV.size() );
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    if (cutPtEta(v1,ptMin, etaMax) ) { 
      double delta; 
      for (unsigned int j = i +1; j < n; ++j) {
	V & v2 = *(dataV[j]); 
	delta = VectorUtil::DeltaR(v1,v2);
	if (delta < deltaMax) { 
	  V v3 = v1 + v2; 
	  nsel++;
	  if ( cutPtEtaAndMass(v3)) 
	    nsel2++; 
	}
	
      }
    }
  }
  tim.Stop();
  print(tim,s);
  //std::cout << nsel << "\n"; 
  t += tim.RealTime();
  return nsel2; 
}  



template <class V>
int testAnalysis2( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  int nsel = 0; 
  double ptMin = 1.;
  double etaMax = 3.;
  unsigned int n = std::min(n2Loop, dataV.size() );
  tim.Start();
  //seal::SealTimer t(tim.name(), true, std::cout); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    if ( cutPtEta(v1, ptMin, etaMax) ) { 
      for (unsigned int j = i +1; j < n; ++j) {
	V & v2 = *(dataV[j]); 
	if ( VectorUtil::DeltaR(v1,v2) < 0.5) nsel++;
      }
    }
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return nsel; 
}  



  template <class V1, class V2> 
  void testConversion( std::vector<V1 *> & dataV1, std::vector<V2 *> & dataV2, TStopwatch & tim, double& t,  std::string s) { 
    
    int n = dataX.size(); 
    dataV2.resize(n); 
    tim.Start();
    for (int i = 0; i < n; ++i) { 
      dataV2[i] = new V2( *dataV1[i] ); 
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
  }



};

int main(int argc,const char *argv[]) { 

  int ngen = 1000;
  if (argc > 1)  ngen = atoi(argv[1]);
  int nloop2 = ngen;
  if (argc > 2)  nloop2 = atoi(argv[1]);


  TStopwatch t;

  VectorTest a(ngen,nloop2);

  a.genData(); 

  int niter = 1;
  for (int i = 0; i < niter; ++i) { 

#ifdef DEBUUG
      std::cout << "iteration " << i << std::endl;
#endif
    
      double t1 = 0;
      double t2 = 0;
      double t3 = 0;

      std::vector<TLorentzVector *> v1;
      std::vector<XYZTVector *> v2;
      std::vector<PtEtaPhiEVector *> v3;

      a.testCreate     (v1, t, t1,    "creation TLorentzVector      " ); 
      a.testCreate     (v2, t, t2,    "creation XYZTVector          " ); 
      a.testCreate     (v3, t, t3,     "creation PtEtaPhiEVector     " ); 

      a.clear(v1);
      a.clear(v2);
      a.clear(v3); 

      a.testCreate2     (v1, t, t1,   "creationSet TLorentzVector   " ); 
      a.testCreate2     (v2, t, t2,   "creationSet  XYZTVector      " ); 
      a.testCreate2     (v3, t, t3,   "creationSet  PtEtaPhiEVector " ); 


      double s1,s2,s3, eps;
      s1=a.testAddition   (v1, t, t1, "Addition TLorentzVector      " );  
      s2=a.testAddition   (v2, t, t2, "Addition XYZTVector          "  ); 
      s3=a.testAddition   (v3, t, t3, "Addition PtEtaPhiEVector     " ); 
      
      eps = 10*s1*std::numeric_limits<double>::epsilon();
#ifdef DEBUG
      std::cout.precision(16);
      std::cout << s1 << "\t" << s2 <<"\t" << s3 << "\n";
#else
      assert( std::fabs(s1-s2) < eps &&  std::fabs(s1-s3)  < eps );
#endif

      s1=a.testDeltaR   (v1, t, t1,      "DeltaR   TLorentzVector      " );  
      s2=a.testDeltaR   (v2, t, t2,      "DeltaR   XYZTVector          " ); 
      s3=a.testDeltaR   (v3, t, t3,      "DeltaR   PtEtaPhiEVector     " ); 

      eps = 10*s1*std::numeric_limits<double>::epsilon();
#ifdef DEBUG
      std::cout.precision(16);
      std::cout << s1 << "\t" << s2 <<"\t" << s3 << "\n";
#else
      assert( std::fabs(s1-s2) < eps &&  std::fabs(s1-s3)  < eps );
#endif

      int n1, n2, n3; 
      n1 = a.testAnalysis (v1, t, t1, "Analysis1 TLorentzVector     " ); 
      n2 = a.testAnalysis (v2, t, t2, "Analysis1 XYZTVector         " ); 
      n3 = a.testAnalysis (v3, t, t3, "Analysis1 PtEtaPhiEVector    " ); 

#ifdef DEBUG
      std::cout << "test Analysis1 - nsel= "  << n1 << "  " << n2 << "  " << n3 << std::endl;
#else
      assert(n1 == n2 && n1 == n3); 
#endif


      n1 = a.testAnalysis2 (v1, t, t1, "Analysis2 TLorentzVector    " ); 
      n2 = a.testAnalysis2 (v2, t, t2, "Analysis2 XYZTVector        " ); 
      n3 = a.testAnalysis2 (v3, t, t3, "Analysis2 PtEtaPhiEVector   " ); 
#ifdef DEBUG
      std::cout << "test Analsys-2 - nsel=" << n1 << "  " << n2 << "  " << n3 << std::endl;
#else
      assert(n1 == n2 && n1 == n3); 
#endif
      //std::cout << n1 << " " << n2 << "  " << n2 << std::endl;

      
      a.clear(v3);
      std::cout << "\n";
      a.testConversion  (v2, v3, t, t3,   "Conversion XYZT-> PtEtaPhiEVector " ); 


      // clean all
      a.clear(v1); 
      a.clear(v2);
      a.clear(v3);

      std::cout << std::endl;
      std::cout << "Total Real Time for  TLorentzVector                = " << t1 << " sec" << std::endl;
      std::cout << "Total Real Time for  ROOT::Math::XYZTVector        = " << t2 << " sec" << std::endl;
      std::cout << "Total Real Time for  ROOT::Math::PtRhoPhiEtaVector = " << t3 << " sec" << std::endl;
   }

  //tr.dump(); 

}

