// @(#)root/test:$Id$
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



#include <vector>
#include <iostream>
#include <algorithm>

#include <assert.h>
#include <map>

#include "TStopwatch.h"

#include "TRandom3.h"
#include "TVector2.h"

#include "Math/Vector2D.h"
#include "Math/Point2D.h"
#include "Math/SVector.h"

#include <cmath>

#include "limits"

#ifdef HAVE_CLHEP
#include "CLHEP/Vector/TwoVector.h"
#endif


//#define DEBUG

using namespace ROOT::Math;




class VectorTest { 

private: 

// global data variables 
  std::vector<double> dataX; 
  std::vector<double> dataY;  

  size_t nGen;
  size_t n2Loop ;


public: 
  
  VectorTest(int n1, int n2) : 
    nGen(n1),
    n2Loop(n2)
  {}
    

  

  void print(TStopwatch & time, std::string s) { 
    int pr = std::cout.precision(8);
    std::cout << s << "\t" << " time = " << time.RealTime() << "\t(sec)\t" 
      //    << time.CpuTime() 
	      << std::endl;
    std::cout.precision(pr);
  }

    
  int check(std::string name, double s1, double s2, double s3, double scale=1) {
    double eps = 10*scale*std::numeric_limits<double>::epsilon();
    if (  std::fabs(s1-s2) < eps*std::fabs(s1) && std::fabs(s1-s3)  < eps*std::fabs(s1) ) return 0; 
    int pr = std::cout.precision(16);
    std::cout << s1 << "\t" << s2 <<"\t" << s3 << "\n";
    std::cout << "Rel Diff 1-2:\t" <<  (s1-s2)/std::fabs(s1) << " Diff 1-3:\t" <<  (s1-s3)/std::fabs(s1) << std::endl;
    std::cout << "Test " << name << " failed !!\n\n"; 
    std::cout.precision(pr);
    return -1; 
  }


  void genData() { 
    int n = nGen;

  // generate 2 d data 
    TRandom3 rdm;
    for (int i = 0; i < n ; ++ i) { 
 
      double phi = rdm.Rndm()*3.1415926535897931; 
      double r   = rdm.Exp(10.);
    // fill vectors 
    
      Polar2DVector q( r, phi); 
      dataX.push_back( q.x() ); 
      dataY.push_back( q.y() ); 

    }
  }



  template <class V> 
  void testCreate( std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
    
    int n = dataX.size(); 
    dataV.resize(n); 
    tim.Start();
    for (int i = 0; i < n; ++i) { 
      dataV[i] = new V( dataX[i], dataY[i] ); 
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
      dataV[i]->SetXY(dataX[i], dataY[i] );
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
  }
  void testCreate2( std::vector<TVector2 *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
    
    int n = dataX.size(); 
    dataV.resize(n); 
    tim.Start();
    for (int i = 0; i < n; ++i) { 
      dataV[i] = new TVector2(); 
      dataV[i]->Set(dataX[i], dataY[i] );
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
  }
#ifdef HAVE_CLHEP
  void testCreate2( std::vector<Hep2Vector *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
    
    int n = dataX.size(); 
    dataV.resize(n); 
    tim.Start();
    for (int i = 0; i < n; ++i) { 
      dataV[i] = new Hep2Vector(); 
      dataV[i]->set(dataX[i], dataY[i] );
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
  }
#endif


  template <class V> 
  void clear(  std::vector<V *> & dataV ) { 
    for (unsigned int i = 0; i < dataV.size(); ++i) { 
      V * p = dataV[i]; 
      delete p; 
    }
    dataV.clear(); 
  
}

template<class V> 
inline double addXY(const V & v) { 
   return v.X() + v.Y(); 
}
inline double addXY(const SVector<double,2> & v) { 
   return v(0) + v(1); 
}
template<class V> 
inline double getSum(const V & v1, const V & v2) { 
   return v1.X()+v1.Y() + v2.X() + v2.Y(); 
}

inline double getSum(const SVector<double,2> & v1, const SVector<double,2> & v2 ) { 
   return v1(0)+v1(1) + v2(0)+v2(1);
}

template<class V> 
inline double dotProd(const V & v1, const V & v2) { 
   return v1 * v2; 
}

inline double dotProd(const XYVector & v1, const XYVector & v2) { 
   return v1.Dot(v2); 
}

inline double dotProd(const SVector<double,2> & v1, const SVector<double,2> & v2 ) { 
   return Dot(v1,v2); 
}


#ifdef HAVE_CLHEP
inline double addXY(const Hep2Vector & v) { 
   return v.x() + v.y(); 
}
inline double getSum(const Hep2Vector & v1, const Hep2Vector & v2 ) { 
   return v1.x() + v1.y() + v2.x() + v2.y(); 
}
#endif

template <class V>
double testAddition( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = dataV.size();
  double tot = 0;
  V v0 = *(dataV[0]);
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    V v3 = v1 + v0;
    tot += addXY(v3);
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  

template <class V>
double testAddition2( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = dataV.size();
  double tot = 0;
  V v0 = *(dataV[0]);
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    const V  & v1 = *(dataV[i]); 
    v0 += v1;
    tot += addXY(v0);
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  

template <class V>
double testAddition3( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
   //unsigned int n = std::min(n2Loop, dataV.size() );
  unsigned int n = dataV.size();
  double tot = 0;
  V v0 = *(dataV[0]);
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
//     for (unsigned int j = i +1; j < n; ++j) {
//       V & v2 = *(dataV[j]); 
//       tot += getSum(v1,v2);
//     }
    tot += getSum(v1,v0);
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  


template <class V>
double testDotProduct( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
   //unsigned int n = std::min(n2Loop, dataV.size() );
  double tot = 0;
  unsigned int n = dataV.size();
  V v0 = *(dataV[0]);
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
//     for (unsigned int j = i +1; j < n; ++j) {
//       V & v2 = *(dataV[j]); 
//       tot += dotProd(v1,v2);
//     }
    tot += dotProd(v1,v0);
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  


template <class V>
double testScale( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = dataV.size();
  double tot = 0;
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    // scale
    V v2 = 2.0*v1;
    tot += addXY(v2);
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  

template <class V>
double testScale2( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
   //unsigned int n = std::min(n2Loop, dataV.size() );
   unsigned int n = dataV.size();
  double tot = 0;
  tim.Start(); 
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    // scale
    v1 *= 2.0;
    tot += addXY(v1);
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  

template <class V>
double testOperations( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
   //unsigned int n = std::min(n2Loop, dataV.size() );
   // test operations like in CMS
   unsigned int n = dataV.size();
   double tot = 0;
   V v0a = *(dataV[0]);
   V v0b = *(dataV[n-1]);
   tim.Start(); 
   for (unsigned int i = 0; i < n; ++i) { 
      V  & v1 = *(dataV[i]); 
      //V v2(v1 - dotProd(v1,v0a)*v0b );
      double a =  dotProd(v1,v0a); 
      V v2(v1 - a*v0b );
      tot += addXY(v2);
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot; 
}  


template<class V> 
inline double dPhi(V & v1, V& v2) { 
   return std::abs(v1.Phi() - v2.Phi() ); 
}

#ifdef HAVE_CLHEP
inline double dPhi(Hep2Vector & v1, Hep2Vector & v2) { 
   return std::abs(v1.phi() - v2.phi() );
}
#endif


template <class V>
double testDeltaPhi( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = std::min(n2Loop, dataV.size() );
  tim.Start(); 
  double tot = 0;
  for (unsigned int i = 0; i < n; ++i) { 
    V  & v1 = *(dataV[i]); 
    for (unsigned int j = i +1; j < n; ++j) {
      V & v2 = *(dataV[j]); 
      double delta = dPhi(v1,v2);
      tot += delta;
    }
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot;
}  


// template <class V>
// int testAnalysis( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
//   int nsel = 0;  
//   int nsel2 = 0; 
//   double deltaMax = 1.;
//   double ptMin = 1.;
//   double etaMax = 3.;
  
//   unsigned int n = std::min(n2Loop, dataV.size() );
//   tim.Start(); 
//   for (unsigned int i = 0; i < n; ++i) { 
//     V  & v1 = *(dataV[i]); 
//     if (cutPtEta(v1,ptMin, etaMax) ) { 
//       double delta; 
//       for (unsigned int j = i +1; j < n; ++j) {
// 	V & v2 = *(dataV[j]); 
// 	delta = VectorUtil::DeltaR(v1,v2);
// 	if (delta < deltaMax) { 
// 	  V v3 = v1 + v2; 
// 	  nsel++;
// 	  if ( cutPtEtaAndMass(v3)) 
// 	    nsel2++; 
// 	}
	
//       }
//     }
//   }
//   tim.Stop();
//   print(tim,s);
//   //std::cout << nsel << "\n"; 
//   t += tim.RealTime();
//   return nsel2; 
// }  



// template <class V>
// int testAnalysis2( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
//   int nsel = 0; 
//   double ptMin = 1.;
//   double etaMax = 3.;
//   unsigned int n = std::min(n2Loop, dataV.size() );
//   tim.Start();
//   //seal::SealTimer t(tim.name(), true, std::cout); 
//   for (unsigned int i = 0; i < n; ++i) { 
//     V  & v1 = *(dataV[i]); 
//     if ( cutPtEta(v1, ptMin, etaMax) ) { 
//       for (unsigned int j = i +1; j < n; ++j) {
// 	V & v2 = *(dataV[j]); 
// 	if ( VectorUtil::DeltaR(v1,v2) < 0.5) nsel++;
//       }
//     }
//   }
//   tim.Stop();
//   print(tim,s);
//   t += tim.RealTime();
//   return nsel; 
// }  



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



  // rotation 
  template <class V, class R> 
  double testRotation( std::vector<V *> & dataV, double rotAngle, TStopwatch & tim, double& t,  std::string s) { 
    
    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) { 
      V  & v1 = *(dataV[i]);
      V v2 = v1;
      v2.Rotate(rotAngle);
      sum += addXY(v2);
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }



};



int main(int argc,const char *argv[]) { 

  int ngen = 1000000;
  if (argc > 1)  ngen = atoi(argv[1]);
  int nloop2 = int(std::sqrt(2.0*ngen)+0.5);
  if (argc > 2)  nloop2 = atoi(argv[2]);

  std::cout << "Test with Ngen = " << ngen << " n2loop = " << nloop2 << std::endl;


  TStopwatch t;

  VectorTest a(ngen,nloop2);

  a.genData(); 

  int niter = 1;
  for (int i = 0; i < niter; ++i) { 

#ifdef DEBUG
      std::cout << "iteration " << i << std::endl;
#endif
    
      double t1 = 0;
      double t2 = 0;
      double t3 = 0;
      double t4 = 0;
      double t5 = 0;
      double t6 = 0;

      std::vector<TVector2 *> v1;
      std::vector<XYVector *> v2;
      std::vector<Polar2DVector *> v3;
      std::vector<XYPoint *> v4;
      std::vector<Polar2DPoint *> v5;
      std::vector<SVector<double,2> *> v6;

      a.testCreate     (v1, t, t1,    "creation TVector2          " ); 
      a.testCreate     (v2, t, t2,    "creation XYVector          " ); 
      a.testCreate     (v3, t, t3,    "creation Polar2DVector     " ); 
      a.testCreate     (v4, t, t4,    "creation XYPoint           " ); 
      a.testCreate     (v5, t, t5,    "creation Polar2DPoint      " ); 
      a.testCreate     (v6, t, t6,    "creation SVector<2>        " ); 
#ifdef HAVE_CLHEP
      double t7 = 0;
      std::vector<Hep2Vector *> v7;
      a.testCreate     (v7, t, t7,        "creation Hep2Vector        " ); 
#endif

      
      a.clear(v3);
      a.clear(v4);
      a.clear(v5);

#ifdef HAVE_CLHEP
      a.clear(v7);
#endif

      std::cout << "\n";
      a.testConversion  (v2, v3, t, t3,   "Conversion XY->Polar      " ); 
      a.testConversion  (v2, v4, t, t4,   "Conversion XYVec->XYPoint " ); 
      a.testConversion  (v2, v5, t, t5,   "Conversion XYVec->PolarP  " ); 

      a.clear(v1);
      a.clear(v2);
      a.clear(v3); 
      a.clear(v4);
      a.clear(v5);
      std::cout << "\n";

      a.testCreate2     (v1, t, t1,   "creationSet TVector2       " ); 
      a.testCreate2     (v2, t, t2,   "creationSet  XYVector      " ); 
      a.testCreate2     (v3, t, t3,   "creationSet  Polar2DVector " ); 
      a.testCreate2     (v4, t, t4,   "creationSet  XYPoint       " ); 
      a.testCreate2     (v5, t, t5,   "creationSet  Polar2DPoint  " ); 
//      a.testCreate2     (v6, t, t6,   "creationSet  Polar2DPoint  " ); 
#ifdef HAVE_CLHEP
      a.testCreate2    (v7, t, t7,    "creationSet Hep2Vector     " ); 
#endif

      std::cout << "\n";

      double s1,s2,s3,s4,s5,s6;
      s1=a.testAddition   (v1, t, t1, "Addition TVector2          " );  
      s2=a.testAddition   (v2, t, t2, "Addition XYVector          "  ); 
      s3=a.testAddition   (v3, t, t3, "Addition Polar2DVector     " );       
      s6=a.testAddition   (v6, t, t6, "Addition SVector<2>        " );       
      a.check("Addition test1",s1,s2,s3);
      a.check("Addition test2",s1,s2,s6);
#ifdef HAVE_CLHEP
      double s7; 
      s7=a.testAddition   (v7, t, t7, "Addition Hep2Vector        " );  
      a.check("Addition",s7,s1,s2);
#endif

      double s0 = s2;
      std::cout << "\n";

      s1=a.testAddition2   (v1, t, t1, "Addition2 TVector2         " );  
      s2=a.testAddition2   (v2, t, t2, "Addition2 XYVector         "  ); 
      s3=a.testAddition2   (v3, t, t3, "Addition2 Polar2DVector    " );       
      s6=a.testAddition2   (v6, t, t6, "Addition2 SVector<2>       " );       
      a.check("Addition2 test1",s1,s2,s3,100);
      a.check("Addition2 test2",s1,s2,s6);
#ifdef HAVE_CLHEP
      s7=a.testAddition2  (v7, t, t7,  "Addition2 Hep2Vector       " );  
      a.check("Addition2 CLHEP",s7,s1,s2);
#endif

      std::cout << "\n";

      s1=a.testAddition3   (v1, t, t1, "Addition3 TVector2         " );  
      s2=a.testAddition3   (v2, t, t2, "Addition3 XYVector         "  ); 
      s3=a.testAddition3   (v3, t, t3, "Addition3 Polar2DVector    " );       
      s6=a.testAddition3   (v6, t, t6, "Addition3 SVector<2>       " );       
      a.check("Addition3 test1",s1,s2,s3);
      a.check("Addition3 test2",s6,s0,s2);
#ifdef HAVE_CLHEP
      s7=a.testAddition3  (v7, t, t7,  "Addition3 Hep2Vector       " );  
      a.check("Addition3 CLHEP",s7,s1,s2);
#endif

      std::cout << "\n";

      s1=a.testDotProduct   (v1, t, t1, "DotProduct TVector2        " );  
      s2=a.testDotProduct   (v2, t, t2, "DotProduct XYVector        "  ); 
//      s3=a.testDotProduct   (v3, t, t3, "DotProduct Polar2DVector   " );       
      s6=a.testDotProduct   (v6, t, t6, "DotProduct SVector<2>      " );       
      a.check("DotProduct test1",s1,s2,s6);
//      a.check("DotProduct test2",s6,s1,s2);
#ifdef HAVE_CLHEP
      s7=a.testDotProduct   (v7, t, t7, "DotProduct Hep2Vector      " );  
      a.check("DotProduct CLHEP",s7,s1,s2);
#endif


      std::cout << "\n";

      s1=a.testDeltaPhi   (v1, t, t1,  "DeltaPhi   TVector2        " );  
      s2=a.testDeltaPhi   (v2, t, t2,  "DeltaPhi   XYVector        " ); 
      s3=a.testDeltaPhi   (v3, t, t3,  "DeltaPhi   Polar2DVector   " ); 
      s4=a.testDeltaPhi   (v4, t, t4,  "DeltaPhi   XYPoint         "  );  
      s5=a.testDeltaPhi   (v5, t, t5,  "DeltaPhi   Polar2DPoint    " ); 
#ifdef WIN32
      //windows is bad here 
      a.check("DeltaPhi",s1,s2,s3,10);      
      a.check("DeltaPhi",s2,s4,s5,10);      
#else
      a.check("DeltaPhi",s1,s2,s3);      
      a.check("DeltaPhi",s2,s4,s5);      
#endif
#ifdef HAVE_CLHEP
      s7=a.testDeltaPhi   (v7, t, t7,  "DeltaPhi   HEP2Vector      " );  
      a.check("DeltaPhi",s7,s1,s2);      
#endif 

      std::cout << "\n";
      s1=a.testScale   (v1, t, t1, "Scale of TVector2          " );  
      s2=a.testScale   (v2, t, t2, "Scale of XYVector          "  ); 
      s3=a.testScale   (v3, t, t3, "Scale of Polar2DVector     " ); 
      s4=a.testScale   (v4, t, t4, "Scale of XYPoint           "  ); 
      s5=a.testScale   (v5, t, t5, "Scale of Polar2DPoint      " ); 
      a.check("Scaling",s1,s2,s3);
      a.check("Scaling",s2,s4,s5, 10);
      s6=a.testScale   (v6, t, t6, "Scale of SVector<2>        " );       
      a.check("Scaling SV",s6,s1,s2);

#ifdef HAVE_CLHEP
      s7=a.testScale   (v7, t, t7, "Scale of HEP2Vector        " );  
      a.check("Scaling CLHEP",s7,s2,s3);
#endif 

      std::cout << "\n";
      s1=a.testScale2   (v1, t, t1, "Scale2 of TVector2          " );  
      s2=a.testScale2   (v2, t, t2, "Scale2 of XYVector          "  ); 
      s3=a.testScale2   (v3, t, t3, "Scale2 of Polar2DVector     " ); 
      s4=a.testScale2   (v4, t, t4, "Scale2 of XYPoint           "  ); 
      s5=a.testScale2   (v5, t, t5, "Scale2 of Polar2DPoint      " ); 
      a.check("Scaling2",s1,s2,s3);
      a.check("Scaling2",s2,s4,s5, 10);
      s6=a.testScale2   (v6, t, t6, "Scale2 of SVector<2>        " );       
      a.check("Scaling2 SV",s6,s1,s2);

#ifdef HAVE_CLHEP
      s7=a.testScale2   (v7, t, t7, "Scale2 of HEP2Vector        " );  
      a.check("Scaling CLHEP",s7,s2,s3);
#endif 

      std::cout << "\n";

      s1=a.testOperations   (v1, t, t1, "Operations of TVector2      " );  
      s2=a.testOperations   (v2, t, t2, "Operations of XYVector      "  ); 
      s6=a.testOperations   (v6, t, t6, "Operations of SVector<2>    " );  
      a.check("Operations testSV",s6,s1,s2);
#ifdef HAVE_CLHEP
      s7=a.testOperations   (v7, t, t7, "Operations of HEP2Vector    " );  
      a.check("Operations CLHEP",s7,s1,s2);
#endif 



#ifdef LATER

      int n1, n2, n3,n4,n5; 
      n1 = a.testAnalysis (v1, t, t1, "Analysis1 TVector2         " ); 
      n2 = a.testAnalysis (v2, t, t2, "Analysis1 XYVector         " ); 
      n3 = a.testAnalysis (v3, t, t3, "Analysis1 Polar2DVector    " ); 
      n4 = a.testAnalysis (v4, t, t4, "Analysis1 XYPoint          "  );  
      n5 = a.testAnalysis (v5, t, t5, "Analysis1 Polar2DPoint     " ); 
      a.check("Analysis1",n1,n2,n3);      
      a.check("Analysis1",n2,n4,n5);      
#ifdef HAVE_CLHEP
      int n6;
      n6 = a.testAnalysis (v7, t, t7, "Analysis1 HEP2Vector       " ); 
      a.check("Analysis2 CLHEP",n6,n1,n2);      
#endif 


      n1 = a.testAnalysis2 (v1, t, t1, "Analysis2 TVector2        " ); 
      n2 = a.testAnalysis2 (v2, t, t2, "Analysis2 XYVector        " ); 
      n3 = a.testAnalysis2 (v3, t, t3, "Analysis2 Polar2DVector   " ); 
      n4 = a.testAnalysis2 (v4, t, t4, "Analysis2 XYPoint         "  );  
      n5 = a.testAnalysis2 (v5, t, t5, "Analysis2 Polar2DPoint    " ); 
      a.check("Analysis2",n1,n2,n3);      
      a.check("Analysis2",n2,n4,n5);      
#ifdef HAVE_CLHEP
      n6 = a.testAnalysis2 (v7, t, t7, "Analysis2 HEP2Vector      " ); 
      a.check("Analysis2 CLHEP",n6,n1,n2);      
#endif 


      n1 = a.testAnalysis3 (v1, t, t1, "Analysis3 TVector2    " ); 
      n2 = a.testAnalysis3 (v2, t, t2, "Analysis3 XYVector        " ); 
      n3 = a.testAnalysis3 (v3, t, t3, "Analysis3 Polar2DVector   " ); 
      n4 = a.testAnalysis3 (v4, t, t4, "Analysis3 XYPoint   "  );  
      n5 = a.testAnalysis3 (v5, t, t5, "Analysis3 Polar2DPoint     " ); 
      a.check("Analysis3",n1,n2,n3);      
      a.check("Analysis3",n2,n4,n5);      
#ifdef HAVE_CLHEP
      n6 = a.testAnalysis3 (v7, t, t7,"Analysis3 HEP2Vector        " ); 
      a.check("Analysis3 CLHEP",n6,n1,n2);      
#endif 

#endif


      // clean all at the end
      a.clear(v1); 
      a.clear(v2);
      a.clear(v3);

      std::cout << std::endl;
      std::cout << "Total Time for  TVector2        = " << t1 << "\t(sec)" << std::endl;
      std::cout << "Total Time for  XYVector        = " << t2 << "\t(sec)" << std::endl;
      std::cout << "Total Time for  Polar2DVector   = " << t3 << "\t(sec)" << std::endl;
#ifdef HAVE_CLHEP
      std::cout << "Total Time for  Hep2Vector      = " << t7 << "\t(sec)" << std::endl;
#endif 
   }

  //tr.dump(); 

}



