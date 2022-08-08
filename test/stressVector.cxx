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
#include "TLorentzVector.h"
#include "TRotation.h"
#include "TLorentzRotation.h"

#include "TMatrixD.h"

#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"
#include "Math/LorentzRotation.h"
#include "Math/Rotation3D.h"
#include "Math/RotationX.h"
#include "Math/RotationY.h"
#include "Math/RotationZ.h"

#include "Math/SMatrix.h"


#include "limits"

//#define DEBUG

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


  int check(std::string name, double s1, double s2, double s3, double scale=1) {
    double eps = 10*scale*std::numeric_limits<double>::epsilon();
    if (  std::fabs(s1-s2) < eps*std::fabs(s1) && std::fabs(s1-s3)  < eps*std::fabs(s1) ) return 0;
    std::cout.precision(16);
    std::cout << s1 << "\t" << s2 <<"\t" << s3 << "\n";
    std::cout << "Test " << name << " failed !!\n\n";
    return -1;
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
      if (i % 2) {
         tot += v3.E();
      } else {
         tot -= v3.E();
      }
    }
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot;
}

template <class V>
double testScale( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) {
  unsigned int n = std::min(n2Loop, dataV.size() );
  double tot = 0;
  tim.Start();
  for (unsigned int i = 0; i < n; ++i) {
    V  & v1 = *(dataV[i]);
    // scale
    v1 = 2.0*v1;
    if (i % 2) {
       tot += v1.E();
    } else {
       tot -= v1.E();
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
 return ( pt > 3. && fabs(mass - 1.) < 0.5 && fabs(eta) < 3. );
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
      if (i % 2) {
         tot += delta;
      } else {
         tot -= delta;
      }
    }
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return tot;
}


template <class V>
int testAnalysis( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) {
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
            if ( cutPtEtaAndMass(v3))
            nsel2++;
         }
      }
    }
  }
  tim.Stop();
  print(tim,s);
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


  // rotation using rotation classes
  template <class V, class R>
  double testRotation( std::vector<V *> & dataV, const R & rot, TStopwatch & tim, double& t,  std::string s) {

    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
      V  & v1 = *(dataV[i]);
      V v2 = rot * v1;
      sum += v2.X() + v2.Y() + v2.Z();
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }


  // test matrix vector multiplication
  template <class V, class M>
  double testMatVec( std::vector<V *> & dataV, const M & mat, TStopwatch & tim, double& t,  std::string s) {

    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
      V  & v1 = *(dataV[i]);
      V v2 = VectorUtil::Mult(mat, v1 );
      sum += v2.X() + v2.Y() + v2.Z();
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }


  // Boost using boost  classes
  template <class V, class B>
  double testBoost1( std::vector<V *> & dataV, const B & bv, TStopwatch & tim, double& t,  std::string s) {

    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
      V  & v1 = *(dataV[i]);
      Boost b(bv);
      V v2 = b(v1);
      sum += v2.X() + v2.Y() + v2.Z() + v2.T();
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }


  // Boost using vector util function
  template <class V, class B>
  double testBoost2( std::vector<V *> & dataV, const B & bv, TStopwatch & tim, double& t,  std::string s) {

    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
      V  & v1 = *(dataV[i]);
      V v2 = VectorUtil::boost(v1,bv);
      sum += v2.X() + v2.Y() + v2.Z() + v2.T();
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }

  // Boost using TLorentzVector
  double testBoost_TL( std::vector<TLorentzVector *> & dataV, const TVector3 & bv, TStopwatch & tim, double& t,  std::string s) {

    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
      TLorentzVector  v2 = *(dataV[i]);
      v2.Boost(bv);
      sum += v2.X() + v2.Y() + v2.Z() + v2.T();
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }



  // Boost using boost  classes
  template <class V>
  double testBoostX1( std::vector<V *> & dataV, double beta, TStopwatch & tim, double& t,  std::string s) {

    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
      V  & v1 = *(dataV[i]);
      BoostX b(beta);
      V v2 = b(v1);
      sum += v2.X() + v2.Y() + v2.Z() + v2.T();
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }

  // Boost using vector util function
  template <class V>
  double testBoostX2( std::vector<V *> & dataV, double beta, TStopwatch & tim, double& t,  std::string s) {

    unsigned int n = std::min(n2Loop, dataV.size() );
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) {
      V  & v1 = *(dataV[i]);
      V v2 = VectorUtil::boostX(v1,beta);
      sum += v2.X() + v2.Y() + v2.Z() + v2.T();
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
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

#ifdef DEBUG
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


      a.clear(v3);
      std::cout << "\n";
      a.testConversion  (v2, v3, t, t3,   "Conversion XYZT->PtEtaPhiE " );

      a.clear(v1);
      a.clear(v2);
      a.clear(v3);

      a.testCreate2     (v1, t, t1,   "creationSet TLorentzVector   " );
      a.testCreate2     (v2, t, t2,   "creationSet  XYZTVector      " );
      a.testCreate2     (v3, t, t3,   "creationSet  PtEtaPhiEVector " );


      double s1,s2,s3;
      s1=a.testAddition   (v1, t, t1, "Addition TLorentzVector      " );
      s2=a.testAddition   (v2, t, t2, "Addition XYZTVector          "  );
      s3=a.testAddition   (v3, t, t3, "Addition PtEtaPhiEVector     " );
      a.check("Addition",s1,s2,s3,10);


      s1=a.testScale   (v1, t, t1, "Scale of TLorentzVector      " );
      s2=a.testScale   (v2, t, t2, "Scale of XYZTVector          "  );
      s3=a.testScale   (v3, t, t3, "Scale of PtEtaPhiEVector     " );
      a.check("Scaling",s1,s2,s3);

      s1=a.testDeltaR   (v1, t, t1,      "DeltaR   TLorentzVector      " );
      s2=a.testDeltaR   (v2, t, t2,      "DeltaR   XYZTVector          " );
      s3=a.testDeltaR   (v3, t, t3,      "DeltaR   PtEtaPhiEVector     " );
      a.check("DeltaR",s1,s2,s3,10);


      int n1, n2, n3;
      n1 = a.testAnalysis (v1, t, t1, "Analysis1 TLorentzVector     " );
      n2 = a.testAnalysis (v2, t, t2, "Analysis1 XYZTVector         " );
      n3 = a.testAnalysis (v3, t, t3, "Analysis1 PtEtaPhiEVector    " );
      a.check("Analysis1",n1,n2,n3);



      n1 = a.testAnalysis2 (v1, t, t1, "Analysis2 TLorentzVector    " );
      n2 = a.testAnalysis2 (v2, t, t2, "Analysis2 XYZTVector        " );
      n3 = a.testAnalysis2 (v3, t, t3, "Analysis2 PtEtaPhiEVector   " );
      a.check("Analysis2",n1,n2,n3);


      // test Rotations on Vectors
      TRotation r1; r1.RotateX(1.); r1.RotateY(2.); r1.RotateZ(3);
      Rotation3D  r2 = RotationZ(3)*RotationY(2)*RotationX(1.);
      // need to go through LorentzRotation since ROOT does not have ROtation*4D Vector
      TLorentzRotation lr1(r1);
      LorentzRotation lr2(r2);
      // apply also a boost
      XYZVector bVec(0.4,0.5,0.6); // bVec.R() must be < 1
      assert( bVec.R() <= 1);
      lr1.Boost(bVec.x(), bVec.y(), bVec.z());
      Boost b(bVec);
      LorentzRotation lrb (b);

      lr2 =  lrb * lr2;
#ifdef DEBUG
      // for TLR need to loop
      std::cout << " TLorentzRotation: " << std::endl;
      for (int i = 0; i < 4; ++i) {
         for (int j = 0; j < 4; ++j)
         std::cout << lr1(i,j) << "  ";
         std::cout << "\n";
      }
      std::cout << "\n";
      std::cout << "LorentzRotation :\n"  << lr2 << std::endl;
#endif

      s1=a.testRotation   (v1, lr1, t, t1,      "TRotation  TLorentzVector      " );
      s2=a.testRotation   (v2, lr2, t, t2,      "Rotation3D XYZTVector          " );
      s3=a.testRotation   (v3, lr2, t, t3,      "Rotation3D PtEtaPhiEVector     " );
      a.check("Rotation",s1,s2,s3,10);


      double s0 = s1;
      // test rotations using the matrix for multiplications
      double rotData[16];
      lr2.GetComponents(rotData, rotData+16);
      TMatrixD m1(4,4,rotData);
      SMatrix<double,4,4>  m2(rotData, rotData+16);
#ifdef DEBUG
      m1.Print();
      std::cout << m2 << std::endl;
#endif

      s1=a.testMatVec    (v2, m1, t, t1,        "TMatrix * XYZTVector           " );
      s2=a.testMatVec    (v2, m2, t, t2,        "SMatrix * XYZTVector           " );
      s3=a.testMatVec    (v3, m2, t, t3,        "SMatrix * PtEtaPhiEVector      " );
      a.check("Matrix mult",s1,s2,s3,10);


      // test Boost
      TVector3 tVec(bVec.X(), bVec.Y(), bVec.Z() );
      s1 = a.testBoost_TL (v1, tVec, t, t1,     "Boost TLorentzVector           ");
      s2 = a.testBoost1   (v2, bVec, t, t2,     "Boost XYZTVector               ");
      s3 = a.testBoost1   (v3, bVec, t, t3,     "Boost PtEtaPhiEVector          ");
      a.check("Boost1",s1,s2,s3,10);

      // test Boost (2)
      s0 = s1;
      s1 = a.testBoost2  (v1, tVec, t, t1,      "Boost2 TLorentzVector          ");
      s2 = a.testBoost2  (v2, bVec, t, t2,      "Boost2 XYZTVector              ");
      s3 = a.testBoost2  (v3, bVec, t, t3,      "Boost2 PtEtaPhiEVector         ");
      a.check("Boost1-2",s0,s1,s2);
      a.check("Boost2",s1,s2,s3,10);

      // test BoostX
      double beta = 0.8;
      s1 = a.testBoostX2   (v1, beta, t, t1,    "BoostX TLorentzVector          ");
      s2 = a.testBoostX1   (v2, beta, t, t2,    "BoostX XYZTVector              ");
      s3 = a.testBoostX1   (v3, beta, t, t3,    "BoostX PtEtaPhiEVector         ");
      a.check("BoostX1",s1,s2,s3,10);

      // test Boost (2)
      s0 = s2;
      s1 = a.testBoostX2  (v1, beta, t, t1,     "BoostX2 TLorentzVector         ");
      s2 = a.testBoostX2  (v2, beta, t, t2,     "BoostX2 XYZTVector             ");
      s3 = a.testBoostX2  (v3, beta, t, t3,     "BoostX2 PtEtaPhiEVector        ");
      a.check("BoostX1-2",s0,s1,s2);
      a.check("BoostX2",s1,s2,s3,10);



      // clean all at the end
      a.clear(v1);
      a.clear(v2);
      a.clear(v3);

      std::cout << std::endl;
      std::cout << "Total Time for  TLorentzVector        = " << t1 << "\t(sec)" << std::endl;
      std::cout << "Total Time for  XYZTVector            = " << t2 << "\t(sec)" << std::endl;
      std::cout << "Total Time for  PtRhoPhiEtaVector     = " << t3 << "\t(sec)" << std::endl;
   }

  //tr.dump();

}

