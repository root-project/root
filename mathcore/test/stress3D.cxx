#include "Math/Vector3D.h"
#include "Math/Point3D.h"
#include "TVector3.h"

#include "Math/Transform3D.h"
#include "Math/Rotation3D.h"
#include "Math/Translation3D.h"
#include "Math/RotationZYX.h"
#include "TRotation.h"

#include "TStopwatch.h"
#include "TRandom3.h"

#include <vector>

using namespace ROOT::Math;

class VectorTest { 

private: 

  size_t n2Loop ;
  size_t nGen;

// global data variables 
  std::vector<double> dataX; 
  std::vector<double> dataY;  
  std::vector<double> dataZ;  



public: 
  
  VectorTest(int n1, int n2) : 
    n2Loop(n1),
    nGen(n2),
    dataX(std::vector<double>(n2)), 
    dataY(std::vector<double>(n2)), 
    dataZ(std::vector<double>(n2)) 
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
    
    // fill vectors 
    
      ROOT::Math::RhoEtaPhiVector q( pt, eta, phi); 
      dataX[i] =  q.x(); 
      dataY[i] =  q.y(); 
      dataZ[i] =  q.z(); 

    }
  }



  template <class V> 
  void testCreate( std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
    
    int n = dataX.size(); 
    dataV.resize(n); 
    tim.Start();
    for (int i = 0; i < n; ++i) { 
      dataV[i] = new V( dataX[i], dataY[i], dataZ[i] ); 
    }
    tim.Stop();
    t += tim.RealTime();
    print(tim,s);
  }


template <class V>
double testVectorAddition( const std::vector<V *> & dataV, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = std::min(n2Loop, dataV.size() );
  tim.Start(); 
  V  vSum = *(dataV[0]);
  for (unsigned int i = 1; i < n; ++i) { 
    vSum += *(dataV[i]); 
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return vSum.Mag2(); 
}  

template <class P>
double testPointAddition( const std::vector<P *> & dataP, TStopwatch & tim, double& t,  std::string s) { 
  unsigned int n = std::min(n2Loop, dataP.size() );
  tim.Start(); 
  P  pSum = *(dataP[0]);
  for (unsigned int i = 1; i < n; ++i) { 
     P & p2 = *(dataP[i]); 
#ifndef HEAP_CREATION
     pSum += ROOT::Math::XYZVector(p2);
#else
     ROOT::Math::XYZVector * v2 = new ROOT::Math::XYZVector(p2);
     pSum += *v2;
#endif
  }
  tim.Stop();
  print(tim,s);
  t += tim.RealTime();
  return pSum.Mag2(); 
}

template<class V> 
double getSum(const V & v) { 
   return v.x() + v.y() + v.z();
} 


// direct translation
template <class V> 
  double testTranslation( std::vector<V *> & dataV, const Translation3D & tr, TStopwatch & tim, double& t,  std::string s) { 
    
    unsigned int n = dataV.size();
    tim.Start();
    double sum = 0;
    double dx,dy,dz;
    tr.GetComponents(dx,dy,dz);
    V vtrans(dx,dy,dz);
    for (unsigned int i = 0; i < n; ++i) { 
      V  & v1 = *(dataV[i]);
      V v2 = v1 + vtrans; 
      sum += getSum(v2);
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }

// transformation  
template <class V, class T> 
  double testTransform( std::vector<V *> & dataV, const T & trans, TStopwatch & tim, double& t,  std::string s) { 
    
    unsigned int n = dataV.size();
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) { 
      V  & v1 = *(dataV[i]);
      V v2 = trans * v1; 
      sum += getSum(v2);
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }

// transformation  product 
template <class V, class T1, class T2> 
  double testTransformProd( std::vector<V *> & dataV, const T1 & trans, const T2 &, TStopwatch & tim, double& t,  std::string s) { 
    
    unsigned int n = dataV.size();
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) { 
      V  & v1 = *(dataV[i]);
      V v2 = T2(XYZVector(v1)) * trans * v1; 
      sum += getSum(v2);
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }

// transformation  product 
template <class V, class T1, class T2> 
double testTransformProd2( std::vector<V *> & dataV, const T1 & trans, const T2 &, TStopwatch & tim, double& t,  std::string s) { 
    
    unsigned int n = dataV.size();
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) { 
      V  & v1 = *(dataV[i]);
      V v2 = trans * T2(XYZVector(v1)) * v1; 
      sum += getSum(v2);
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }

// transformation  product 
template <class V, class T1, class T2> 
double testTransformProd3( std::vector<V *> & dataV, const T1 & trans1, const T2 & trans2, TStopwatch & tim, double& t,  std::string s) { 
    
    unsigned int n = dataV.size();
    tim.Start();
    double sum = 0;
    for (unsigned int i = 0; i < n; ++i) { 
      V  & v1 = *(dataV[i]);
      V v2 = trans2 * trans1 * v1; 
      sum += getSum(v2);
    }
    tim.Stop();
    print(tim,s);
    t += tim.RealTime();
    return sum;
  }

};  // end class VectorTest


int main(int argc,const char *argv[]) { 

  int ngen = 1000000;
  if (argc > 1)  ngen = atoi(argv[1]);
  int nloop2 = ngen;
  if (argc > 2)  nloop2 = atoi(argv[1]);

  std::cout << "Test with Ngen = " << ngen << " n2loop = " << nloop2 << std::endl;

  TStopwatch t;

  VectorTest a(ngen,nloop2);

  a.genData(); 


  double t1 = 0;
  double t2 = 0;
  double t3 = 0;

  std::vector<TVector3 *> v1;
  std::vector<ROOT::Math::XYZVector *> v2;
  std::vector<ROOT::Math::XYZPoint *> v3;

  double s1,s2,s3;

  a.genData(); 
  a.testCreate     (v2, t, t2,    "creation XYZVector     " );  

  a.genData(); 
  a.testCreate     (v3, t, t3,    "creation XYZPoint      " ); 

  a.genData(); 
  a.testCreate     (v1, t, t1,    "creation TVector3      " ); 

  std::cout << "\n";


  s1=a.testVectorAddition   (v1, t, t1, "Addition TVector3      " );  
  s2=a.testVectorAddition   (v2, t, t2, "Addition XYZVector     "  ); 
  s3=a.testPointAddition    (v3, t, t3, "Addition XYZPoint      " );       

  a.check("Addition",s1,s2,s3);

#ifdef MORETEST

  s2=a.testVectorAddition   (v2, t, t2, "Addition XYZVector     "  ); 
  s1=a.testVectorAddition   (v1, t, t1, "Addition TVector3      " );  
  s3=a.testPointAddition    (v3, t, t3, "Addition XYZPoint      " );       

  s2=a.testVectorAddition   (v2, t, t2, "Addition XYZVector     "  ); 
  s3=a.testPointAddition    (v3, t, t3, "Addition XYZPoint      " );       
  s1=a.testVectorAddition   (v1, t, t1, "Addition TVector3      " );  

  s1=a.testVectorAddition   (v1, t, t1, "Addition TVector3      " );  
  s3=a.testPointAddition    (v3, t, t3, "Addition XYZPoint      " );       
  s2=a.testVectorAddition   (v2, t, t2, "Addition XYZVector     "  ); 

  s3=a.testPointAddition    (v3, t, t3, "Addition XYZPoint      " );       
  s2=a.testVectorAddition   (v2, t, t2, "Addition XYZVector     "  ); 
  s1=a.testVectorAddition   (v1, t, t1, "Addition TVector3      " );  

  s3=a.testPointAddition    (v3, t, t3, "Addition XYZPoint      " );       
  s1=a.testVectorAddition   (v1, t, t1, "Addition TVector3      " );  
  s2=a.testVectorAddition   (v2, t, t2, "Addition XYZVector     "  ); 

#endif
  std::cout << "\n";

  // test the rotation and transformations
  TRotation r1; r1.RotateZ(3.); r1.RotateY(2.); r1.RotateX(1);
  Rotation3D  r2 (RotationZYX(3., 2., 1.) );
  
  s1=a.testTransform   (v1,  r1, t, t1,  "TRotation  TVector3     " );  
  s2=a.testTransform   (v2,  r2, t, t2,  "Rotation3D XYZVector    " ); 
  s3=a.testTransform   (v3,  r2, t, t3,  "Rotation3D XYZPoint     " ); 

  a.check("Rotation3D",s1,s2,s3);
  double s2a = s2; 
  std::cout << "\n";

  double s4;
  double t0;
  Translation3D tr(1.,2.,3.);
  s1=a.testTranslation (v1,  tr, t, t1,  "Shift         TVector3  " ); 
  s2=a.testTranslation (v2,  tr, t, t2,  "Shift         XYZVector " ); 
  s3=a.testTransform   (v3,  tr, t, t3,  "Translation3D XYZPoint  " ); 
  s4=a.testTransform   (v2,  tr, t, t0,  "Translation3D XYZVector " ); 

  a.check("Translation3D",s1,s2,s3);

  std::cout << "\n";

  Transform3D tf(r2,tr);
  //s1=a.testTransform   (v1,  tf, t, t1,  "Transform3D TVector3    " );  
  s2=a.testTransform   (v2,  tf, t, t0,  "Transform3D XYZVector   " ); 
  s3=a.testTransform   (v3,  tf, t, t0,  "Transform3D XYZPoint    " ); 
  double s2b = s2; 


  std::cout << "\n";
  
  // test product of one vs the other 

  double s5;
  // rotation x translation
  //s1=a.testTransformProd   (v1,  r2, tf, t, t1,  "Delta * Rot  TVector3   " );  
  s2=a.testTransformProd   (v2,  r2, tr, t, t0,  "Delta * Rot  XYZVector  " ); 
  s3=a.testTransformProd   (v3,  r2, tr, t, t0,  "Delta * Rot  XYZPoint   " ); 

  Transform3D tfr(r2);
  s4=a.testTransformProd   (v2,  tfr, tf, t, t0,  "Delta * Rot(T)  XYZVector  " ); 
  s5=a.testTransformProd   (v3,  tfr, tf, t, t0,  "Delta * Rot(T)  XYZPoint   " ); 
  a.check("Delta * Rot",s3,s5,s5);
  // only rot on vectors
  a.check("Trans Vec",s2a,s2b,s2);
  a.check("Trans Vec",s2a,s2,s4);


  std::cout << "\n";

  // translation x rotation
  //s1=a.testTransformProd2   (v1,  r2, tf, t, t1, "Rot * Delta  TVector3   " );  
  s2=a.testTransformProd2   (v2,  r2, tr, t, t0, "Rot * Delta  XYZVector  " ); 
  s3=a.testTransformProd2   (v3,  r2, tr, t, t0, "Rot * Delta  XYZPoint   " ); 

  s4=a.testTransformProd2   (v2,  tfr, tf, t, t0,  "Rot * Delta(T)  XYZVector  " ); 
  s5=a.testTransformProd2   (v3,  tfr, tf, t, t0,  "Rot * Delta(T)  XYZPoint   " ); 

  a.check("Rot * Delta",s3,s5,s5);
  // only rot per vec
  a.check("Trans Vec",s2a,s2,s4);


  std::cout << "\n";
  s2=a.testTransformProd   (v2,  tf, Translation3D(), t, t0,  "Delta * Trans  XYZVector  " ); 
  s3=a.testTransformProd   (v3,  tf, Translation3D(), t, t0,  "Delta * Trans  XYZPoint   " ); 
  s4=a.testTransformProd   (v2,  tf, Transform3D(), t, t0,  "TDelta * Trans  XYZVector  " ); 
  s5=a.testTransformProd   (v3,  tf, Transform3D(), t, t0,  "TDelta * Trans  XYZPoint   " ); 
  a.check("Delta * Trans",s3,s5,s5);
  a.check("Delta * Trans Vec",s2a,s2,s4);

  std::cout << "\n";
  s2=a.testTransformProd2   (v2,  tf, Translation3D(), t, t0,   "Trans * Delta XYZVector  " ); 
  s3=a.testTransformProd2   (v3,  tf, Translation3D(), t, t0,   "Trans * Delta XYZPoint   " ); 
  s4=a.testTransformProd2   (v2,  tf, Transform3D(), t, t0,     "Trans * TDelta XYZVector  " ); 
  s5=a.testTransformProd2   (v3,  tf, Transform3D(), t, t0,     "Trans * TDelta XYZPoint   " ); 
  a.check("Delta * Trans",s3,s5,s5);
  a.check("Delta * Trans Vec",s2a,s2,s4);

  std::cout << "\n";
  s2=a.testTransformProd   (v2,  tf, Translation3D(), t, t0,  "Delta * Trans  XYZVector  " ); 
  s3=a.testTransformProd   (v3,  tf, Translation3D(), t, t0,  "Delta * Trans  XYZPoint   " ); 
  s4=a.testTransformProd   (v2,  tf, Transform3D(), t, t0,  "TDelta * Trans  XYZVector  " ); 
  s5=a.testTransformProd   (v3,  tf, Transform3D(), t, t0,  "TDelta * Trans  XYZPoint   " ); 
  a.check("Delta * Trans",s3,s5,s5);
  a.check("Delta * Trans Vec",s2a,s2,s4);

  std::cout << "\n";
  s2=a.testTransformProd2   (v2,  tf, Translation3D(), t, t0,   "Trans * Delta XYZVector  " ); 
  s3=a.testTransformProd2   (v3,  tf, Translation3D(), t, t0,   "Trans * Delta XYZPoint   " ); 
  s4=a.testTransformProd2   (v2,  tf, Transform3D(), t, t0,     "Trans * TDelta XYZVector  " ); 
  s5=a.testTransformProd2   (v3,  tf, Transform3D(), t, t0,     "Trans * TDelta XYZPoint   " ); 
  a.check("Delta * Trans",s3,s5,s5);
  a.check("Delta * Trans Vec",s2a,s2,s4);

  std::cout << "\n";
  s1=a.testTransformProd3   (v1,  r2, r2, t, t0,  "Rot  * Rot  TVector3     " ); 
  s2=a.testTransformProd3   (v2,  r2, r2, t, t0,  "Rot  * Rot  XYZVector    " ); 
  s3=a.testTransformProd3   (v3,  r2, r2, t, t0,  "Rot  * Rot  XYZPoint     " ); 
  a.check("Rot * Rot",s1,s2,s3);
  s2a = s2; 
 
  std::cout << "\n";

  s2=a.testTransformProd3   (v2,  tf, r2, t, t0,  "Rot   * Trans  XYZVector  " ); 
  s3=a.testTransformProd3   (v3,  tf, r2, t, t0,  "Rot   * Trans  XYZPoint   " ); 
  s4=a.testTransformProd3   (v2,  tf, Transform3D(r2), t, t0,  "TRot * Trans  XYZVector  " ); 
  s5=a.testTransformProd3   (v3,  tf, Transform3D(r2), t, t0,  "TRot * Trans  XYZPoint   " ); 
  a.check("Rot * Trans Pnt",s3,s5,s5);
  a.check("Rot * Trans Vec",s2a,s2,s4);

  std::cout << "\n";

  s2=a.testTransformProd3   (v2,  r2, tf, t, t0,  "Trans * Rot    XYZVector  " ); 
  s3=a.testTransformProd3   (v3,  r2, tf, t, t0,  "Trans * Rot    XYZPoint   " ); 
  s4=a.testTransformProd3   (v2,  Transform3D(r2), tf, t, t0,  "Trans * TRot  XYZVector  " ); 
  s5=a.testTransformProd3   (v3,  Transform3D(r2), tf, t, t0,  "Trans * TRot  XYZPoint   " ); 

  a.check("Rot * Trans Pnt",s3,s5,s5);
  a.check("Rot * Trans Vec",s2a,s2,s4);

  std::cout << "\n";

  std::cout << "Total Time for  TVector3        = " << t1 << "\t(sec)" << std::endl;
  std::cout << "Total Time for  XYZVector       = " << t2 << "\t(sec)" << std::endl;
  std::cout << "Total Time for  XYZPoint        = " << t3 << "\t(sec)" << std::endl;

}
