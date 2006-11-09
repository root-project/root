#include "Math/Vector3D.h"
#include "Math/Point3D.h"
#include "TVector3.h"

#include "TStopwatch.h"
#include "TRandom3.h"


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

};


int main(int argc,const char *argv[]) { 

  int ngen = 1000000;
  if (argc > 1)  ngen = atoi(argv[1]);
  int nloop2 = ngen;
  if (argc > 2)  nloop2 = atoi(argv[1]);


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

#ifdef MORETEST
  t1 = 0; 
  t2 = 0; 
  t3 = 0; 
#endif

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

  std::cout << "Total Time for  TVector3        = " << t1 << "\t(sec)" << std::endl;
  std::cout << "Total Time for  XYZVector       = " << t2 << "\t(sec)" << std::endl;
  std::cout << "Total Time for  XYZPoint        = " << t3 << "\t(sec)" << std::endl;

}
