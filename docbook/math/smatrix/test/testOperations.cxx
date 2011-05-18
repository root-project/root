#define  ENABLE_TEMPORARIES

#include "Math/SVector.h"
#include "Math/SMatrix.h"


#include "TMatrixD.h"
#include "TVectorD.h"

#include "TRandom3.h"
#include "TH1D.h" 
#include "TProfile.h" 
#include "TFile.h" 

//#define HAVE_CLHEP
#define TEST_SYM

#ifdef TEST_ALL_MATRIX_SIZES
#define REPORT_TIME
#endif
#ifndef NITER
#define NITER 1  // number of iterations
#endif
#ifndef NLOOP_MIN
#define NLOOP_MIN 1000;
#endif

#ifdef HAVE_CLHEP
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#endif

//#define DEBUG

#include <iostream>

// #ifndef NDIM1
// #define NDIM1 5
// #endif
// #ifndef NDIM2
// #define NDIM2 5
// #endif


int NLOOP; 
//#define NLOOP 1

//#define DEBUG


#include "matrix_op.h"
#include "matrix_util.h"
#include <map>





template<unsigned int NDIM1, unsigned int NDIM2> 
int test_smatrix_op() {
    
  // need to write explicitly the dimensions
   

  typedef SMatrix<double, NDIM1, NDIM1> MnMatrixNN;
  typedef SMatrix<double, NDIM2, NDIM2> MnMatrixMM;
  typedef SMatrix<double, NDIM1, NDIM2> MnMatrixNM;
  typedef SMatrix<double, NDIM2 , NDIM1> MnMatrixMN;
  typedef SVector<double, NDIM1> MnVectorN;
  typedef SVector<double, NDIM2> MnVectorM;
  


  int first = NDIM1;  //Can change the size of the matrices
  int second = NDIM2;
  

  std::cout << "************************************************\n";
  std::cout << "  SMatrix operations test  "   <<  first << " x " << second  << std::endl;
  std::cout << "************************************************\n";


  double t_veq, t_meq, t_vad, t_mad, t_dot, t_mv, t_gmv, t_mm, t_prd, t_inv, t_vsc, t_msc, t_ama, t_tra = 0;
  double totTime1, totTime2; 
  
  
   
  double r1,r2;
  int npass = NITER; 
  TRandom3 r(111);
  for (int k = 0; k < npass; k++) {


    MnMatrixNM A;
    MnMatrixMN B;
    MnMatrixNN C; 
    MnMatrixMM D; 
    MnVectorN v;
    MnVectorM p;


    TStopwatch w; 
    {       
      //seal::SealTimer t(init,false);
      // fill matrices with random data
      fillRandomMat(r,A,first,second);
      fillRandomMat(r,B,second,first);
      fillRandomMat(r,C,first,first);
      fillRandomMat(r,D,second,second);

      fillRandomVec(r,v,first);
      fillRandomVec(r,p,second);

    }


//     MnSymMatrixMM I; 
//     for(int i = 0; i < second; i++) 
//       I(i,i) = 1;
     
#ifdef DEBUG
    std::cout << "pass " << k << std::endl;
    if (k == 0) { 
      std::cout << " A = " << A << std::endl;
      std::cout << " B = " << B << std::endl;
      std::cout << " C = " << C << std::endl;
      std::cout << " D = " << D << std::endl;
      std::cout << " v = " << v << std::endl;
      std::cout << " p = " << p << std::endl;
    }
#endif

    w.Start(); 



	        
    MnVectorN   v1;  testMV(A,v,t_mv,v1);
    //if (k == 0) v1.Print(std::cout);
    MnVectorN   v2;  testGMV(A,v,v1,t_gmv,v2);
    //if (k == 0) v2.Print(std::cout);
    MnMatrixNN  C0;  testMM(A,B,C,t_mm,C0);
    //if (k == 0) C0.Print(std::cout);
    MnMatrixNN  C1;  testATBA_S(B,C0,t_ama,C1);
    //if (k == 0) C1.Print(std::cout);
    MnMatrixNN  C2;  testInv_S(C1,t_inv,C2);
    MnVectorN   v3;  testVeq(v,t_veq,v3);
    MnVectorN   v4;  testVad(v2,v3,t_vad,v4);
    MnVectorN   v5;  testVscale(v4,2.0,t_vsc,v5);
    MnMatrixNN  C3;  testMeq(C,t_meq,C3);
    MnMatrixNN  C4;  testMad(C2,C3,t_mad,C4);
    MnMatrixNN  C5;  testMscale(C4,0.5,t_msc,C5);
    MnMatrixNN  C6;  testMT_S(C5,t_tra,C6);

#ifdef DEBUG
    if (k == 0) { 
      std::cout << " C6 = " << C5 << std::endl;
      std::cout << " v5 = " << v5 << std::endl;
    }
#endif

    r1 = testDot_S(v3,v5,t_dot);

    r2 = testInnerProd_S(C6,v5,t_prd);

  
    w.Stop();
    totTime1 = w.RealTime();
    totTime2 = w.CpuTime();
  
  }
  //tr.dump();

  //double totTime = t_veq + t_meq + t_vad + t_mad + t_dot + t_mv + t_gmv + t_mm + t_prd + t_inv + t_vsc + t_msc + t_ama + t_tra; 
  std::cout << "Total Time = " << totTime1 << "  (s) " << " cpu " <<  totTime2 << "  (s) " << std::endl; 
  std::cerr << "SMatrix:     r1 = " << r1 << " r2 = " << r2 << std::endl; 

  return 0;
}



#ifdef TEST_SYM
template<unsigned int NDIM1, unsigned int NDIM2> 
int test_smatrix_sym_op() {
    
  // need to write explicitly the dimensions
   

  typedef SMatrix<double, NDIM1, NDIM1, MatRepSym<double,NDIM1> > MnSymMatrixNN;
  typedef SMatrix<double, NDIM1, NDIM1 > MnMatrixNN;
  typedef SVector<double, NDIM1> MnVectorN;
  


  int first = NDIM1;  //Can change the size of the matrices
  

  std::cout << "************************************************\n";
  std::cout << "  SMatrixSym operations test  "   <<  first << " x " << first << std::endl;
  std::cout << "************************************************\n";


  double t_meq, t_mad, t_mv, t_gmv, t_mm, t_prd, t_inv, t_msc, t_ama = 0;
  double totTime1, totTime2; 
  
  
   
  double r1;
  int npass = NITER; 
  TRandom3 r(111);
  for (int k = 0; k < npass; k++) {


    MnSymMatrixNN A;
    MnSymMatrixNN B;
    MnMatrixNN C;
    MnVectorN v;


    TStopwatch w; 
    {       
      // fill matrices with random data
      fillRandomSym(r,A,first);
      fillRandomSym(r,B,first);
      fillRandomMat(r,C,first,first);

      fillRandomVec(r,v,first);

    }

     
#ifdef DEBUG
    std::cout << "pass " << k << std::endl;
    if (k == 0) { 
      std::cout << " A = " << A << std::endl;
      std::cout << " B = " << B << std::endl;
      std::cout << " C = " << C << std::endl;
      std::cout << " v = " << v << std::endl;
    }
#endif

    w.Start(); 
	        
    MnVectorN   v1;  testMV(A,v,t_mv,v1);
    MnVectorN   v2;  testGMV(A,v,v1,t_gmv,v2);
    MnMatrixNN  C0;  testMM(A,B,C,t_mm,C0);
    MnSymMatrixNN  C1;  testATBA_S2(C0,B,t_ama,C1);
    MnSymMatrixNN  C2;  testInv_S(A,t_inv,C2);
    MnSymMatrixNN  C3;  testMeq(C2,t_meq,C3);
    MnSymMatrixNN  C4;  testMad(A,C3,t_mad,C4);
    MnSymMatrixNN  C5;  testMscale(C4,0.5,t_msc,C5);

    r1 = testInnerProd_S(C5,v2,t_prd);

     
#ifdef DEBUG
    std::cout << "output matrices" << std::endl;
    if (k == 0) { 
      std::cout << " C1 = " << C1 << std::endl;
      std::cout << " C3 = " << C3 << std::endl;
      std::cout << " C4 = " << C4 << std::endl;
      std::cout << " C5 = " << C5 << std::endl;
    }
#endif

  
    w.Stop();
    totTime1 = w.RealTime();
    totTime2 = w.CpuTime();

  
  }
  //tr.dump();

  //double totTime = t_meq + t_mv + t_gmv + t_mm + t_prd + t_inv + t_mad + t_msc + t_ama; 
  std::cout << "Total Time = " << totTime1 << "  (s)  -  cpu " <<  totTime2 << "  (s) " << std::endl; 
  std::cerr << "SMatrixSym:  r1 = " << r1 << std::endl; 

  return 0;
}
#endif


// ROOT test 


template<unsigned int NDIM1, unsigned int NDIM2> 
int test_tmatrix_op() {


    

  typedef TMatrixD MnMatrix;
  typedef TVectorD MnVector;
  
//   typedef boost::numeric::ublas::matrix<double>  MnMatrix;  
  //typedef HepSymMatrix MnSymMatrixHep; 


  int first = NDIM1;  //Can change the size of the matrices
  int second = NDIM2;


  std::cout << "************************************************\n";
  std::cout << "  TMatrix operations test  "   <<  first << " x " << second  << std::endl;
  std::cout << "************************************************\n";
  
  double t_veq, t_meq, t_vad, t_mad, t_dot, t_mv, t_gmv, t_mm, t_prd, t_inv, t_vsc, t_msc, t_ama, t_tra = 0;
  double totTime1, totTime2; 
   
  double r1,r2;
  int npass = NITER; 
  TRandom3 r(111);
  gMatrixCheck = 0;
  
  for (int k = 0; k < npass; k++) {


    MnMatrix   A(NDIM1,NDIM2);
    MnMatrix   B(NDIM2,NDIM1);
    MnMatrix   C(NDIM1,NDIM1); 
    MnMatrix   D(NDIM2,NDIM2); 
    MnVector   v(NDIM1);
    MnVector   p(NDIM2);


    TStopwatch w; 
    { 
      // fill matrices with random data
      fillRandomMat(r,A,first,second);
      fillRandomMat(r,B,second,first);
      fillRandomMat(r,C,first,first);
      fillRandomMat(r,D,second,second);

      fillRandomVec(r,v,first);
      fillRandomVec(r,p,second);

    }
     
#ifdef DEBUG
    std::cout << "pass " << k << std::endl;
    if (k == 0) { 
      A.Print(); B.Print(); C.Print(); D.Print(); v.Print(); p.Print();
    }
#endif
    w.Start(); 
	
    
    MnVector v1(NDIM1);        testMV_T(A,v,t_mv,v1);
    //if (k == 0) v1.Print();
    MnVector v2(NDIM1);        testGMV_T(A,v,v1,t_gmv,v2);
    //if (k == 0) v2.Print();
    MnMatrix C0(NDIM1,NDIM1);  testMM_T(A,B,C,t_mm,C0);
    //if (k == 0) C0.Print();
    MnMatrix C1(NDIM1,NDIM1);  testATBA_T(B,C0,t_ama,C1);
    //if (k == 0) C1.Print();
    MnMatrix C2(NDIM1,NDIM1);  testInv_T(C1,t_inv,C2);
    //if (k == 0) C2.Print();
    MnVector v3(NDIM1);        testVeq(v,t_veq,v3);
    MnVector v4(NDIM1);        testVad_T(v2,v3,t_vad,v4);
    MnVector v5(NDIM1);        testVscale_T(v4,2.0,t_vsc,v5);
    MnMatrix C3(NDIM1,NDIM1);  testMeq(C,t_meq,C3);
    MnMatrix C4(NDIM1,NDIM1);  testMad_T(C2,C3,t_mad,C4);
    //if (k == 0) C4.Print();
    MnMatrix C5(NDIM1,NDIM1);  testMscale_T(C4,0.5,t_msc,C5);
    //if (k == 0) C5.Print();
    MnMatrix C6(NDIM1,NDIM1);  testMT_T(C5,t_tra,C6);

#ifdef DEBUG
    if (k == 0) { 
      C6.Print();
      v5.Print();
    }
#endif

    r1 = testDot_T(v3,v5,t_dot);

    r2 = testInnerProd_T(C6,v5,t_prd);

    //MnMatrix C2b(NDIM1,NDIM1); testInv_T2(C1,t_inv2,C2b);

  
    w.Stop();
    totTime1 = w.RealTime();
    totTime2 = w.CpuTime();
  }
  //  tr.dump();

  //double totTime = t_veq + t_meq + t_vad + t_mad + t_dot + t_mv + t_gmv + t_mm + t_prd + t_inv + t_inv2 + t_vsc + t_msc + t_ama + t_tra; 
  std::cout << "Total Time = " << totTime1 << "  (s)  -  cpu " <<  totTime2 << "  (s) " << std::endl; 
  std::cerr << "TMatrix:     r1 = " << r1 << " r2 = " << r2 << std::endl; 

  return 0;

}



#ifdef TEST_SYM
template<unsigned int NDIM1, unsigned int NDIM2> 
int test_tmatrix_sym_op() {
    
  // need to write explicitly the dimensions
   

  typedef TMatrixDSym MnSymMatrix;
  typedef TMatrixD    MnMatrix;
  typedef TVectorD MnVector;
  


  int first = NDIM1;  //Can change the size of the matrices
  

  std::cout << "************************************************\n";
  std::cout << "  TMatrixSym operations test  "   <<  first << " x " << first << std::endl;
  std::cout << "************************************************\n";


  double t_meq, t_mad, t_mv, t_gmv, t_mm, t_prd, t_inv, t_msc, t_ama = 0;
  double totTime1, totTime2; 
  
  
   
  double r1;
  int npass = NITER; 
  TRandom3 r(111);
  for (int k = 0; k < npass; k++) {


    MnSymMatrix A(NDIM1);
    MnSymMatrix B(NDIM1);
    MnMatrix C(NDIM1,NDIM1);
    MnVector v(NDIM1);
#define N NDIM1

    TStopwatch w; 

    {       
      // fill matrices with random data
      fillRandomSym(r,A,first);
      fillRandomSym(r,B,first);
      fillRandomMat(r,C,first,first);

      fillRandomVec(r,v,first);

    }

     
#ifdef DEBUG
    std::cout << "pass " << k << std::endl;
    if (k == 0) { 
      A.Print(); B.Print(); C.Print();  v.Print(); 
    }
#endif

    w.Start(); 
	        
    MnVector   v1(N);  testMV_T(A,v,t_mv,v1);
    MnVector   v2(N);  testGMV_T(A,v,v1,t_gmv,v2);
    MnMatrix   C0(N,N);  testMM_T(A,B,C,t_mm,C0);
    MnSymMatrix  C1(N);  testATBA_T2(C0,B,t_ama,C1);
    MnSymMatrix  C2(N);  testInv_T(A,t_inv,C2);
    MnSymMatrix  C3(N);  testMeq(C2,t_meq,C3);
    MnSymMatrix  C4(N);  testMad_T(A,C3,t_mad,C4);
    MnSymMatrix  C5(N);  testMscale_T(C4,0.5,t_msc,C5);

    r1 = testInnerProd_T(C5,v2,t_prd);

#ifdef DEBUG
    std::cout << "output matrices" << std::endl;
    if (k == 0) { 
      C1.Print(); C3.Print(); C4.Print(); C5.Print();
    }
#endif

    w.Stop();
    totTime1 = w.RealTime();
    totTime2 = w.CpuTime();
  
  }
  //tr.dump();

  //double totTime = t_meq + t_mv + t_gmv + t_mm + t_prd + t_inv + t_mad + t_msc + t_ama; 
  std::cout << "Total Time = " << totTime1 << "  (s)  -  cpu " <<  totTime2 << "  (s) " << std::endl; 
  std::cerr << "TMatrixSym:  r1 = " << r1 << std::endl; 

  return 0;
}
#endif  // end TEST_SYM

#ifdef HAVE_CLHEP

template<unsigned int NDIM1, unsigned int NDIM2> 
int test_hepmatrix_op() {


    

  typedef HepMatrix MnMatrix;
  typedef HepVector MnVector;
  


  int first = NDIM1;  //Can change the size of the matrices
  int second = NDIM2;


  std::cout << "************************************************\n";
  std::cout << "  HepMatrix operations test  "   <<  first << " x " << second  << std::endl;
  std::cout << "************************************************\n";
  
  double t_veq, t_meq, t_vad, t_mad, t_dot, t_mv, t_gmv, t_mm, t_prd, t_inv, t_vsc, t_msc, t_ama, t_tra = 0;
  

  double totTime1, totTime2; 
   
  double r1,r2;
  int npass = NITER; 
  TRandom3 r(111);

  for (int k = 0; k < npass; k++) {


    MnMatrix   A(NDIM1,NDIM2);
    MnMatrix   B(NDIM2,NDIM1);
    MnMatrix   C(NDIM1,NDIM1); 
    MnMatrix   D(NDIM2,NDIM2); 
    MnVector   v(NDIM1);
    MnVector   p(NDIM2);

    TStopwatch w; 

    { 
      // fill matrices with random data
      fillRandomMat(r,A,first,second,1);
      fillRandomMat(r,B,second,first,1);
      fillRandomMat(r,C,first,first,1);
      fillRandomMat(r,D,second,second,1);

      fillRandomVec(r,v,first);
      fillRandomVec(r,p,second);
    }

#ifdef DEBUG
    std::cout << "pass " << k << std::endl;
    if (k == 0) { 
      std::cout << " A = " << A << std::endl;
      std::cout << " B = " << B << std::endl;
      std::cout << " C = " << C << std::endl;
      std::cout << " D = " << D << std::endl;
      std::cout << " v = " << v << std::endl;
      std::cout << " p = " << p << std::endl;
    }
#endif
	
    w.Start(); 
    
    MnVector v1(NDIM1);        testMV(A,v,t_mv,v1);
    MnVector v2(NDIM1);        testGMV(A,v,v1,t_gmv,v2);
    MnMatrix C0(NDIM1,NDIM1);  testMM_C(A,B,C,t_mm,C0);
    MnMatrix C1(NDIM1,NDIM1);  testATBA_C(B,C0,t_ama,C1);
    //std::cout << " C1 = " << C1 << std::endl;
    MnMatrix C2(NDIM1,NDIM1);  testInv_C(C1,t_inv,C2);
    //std::cout << " C2 = " << C2 << std::endl;
    MnVector v3(NDIM1);        testVeq(v,t_veq,v3);
    MnVector v4(NDIM1);        testVad(v2,v3,t_vad,v4);
    MnVector v5(NDIM1);        testVscale(v4,2.0,t_vsc,v5);
    MnMatrix C3(NDIM1,NDIM1);  testMeq_C(C,t_meq,C3);
    MnMatrix C4(NDIM1,NDIM1);  testMad_C(C2,C3,t_mad,C4);
    //std::cout << " C4 = " << C4 << std::endl;
    MnMatrix C5(NDIM1,NDIM1);  testMscale_C(C4,0.5,t_msc,C5);
    //std::cout << " C5 = " << C5 << std::endl;
    MnMatrix C6(NDIM1,NDIM1);  testMT_C(C5,t_tra,C6);


    r1 = testDot_C(v3,v5,t_dot);
    r2 = testInnerProd_C(C6,v5,t_prd);

#ifdef DEBUG
    if (k == 0) { 
      std::cout << " C6 = " << C6 << std::endl;
      std::cout << " v5 = " << v5 << std::endl;
    }
#endif

    //    MnMatrix C2b(NDIM1,NDIM1); testInv_T2(C1,t_inv2,C2b);

    w.Stop();
    totTime1 = w.RealTime();
    totTime2 = w.CpuTime();
  
  }
  //  tr.dump();

  std::cout << "Total Time = " << totTime1 << "  (s)  -  cpu " <<  totTime2 << "  (s) " << std::endl; 
  std::cerr << "HepMatrix:   r1 = " << r1 << " r2 = " << r2 << std::endl; 

  return 0;
}


#ifdef TEST_SYM
template<unsigned int NDIM1, unsigned int NDIM2> 
int test_hepmatrix_sym_op() {
    
  // need to write explicitly the dimensions
   

  typedef HepSymMatrix MnSymMatrix;
  typedef HepMatrix    MnMatrix;
  typedef HepVector MnVector;
  


  int first = NDIM1;  //Can change the size of the matrices
  

  std::cout << "************************************************\n";
  std::cout << "  HepMatrixSym operations test  "   <<  first << " x " << first << std::endl;
  std::cout << "************************************************\n";


  double t_meq, t_mad, t_mv, t_gmv, t_mm, t_prd, t_inv, t_msc, t_ama = 0;
  
  double totTime1, totTime2; 
  
   
  double r1;
  int npass = NITER; 
  TRandom3 r(111);
  for (int k = 0; k < npass; k++) {


    MnSymMatrix A(NDIM1);
    MnSymMatrix B(NDIM1);
    MnMatrix C(NDIM1,NDIM1);
    MnVector v(NDIM1);
#define N NDIM1

    TStopwatch w; 

    {       
      // fill matrices with random data
      fillRandomSym(r,A,first,1);
      fillRandomSym(r,B,first,1);
      fillRandomMat(r,C,first,first,1);
      fillRandomVec(r,v,first);

    }

     
#ifdef DEBUG
    std::cout << "pass " << k << std::endl;
    if (k == 0) { 
    }
#endif
    
    w.Start(); 
	        
    MnVector   v1(N);  testMV(A,v,t_mv,v1);
    MnVector   v2(N);  testGMV(A,v,v1,t_gmv,v2);
    MnMatrix   C0(N,N);  testMM_C(A,B,C,t_mm,C0);
    MnSymMatrix  C1(N);  testATBA_C2(C0,B,t_ama,C1);
    MnSymMatrix  C2(N);  testInv_C(A,t_inv,C2);
    MnSymMatrix  C3(N);  testMeq_C(C2,t_meq,C3);
    MnSymMatrix  C4(N);  testMad_C(A,C3,t_mad,C4);
    MnSymMatrix  C5(N);  testMscale_C(C4,0.5,t_msc,C5);

    r1 = testInnerProd_C(C5,v2,t_prd);

#ifdef DEBUG
    std::cout << "output matrices" << std::endl;
    if (k == 0) { 
    }
#endif

    w.Stop();
    totTime1 = w.RealTime();
    totTime2 = w.CpuTime();
  
  }
  //tr.dump();

  std::cout << "Total Time = " << totTime1 << "  (s)  -  cpu " <<  totTime2 << "  (s) " << std::endl; 
  std::cerr << "HepMatrixSym: r1 = " << r1 << std::endl; 

  return 0;
}

#endif  // TEST_SYM
#endif  // HAVE_CLHEP


#if defined(HAVE_CLHEP) && defined (TEST_SYM)
#define NTYPES 6
#define TEST(N) \
   MATRIX_SIZE=N;  \
   TEST_TYPE=0; test_smatrix_op<N,N>(); \
   TEST_TYPE=1; test_tmatrix_op<N,N>();     \
   TEST_TYPE=2; test_hepmatrix_op<N,N>();   \
   TEST_TYPE=3; test_smatrix_sym_op<N,N>(); \
   TEST_TYPE=4; test_tmatrix_sym_op<N,N>();     \
   TEST_TYPE=5; test_hepmatrix_sym_op<N,N>();
#elif !defined(HAVE_CLHEP) && defined (TEST_SYM)
#define NTYPES 4
#define TEST(N) \
   MATRIX_SIZE=N;  \
   TEST_TYPE=0; test_smatrix_op<N,N>(); \
   TEST_TYPE=1; test_tmatrix_op<N,N>();     \
   TEST_TYPE=2; test_smatrix_sym_op<N,N>(); \
   TEST_TYPE=3; test_tmatrix_sym_op<N,N>();     
#elif defined(HAVE_CLHEP) && !defined (TEST_SYM)
#define NTYPES 3
#define TEST(N) \
   MATRIX_SIZE=N;  \
   TEST_TYPE=0; test_smatrix_op<N,N>(); \
   TEST_TYPE=1; test_tmatrix_op<N,N>();     \
   TEST_TYPE=2; test_hepmatrix_op<N,N>();
#else 
#define NTYPES 2
#define TEST(N) \
   TEST_TYPE=0; test_smatrix_op<N,N>(); \
   TEST_TYPE=1; test_tmatrix_op<N,N>();     
#endif



int TEST_TYPE; 
int MATRIX_SIZE; 
#ifdef REPORT_TIME
std::vector< std::map<std::string, TH1D *> > testTimeResults(NTYPES); 
std::vector< std::string > typeNames(NTYPES);

void ROOT::Math::test::reportTime(std::string s, double time) { 
  assert( TEST_TYPE >= 0 && TEST_TYPE < NTYPES );
  std::map<std::string, TH1D * > & result = testTimeResults[TEST_TYPE];
  
  std::map<std::string, TH1D * >::iterator pos = result.find(s);   
  TH1D * h = 0; 
  if (  pos != result.end() ) { 
    h = pos->second; 
  }
  else { 
    // add new elements in map
    //std::cerr << "insert element in map" << s << typeNames[TEST_TYPE] << std::endl;
    std::string name = typeNames[TEST_TYPE] + "_" + s; 
    h = new TProfile(name.c_str(), name.c_str(),100,0.5,100.5);
    //result.insert(std::map<std::string, TH1D * >::value_type(s,h) ); 
    result[s] = h;
  }
  double scale=1; 
  if (s.find("dot") != std::string::npos || 
      s.find("V=V") != std::string::npos || 
      s.find("V+V") != std::string::npos ) scale = 10;  
  h->Fill(double(MATRIX_SIZE),time/double(NLOOP*NITER*scale) ); 
}
#endif

int testOperations() { 

  NLOOP = 1000*NLOOP_MIN 
  initValues();

  TEST(5)

   return 0;   
}


int main(int argc , char *argv[] ) { 


  std::string fname = "testOperations"; 
  if (argc > 1) { 
    std::string platf(argv[1]); 
    fname = fname + "_" + platf; 
  }
  fname = fname + ".root"; 


#ifdef REPORT_TIME
  TFile * f = new TFile(fname.c_str(),"recreate");

  typeNames[0] = "SMatrix";
  typeNames[1] = "TMatrix";
#if !defined(HAVE_CLHEP) && defined (TEST_SYM)
  typeNames[2] = "SMatrix_sym";
  typeNames[3] = "TMatrix_sym";
#elif defined(HAVE_CLHEP) && defined (TEST_SYM)
  typeNames[2] = "HepMatrix";
  typeNames[3] = "SMatrix_sym";
  typeNames[4] = "TMatrix_sym";
  typeNames[5] = "HepMatrix_sym";
#elif defined(HAVE_CLHEP) && !defined (TEST_SYM)
  typeNames[2] = "HepMatrix";
#endif

#endif

#ifndef TEST_ALL_MATRIX_SIZES
//   NLOOP = 1000*NLOOP_MIN 
//   initValues();

//   TEST(5)
//   NLOOP = 50*NLOOP_MIN;
//   TEST(30);

  return testOperations();

#else
  NLOOP = 5000*NLOOP_MIN; 
  initValues();



  TEST(2);
  TEST(3);
  TEST(4);
  NLOOP = 1000*NLOOP_MIN 
  TEST(5);
  TEST(6);
  TEST(7);
  TEST(10); 
  NLOOP = 100*NLOOP_MIN; 
  TEST(15); 
  TEST(20);
  NLOOP = 50*NLOOP_MIN; 
  TEST(30);
//   NLOOP = NLOOP_MIN;
//   TEST(50);
//   TEST(75);
//   TEST(100);
#endif

#ifdef REPORT_TIME
  f->Write();
  f->Close();
#endif

}

