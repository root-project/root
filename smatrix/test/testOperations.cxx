

#include "Math/SVector.h"
#include "Math/SMatrix.h"

#include "TMatrixD.h"
#include "TVectorD.h"

#include "TRandom3.h"

// #include "SealUtil/SealTimer.h"
// #include "SealUtil/SealHRRTChrono.h"
// #include "SealUtil/TimingReport.h"

#include <iostream>

#ifndef NDIM1
#define NDIM1 5
#endif
#ifndef NDIM2
#define NDIM2 5
#endif

#define NITER 1  // number of iterations

using namespace ROOT::Math;

#include "matrix_op.h"



template<class V> 
double testDot_S(const V & v1, const V & v2, double & time) {  
  test::Timer t(time,"dot ");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = Dot(v1,v2);  
    }
  return result; 
}

template<class M, class V> 
double testInnerProd_S(const M & a, const V & v, double & time) {  
  test::Timer t(time,"prod");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
#ifndef WIN32
      result = Product(v,a);  
#else 
      // cannot instantiate on Windows (don't know why? )
      V tmp = a*v; 
      result = Dot(v,tmp);
#endif
    }
  return result; 
}

//inversion
template<class M> 
void  testInv_S( const M & a,  double & time, M& result){ 
  test::Timer t(time,"inv ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = a.Inverse();  
    }
}

// for root


template<class V> 
double testDot_T(const V & v1, const V & v2, double & time) {  
  test::Timer t(time,"dot ");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = v1*v2;
    }
  return result; 
}

template<class M, class V> 
double testInnerProd_T(const M & a, const V & v, double & time) {  
  test::Timer t(time,"prod");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      V tmp = a * v;
      result = v * tmp;
    }
  return result; 
}

//inversion 
template<class M> 
void  testInv_T(const M & a,  double & time, M& result){ 
  test::Timer t(time,"inv ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = a; 
      result.InvertFast(); 
    }
}

template<class M> 
void  testInv_T2(const M & a,  double & time, M& result){ 
  test::Timer t(time,"inv2");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = a; 
      result.InvertFast();  
    }
}



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

 
//   seal::TimingReport tr;
//   seal::TimingItem & init = tr.item<seal::SealHRRTChrono>("init");

//   seal::TimingItem & t_veq = tr.item<seal::SealHRRTChrono>("smatrix veq");
//   seal::TimingItem & t_meq = tr.item<seal::SealHRRTChrono>("smatrix meq");
//   seal::TimingItem & t_vad = tr.item<seal::SealHRRTChrono>("smatrix vad");
//   seal::TimingItem & t_mad = tr.item<seal::SealHRRTChrono>("smatrix mad");
//   seal::TimingItem & t_dot = tr.item<seal::SealHRRTChrono>("smatrix dot");
//   seal::TimingItem & t_mv  = tr.item<seal::SealHRRTChrono>("smatrix MV ");
//   seal::TimingItem & t_gmv = tr.item<seal::SealHRRTChrono>("smatrix GMV");
//   seal::TimingItem & t_mm  = tr.item<seal::SealHRRTChrono>("smatrix GMM");
//   seal::TimingItem & t_prd = tr.item<seal::SealHRRTChrono>("smatrix prd");
//   seal::TimingItem & t_inv = tr.item<seal::SealHRRTChrono>("smatrix inv");

  double t_veq, t_meq, t_vad, t_mad, t_dot, t_mv, t_gmv, t_mm, t_prd, t_inv, t_vsc, t_msc = 0;
  
  
   
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

    {       
      //seal::SealTimer t(init,false);
      // fill matrices with ranodm data
      for(int i = 0; i < first; i++) 
	for(int j = 0; j < second; j++)
	  A(i,j) = r.Rndm() + 1.;
      for(int i = 0; i < second; i++) 
	for(int j = 0; j < first; j++) 
	  B(i,j) = r.Rndm() + 1.;
      for(int i = 0; i < first; i++) 
	for(int j = 0; j < first; j++) 
	  C(i,j) = r.Rndm() + 1.;
      for(int i = 0; i < second; i++) 
	for(int j = 0; j < second; j++) 
	  D(i,j) = r.Rndm() + 1.;
	
//       // sym matrices 
//       for(int i = 0; i < second; i++) 
// 	for(int j = 0; j <=i; j++) {  
// 	  C(i,j) = r.Rndm() + 1.;
//       if (j != i) Cp(j,i) = Cp(i,j);
// 	}
      // vectors
      for(int i = 0; i < first; i++) 
	v(i) = r.Rndm() + 1.;
      for(int i = 0; i < second; i++) 
	p(i) = r.Rndm() + 1.;

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
	        
    MnVectorN   v1;  testMV(A,v,t_mv,v1);
    MnVectorN   v2;  testGMV(A,v,v1,t_gmv,v2);
    MnMatrixNN  C1;  testMM(A,B,C,t_mm,C1);
    MnMatrixNN  C2;  testInv_S(C1,t_inv,C2);
    MnVectorN   v3;  testVeq(v,t_veq,v3);
    MnVectorN   v4;  testVad(v2,v3,t_vad,v4);
    MnVectorN   v5;  testVscale(v4,2.0,t_vsc,v5);
    MnMatrixNN  C3;  testMeq(C,t_meq,C3);
    MnMatrixNN  C4;  testMad(C2,C3,t_mad,C4);
    MnMatrixNN  C5;  testMscale(C4,0.5,t_msc,C5);

    r1 = testDot_S(v3,v5,t_dot);
    r2 = testInnerProd_S(C5,v5,t_prd);
  
  }
  //tr.dump();

  std::cout << " r1 = " << r1 << " r2 = " << r2 << std::endl; 

  return 0;
}

// ROOT test 


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
  
  double t_veq, t_meq, t_vad, t_mad, t_dot, t_mv, t_gmv, t_mm, t_prd, t_inv, t_inv2, t_vsc, t_msc = 0;
  
   
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

    { 
      //      seal::SealTimer t(init,false);
      // fill matrices with ranodm data
      for(int i = 0; i < first; i++) 
	for(int j = 0; j < second; j++)
	  A(i,j) = r.Rndm() + 1.;
      for(int i = 0; i < second; i++) 
	for(int j = 0; j < first; j++) 
	  B(i,j) = r.Rndm() + 1.;
      for(int i = 0; i < first; i++) 
	for(int j = 0; j < first; j++) 
	  C(i,j) = r.Rndm() + 1.;
      for(int i = 0; i < second; i++) 
	for(int j = 0; j < second; j++) 
	  D(i,j) = r.Rndm() + 1.;
	
//       // sym matrices 
//       for(int i = 0; i < second; i++) 
// 	for(int j = 0; j <=i; j++) {  
// 	  C(i,j) = r.Rndm() + 1.;
//       if (j != i) Cp(j,i) = Cp(i,j);
// 	}
      // vectors
      for(int i = 0; i < first; i++) 
	v(i) = r.Rndm() + 1.;
      for(int i = 0; i < second; i++) 
	p(i) = r.Rndm() + 1.;

    }


//     MnSymMatrixMM I; 
//     for(int i = 0; i < second; i++) 
//       I(i,i) = 1;
     
#ifdef DEBUG
    std::cout << "pass " << k << std::endl;
    if (k == 0) { 
      A.Print(); B.Print(); C.Print(); D.Print(); v.Print(); p.Print();
    }
#endif
	
    
    MnVector v1(NDIM1);        testMV(A,v,t_mv,v1);
    MnVector v2(NDIM1);        testGMV(A,v,v1,t_gmv,v2);
    MnMatrix C1(NDIM1,NDIM1);  testMM(A,B,C,t_mm,C1);
    MnMatrix C2(NDIM1,NDIM1);  testInv_T(C1,t_inv,C2);
    MnVector v3(NDIM1);        testVeq(v,t_veq,v3);
    MnVector v4(NDIM1);        testVad(v2,v3,t_vad,v4);
    MnVector v5(NDIM1);        testVscale(v4,2.0,t_vsc,v5);
    MnMatrix C3(NDIM1,NDIM1);  testMeq(C,t_meq,C3);
    MnMatrix C4(NDIM1,NDIM1);  testMad(C2,C3,t_mad,C4);
    MnMatrix C5(NDIM1,NDIM1);  testMscale(C4,0.5,t_msc,C5);


    r1 = testDot_T(v3,v5,t_dot);
    r2 = testInnerProd_T(C5,v5,t_prd);

    MnMatrix C2b(NDIM1,NDIM1); testInv_T2(C1,t_inv2,C2b);

  
  }
  //  tr.dump();

  std::cout << " r1 = " << r1 << " r2 = " << r2 << std::endl; 

  return 0;
}



int main() { 

  test_smatrix_op();
  test_tmatrix_op();
}
