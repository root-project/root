#ifndef MATRIX_OP_H 
#define MATRIX_OP_H

#include "TestTimer.h"

// define funcitons for matrix operations

//#define DEBUG
//#ifndef NLOOP
//#define NLOOP 1000000
//#endif
using namespace ROOT::Math;


// vector assignment
template<class V> 
void testVeq(const V & v, double & time, V & result) {  
  test::Timer t(time,"V=V ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = v;  
    }
}

// matrix assignmnent
template<class M> 
void testMeq(const M & m, double & time, M & result) {  
  test::Timer t(time,"M=M ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = m;  
    }
}



// vector sum 
template<class V> 
void testVad(const V & v1, const V & v2, double & time, V & result) {  
  test::Timer t(time,"V+V ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = v1 + v2;  
    }
}

// matrix sum 
template<class M> 
void testMad(const M & m1, const M & m2, double & time, M & result) {  
  test::Timer t(time,"M+M ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = m1 + m2;  
    }
}

// vector * constant
template<class V> 
void testVscale(const V & v1, double a, double & time, V & result) {  
  test::Timer t(time,"a*V ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = a * v1;   // v1 * a does not exist in ROOT   
    }
}


// matrix * constant
template<class M> 
void testMscale(const M & m1, double a, double & time, M & result) {  
  test::Timer t(time,"a*M ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = m1 * a;  
    }
}


// simple Matrix vector op
template<class M, class V> 
void testMV(const M & mat, const V & v, double & time, V & result) {  
  test::Timer t(time,"M*V ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = mat * v;  
    }
}

// general matrix vector op
template<class M, class V> 
void testGMV(const M & mat, const V & v1, const V & v2, double & time, V & result) {  
  test::Timer t(time,"M*V+");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = mat * v1 + v2;  
    }
}

// general matrix matrix op 
template<class A, class B, class C> 
void testMM(const A & a, const B & b, const C & c, double & time, C & result) {  
  test::Timer t(time,"M*M ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = a * b + c;  
    }
}



// specialized functions (depending on the package) 

//smatrix
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
      //#ifndef WIN32
      result = Similarity(v,a);  
// #else 
//       // cannot instantiate on Windows (don't know why? )
//       V tmp = a*v; 
//       result = Dot(v,tmp);
// #endif
    }
  return result; 
}

//inversion
template<class M> 
void  testInv_S( const M & a,  double & time, M& result){ 
  test::Timer t(time,"inv ");
  int ifail = 0;
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = a.Inverse(ifail);
      // assert(ifail == 0);
    }
}


// general matrix matrix op
template<class A, class B, class C> 
void testATBA_S(const A & a, const B & b, double & time, C & result) {  
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      //result = Transpose(a) * b * a;  
      //result = a * b * Transpose(a);  
      //result = a * b * a;  
      C tmp = b * Transpose(a);
      result = a * tmp; 
    }
}

// general matrix matrix op
template<class A, class B, class C> 
void testATBA_S2(const A & a, const B & b, double & time, C & result) {  
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      //result = Transpose(a) * b * a;  
      //result = a * b * Transpose(a);  
      //result = a * b * a;  
      result  = SimilarityT(a,b);
      //result = a * result; 
    }
}

template<class A, class C> 
void testMT_S(const A & a, double & time, C & result) {  
  test::Timer t(time,"Transp");
  for (int l = 0; l < NLOOP; l++) 	
    {
      //result = Transpose(a) * b * a;  
      //result = a * b * Transpose(a);  
      //result = a * b * a;  
      result  = Transpose(a);
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



// general matrix matrix op
template<class A, class B, class C> 
void testATBA_T(const A & a, const B & b, double & time, C & result) {  
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      A a2 = a; a2.T();
      result = a * b * a2;  
      //result = b * a2;  
    }
}


// general matrix matrix op
template<class A, class B, class C> 
void testATBA_T2(const A & a, const B & b, double & time, C & result) {  
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = b;
      result = result.Similarity(a); 
    }
}

template<class A, class C> 
void testMT_T(const A & a, double & time, C & result) {  
  test::Timer t(time,"Transp");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result  = a; 
      result = result.T();
    }
}

 
// for clhep


//smatrix
template<class V> 
double testDot_C(const V & v1, const V & v2, double & time) {  
  test::Timer t(time,"dot ");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = dot(v1,v2);  
    }
  return result; 
}

template<class M, class V> 
double testInnerProd_C(const M & a, const V & v, double & time) {  
  test::Timer t(time,"prod");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      V tmp = a*v; 
      result = dot(v,tmp);
    }
  return result; 
}

//inversion
template<class M> 
void  testInv_C( const M & a,  double & time, M& result){ 
  test::Timer t(time,"inv ");
  int ifail = 0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = a.inverse(ifail);  
    }
}

// general matrix matrix op
template<class A, class B, class C> 
void testATBA_C(const A & a, const B & b, double & time, C & result) {  
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      //result = a.T() * b * a;  
      result = a * b * a.T();  
    }
}


template<class A, class B, class C> 
void testATBA_C2(const A & a, const B & b, double & time, C & result) {  
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      result = b.similarity(a); 
    }
}


#endif   
