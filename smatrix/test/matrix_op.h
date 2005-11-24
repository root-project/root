#include "TestTimer.h"

// define funcitons for matrix operations

//#define DEBUG
#ifndef NLOOP
#define NLOOP 1000000
#endif


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
