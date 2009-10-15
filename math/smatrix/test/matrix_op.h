#ifndef MATRIX_OP_H 
#define MATRIX_OP_H

#include "TestTimer.h"

// define functions for matrix operations

//#define DEBUG
//#ifndef NLOOP
//#define NLOOP 1000000
//#endif
#include <vector>

using namespace ROOT::Math;

std::vector<float> gV; 

void initValues() { 
  gV.reserve(10*NLOOP); 
  TRandom3 r; 
  std::cout << "init smearing vector ";
  for (int l = 0; l < 10*NLOOP; l++) 	
    {
      gV.push_back( r.Rndm() );  
    } 
  std::cout << " with size  " << gV.size() << std::endl;

}


// vector assignment
template<class V> 
void testVeq(const V & v, double & time, V & result) {  
  V vtmp = v; 
  test::Timer t(time,"V=V ");
  for (int l = 0; l < 10*NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      result = vtmp;  
    }
}

// matrix assignmnent
template<class M> 
void testMeq(const M & m, double & time, M & result) {  
  M mtmp = m;
  test::Timer t(time,"M=M ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(0,0) = gV[l];
      result = mtmp;  
    }
}



// vector sum 
template<class V> 
void testVad(const V & v1, const V & v2, double & time, V & result) {  
  V vtmp = v2; 
  test::Timer t(time,"V+V ");
  for (int l = 0; l < 10*NLOOP; l++) 	
    {
      vtmp[0] = gV[l]; 
      result = v1 + vtmp;  
    }
}

// matrix sum 
template<class M> 
void testMad(const M & m1, const M & m2, double & time, M & result) {  
  M mtmp = m2;
  test::Timer t(time,"M+M ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(0,0) = gV[l]; 
      result = m1; result += mtmp;  
      //M tmp = m1 + mtmp;
      //result = tmp;  
    }
}

// vector * constant
template<class V> 
void testVscale(const V & v1, double a, double & time, V & result) {  
  V vtmp = v1; 
  test::Timer t(time,"a*V ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      result = a * vtmp;   // v1 * a does not exist in ROOT   
    }
}


// matrix * constant
template<class M> 
void testMscale(const M & m1, double a, double & time, M & result) {  
  M mtmp = m1;
  test::Timer t(time,"a*M ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(0,0) = gV[l];
      //result = mtmp * a;  
      result = mtmp; result *= a;
    }
}


// simple Matrix vector op
template<class M, class V> 
void testMV(const M & mat, const V & v, double & time, V & result) {  
  V vtmp = v; 
  test::Timer t(time,"M*V ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      result = mat * vtmp;  
    }
}

// general matrix vector op
template<class M, class V> 
void testGMV(const M & mat, const V & v1, const V & v2, double & time, V & result) {  
  V vtmp = v1; 
  test::Timer t(time,"M*V+");
  for (int l = 0; l < NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      result = mat * vtmp + v2; 
    }
}


// general matrix matrix op 
template<class A, class B, class C> 
void testMM(const A & a, const B & b, const C & c, double & time, C & result) {  
  B btmp = b; 
  test::Timer t(time,"M*M ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      btmp(0,0) = gV[l];
      result = a * btmp + c;
    }
}




// specialized functions (depending on the package) 

//smatrix
template<class V> 
double testDot_S(const V & v1, const V & v2, double & time) {  
  V vtmp = v2; 
  test::Timer t(time,"dot ");
  double result=0; 
  for (int l = 0; l < 10*NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      result = Dot(v1,vtmp);  
    }
  return result; 
}


// double testDot_S(const std::vector<V*> & w1, const std::vector<V*> & w2, double & time) {  
//   test::Timer t(time,"dot ");
//   double result=0; 
//   for (int l = 0; l < NLOOP; l++) 	
//     {
//       V & v1 = *w1[l]; 
//       V & v2 = *w2[l]; 
//       result = Dot(v1,v2);  
//     }
//   return result; 
// }

template<class M, class V> 
double testInnerProd_S(const M & a, const V & v, double & time) {  
  V vtmp = v; 
  test::Timer t(time,"prod");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      vtmp[0] =  gV[l];
      result = Similarity(vtmp,a);  
    }
  return result; 
}

//inversion
template<class M> 
void  testInv_S( const M & m,  double & time, M& result){
  M mtmp = m;
  test::Timer t(time,"inv ");
  int ifail = 0;
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(0,0) = gV[l]; 
      //result = mtmp.InverseFast(ifail);
      result = mtmp.Inverse(ifail);
      // assert(ifail == 0);
    }
}


// general matrix matrix op
template<class A, class B, class C> 
void testATBA_S(const A & a, const B & b, double & time, C & result) {  
  B btmp = b;
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      //result = Transpose(a) * b * a;  
      //result = a * b * Transpose(a);  
      //result = a * b * a;  
      btmp(0,0) = gV[l];
      //      A at = Transpose(a);
      C tmp = btmp * Transpose(a);
      result = a * tmp; 
    }
}

// general matrix matrix op
template<class A, class B, class C> 
void testATBA_S2(const A & a, const B & b, double & time, C & result) {  
  B btmp = b;

  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      //result = Transpose(a) * b * a;  
      //result = a * b * Transpose(a);  
      //result = a * b * a;  
      btmp(0,0) = gV[l];
      result  = SimilarityT(a,b);
      //result = a * result; 
    }
}

template<class A, class C> 
void testMT_S(const A & a, double & time, C & result) {  
  A atmp = a;
  test::Timer t(time,"Transp");
  for (int l = 0; l < NLOOP; l++) 	
    {
      //result = Transpose(a) * b * a;  
      //result = a * b * Transpose(a);  
      //result = a * b * a;  
      atmp(0,0) = gV[l];
      result  = Transpose(atmp);
    }
}

/////////////////////////////////////
// for root
//////////////////////////////////

// simple Matrix vector op
template<class M, class V> 
void testMV_T(const M & mat, const V & v, double & time, V & result) {
  V vtmp = v; 
  test::Timer t(time,"M*V ");
  for (int l = 0; l < NLOOP; l++)
    {
      vtmp[0] = gV[l];
      Add(result,0.0,mat,vtmp);
    }
} 
  
// general matrix vector op
template<class M, class V> 
void testGMV_T(const M & mat, const V & v1, const V & v2, double & time, V & result) {
  V vtmp = v1;
  test::Timer t(time,"M*V+");
  for (int l = 0; l < NLOOP; l++)
    {
      vtmp[0] = gV[l];
      memcpy(result.GetMatrixArray(),v2.GetMatrixArray(),v2.GetNoElements()*sizeof(Double_t));
      Add(result,1.0,mat,vtmp);
    }
}

// general matrix matrix op
template<class A, class B, class C> 
void testMM_T(const A & a, const B & b, const C & c, double & time, C & result) {
  B btmp = b; 
  test::Timer t(time,"M*M ");
  for (int l = 0; l < NLOOP; l++)
    {
      btmp(0,0) = gV[l];
      result.Mult(a,btmp);
      result += c;
    }
} 

// matrix sum
template<class M> 
void testMad_T(const M & m1, const M & m2, double & time, M & result) {
  M mtmp = m2;
  test::Timer t(time,"M+M ");
  for (int l = 0; l < NLOOP; l++)
    {
      mtmp(0,0) = gV[l];
      result.Plus(m1,mtmp);
    }
}

template<class A, class B, class C> 
void testATBA_T(const A & a, const B & b, double & time, C & result) {
  B btmp = b;
  test::Timer t(time,"At*M*A");
  C tmp = a;
  for (int l = 0; l < NLOOP; l++)
    {
      btmp(0,0) = gV[l]; 
      tmp.Mult(a,btmp);
      result.MultT(tmp,a);
    }
}

template<class V> 
double testDot_T(const V & v1, const V & v2, double & time) {  
  V vtmp = v2;
  test::Timer t(time,"dot ");
  double result=0; 
  for (int l = 0; l < 10*NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      result = Dot(v1,vtmp);
    }
  return result; 
}

template<class M, class V> 
double testInnerProd_T(const M & a, const V & v, double & time) {  
  V vtmp = v; 
  test::Timer t(time,"prod");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) { 
    vtmp[0] =  gV[l];
    result = a.Similarity(vtmp);
  }
  return result; 
}

//inversion 
template<class M> 
void  testInv_T(const M & m,  double & time, M& result){ 
  M mtmp = m;
  test::Timer t(time,"inv ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(0,0) = gV[l]; 
      memcpy(result.GetMatrixArray(),mtmp.GetMatrixArray(),mtmp.GetNoElements()*sizeof(Double_t));
      result.InvertFast(); 
    }
}

template<class M> 
void  testInv_T2(const M & m,  double & time, M& result){ 
  M mtmp = m;
  test::Timer t(time,"inv2");
  for (int l = 0; l < NLOOP; l++) 	
    {
      memcpy(result.GetMatrixArray(),mtmp.GetMatrixArray(),mtmp.GetNoElements()*sizeof(Double_t));
      result.InvertFast();  
    }
}


// vector sum
template<class V> 
void testVad_T(const V & v1, const V & v2, double & time, V & result) {
  V vtmp = v2; 
  test::Timer t(time,"V+V ");;
  for (int l = 0; l < 10*NLOOP; l++)
    {
      vtmp[0] = gV[l];
      result.Add(v1,vtmp);
    }
}

// vector * constant
template<class V> 
void testVscale_T(const V & v1, double a, double & time, V & result) {
  V vtmp = v1; 
  test::Timer t(time,"a*V ");;
  for (int l = 0; l < NLOOP; l++)
    {
      // result = a * v1;
      result.Zero();
      vtmp[0] = gV[l];
      Add(result,a,vtmp);
    }
}

// general matrix matrix op
template<class A, class B, class C> 
void testATBA_T2(const A & a, const B & b, double & time, C & result) {  
  B btmp = b;
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      btmp(0,0) = gV[l]; 
      memcpy(result.GetMatrixArray(),btmp.GetMatrixArray(),btmp.GetNoElements()*sizeof(Double_t));
      result.Similarity(a); 
    }
}

// matrix * constant
template<class M>
void testMscale_T(const M & m1, double a, double & time, M & result) {
  M mtmp = m1;
  test::Timer t(time,"a*M ");;
  for (int l = 0; l < NLOOP; l++)
    {
      //result = a * m1;
      result.Zero();
      mtmp(0,0) = gV[l];
      Add(result,a,mtmp);
    }
}

template<class A, class C> 
void testMT_T(const A & a, double & time, C & result) {  
  A atmp = a;
  test::Timer t(time,"Transp");
  for (int l = 0; l < NLOOP; l++) 	
    {
      atmp(0,0) = gV[l];
      result.Transpose(atmp);
    }
}

//////////////////////////////////////////// 
// for clhep
////////////////////////////////////////////

//smatrix
template<class V> 
double testDot_C(const V & v1, const V & v2, double & time) {  
  V vtmp =  v2;
  test::Timer t(time,"dot ");
  double result=0; 
  for (int l = 0; l < 10*NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      result = dot(v1,vtmp);  
    }
  return result; 
}

template<class M, class V> 
double testInnerProd_C(const M & a, const V & v, double & time) {  
  V vtmp = v; 
  test::Timer t(time,"prod");
  double result=0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      vtmp[0] = gV[l];
      V tmp = a*vtmp; 
      result = dot(vtmp,tmp);
    }
  return result; 
}


// matrix assignmnent(index starts from 1)
template<class M> 
void testMeq_C(const M & m, double & time, M & result) {  
  M mtmp = m;
  test::Timer t(time,"M=M ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(1,1) = gV[l];
      result = mtmp;  
    }
}

// matrix sum 
template<class M> 
void testMad_C(const M & m1, const M & m2, double & time, M & result) {  
  M mtmp = m2;
  test::Timer t(time,"M+M ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(1,1) = gV[l]; 
      result = m1; result += mtmp;  
    }
}


// matrix * constant
template<class M> 
void testMscale_C(const M & m1, double a, double & time, M & result) {  
  M mtmp = m1;
  test::Timer t(time,"a*M ");;
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(1,1) = gV[l];
      result = mtmp * a;  
    }
}


// clhep matrix matrix op (index starts from 1)
template<class A, class B, class C> 
void testMM_C(const A & a, const B & b, const C & c, double & time, C & result) {  
  B btmp = b; 
  test::Timer t(time,"M*M ");
  for (int l = 0; l < NLOOP; l++) 	
    {
      btmp(1,1) = gV[l];
      result = a * btmp + c;
    }
}


//inversion
template<class M> 
void  testInv_C( const M & a,  double & time, M& result){ 
  M mtmp = a;
  test::Timer t(time,"inv ");
  int ifail = 0; 
  for (int l = 0; l < NLOOP; l++) 	
    {
      mtmp(1,1) = gV[l]; 
      result = mtmp.inverse(ifail); 
      if (ifail) {std::cout <<"error inverting" << mtmp << std::endl; return; } 
    }
}

// general matrix matrix op
template<class A, class B, class C> 
void testATBA_C(const A & a, const B & b, double & time, C & result) {  
  B btmp = b;
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      btmp(1,1) = gV[l];
      //result = a.T() * b * a;  
      result = a * btmp * a.T();  
    }
}


template<class A, class B, class C> 
void testATBA_C2(const A & a, const B & b, double & time, C & result) { 
  B btmp = b; 
  test::Timer t(time,"At*M*A");
  for (int l = 0; l < NLOOP; l++) 	
    {
      btmp(1,1) = gV[l]; 
      result = btmp.similarity(a); 
    }
}


template<class A, class C> 
void testMT_C(const A & a, double & time, C & result) {  
  A atmp = a;
  test::Timer t(time,"Transp");
  for (int l = 0; l < NLOOP; l++) 	
    {
      atmp(1,1) = gV[l];
      result  = atmp.T();
    }
}


#endif   
