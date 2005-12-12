#include <cmath>
#include "Math/SVector.h"
#include "Math/SMatrix.h"

#include <iostream>
#include <vector>

using namespace ROOT::Math;

using std::cout;
using std::endl;


//#define TEST_STATIC_CHECK  // for testing compiler failures (static check)

#define XXX

int compare( double a, double b) { 
  if (a == b) return 0; 
  std::cout << "\nFailure " << a << " diffent than " << b << std::endl;
  return 1;
}

int test1() { 

  SVector<float,3> x(4,5,6);
  SVector<float,2> y(2.0,3.0);
  cout << "x: " << x << endl;
  cout << "y: " << y << endl;

//   float yy1=2.; float yy2 = 3;
//   SVector<float,2> y2(yy1,yy2);
  
  
  SMatrix<float,4,3> A;
  SMatrix<float,2,2> B;
  
  A.Place_in_row(y, 1, 1);
  A.Place_in_col(x + 2, 1, 0);
  A.Place_at(B , 2, 1);
  cout << "A: " << endl << A << endl;
  
#ifdef TEST_STATIC_CHECK
  // create a vector of size 2 from 3 arguments
  SVector<float, 2> v(1,2,3);
#endif
  
  // test STL interface
  
  //float p[2] = {1,2};
  float m[4] = {1,2,3,4};
  
  //SVector<float, 2> sp(p,2);
  SMatrix<float, 2,2> sm(m,4);

  //cout << "sp: " << endl << sp << endl;
  cout << "sm: " << endl << sm << endl;

  //std::vector<float> vp(sp.begin(), sp.end() );
  std::vector<float> vm(sm.begin(), sm.end() );

  //SVector<float, 2> sp2(vp.begin(),vp.end());
  //SVector<float, 2> sp2(vp.begin(),vp.size());
  SMatrix<float, 2,2> sm2(vm.begin(),vm.end());

  //if ( sp2 != sp) { cout << "Test STL interface for SVector failed" << endl; return -1; }
  if ( sm2 != sm) { cout << "Test STL interface for SMatrix failed" << endl; return -1; }
    
  return 0;
    
}

int test2() { 
#ifdef XXX
  SMatrix<double,3> A;
  A(0,0) = A(1,0) = 1;
  A(0,1) = 3;
  A(1,1) = A(2,2) = 2;
  cout << "A: " << endl << A << endl;

  SVector<double,3> x = A.Row(0);
  cout << "x: " << x << endl;

  SVector<double,3> y = A.Col(1);
  cout << "y: " << y << endl;

  return 0;
#endif
}

int test3() { 
#ifdef XXX
  SMatrix<double,3> A;
  A(0,0) = A(0,1) = A(1,0) = 1;
  A(1,1) = A(2,2) = 2;

  SMatrix<double,3> B = A; // save A in B
  cout << "A: " << endl << A << endl;

  double det = 0.;
  A.Sdet(det);
  cout << "Determinant: " << det << endl;
  // WARNING: A has changed!!
  cout << "A again: " << endl << A << endl;
  A = B; 

  A.Invert();
  cout << "A^-1: " << endl << A << endl;

  // check if this is really the inverse:
  cout << "A^-1 * B: " << endl << A * B << endl;

  return 0;
#endif
}

int test4() { 
#ifdef XXX
  SMatrix<double,3> A;
  A(0,0) = A(0,1) = A(1,0) = 1;
  A(1,1) = A(2,2) = 2;
  cout << " A: " << endl << A << endl;

  SVector<double,3> x(1,2,3);
  cout << "x: " << x << endl;

  // we add 1 to each component of x and A
  cout << " (x+1)^T * (A+1) * (x+1): " << Product(x+1,A+1) << endl;

  return 0;
#endif
}

int test5() { 
#ifdef XXX
  SMatrix<float,4,3> A;
  A(0,0) = A(0,1) = A(1,1) = A(2,2) = 4.;
  A(2,3) = 1.;
  cout << "A: " << endl << A << endl;
  SVector<float,4> x(1,2,3,4);
  cout << " x: " << x << endl;
  SVector<float,3> a(1,2,3);
  cout << " a: " << a << endl;
  SVector<float,4> y = x + A * a;
  //    SVector<float,4> y = A * a;
  cout << " y: " << y << endl;

  SVector<float,3> b = (x+1) * (A+1);
  cout << " b: " << b << endl;

  return 0;
#endif
}

int test6() { 
#ifdef XXX
  SMatrix<float,4,2> A;
  A(0,0) = A(0,1) = A(1,1) = A(2,0) = A(3,1) = 4.;
  cout << "A: " << endl << A << endl;
  
  SMatrix<float,2,3> S;
  S(0,0) = S(0,1) = S(1,1) = S(0,2) = 1.;
  cout << " S: " << endl << S << endl;
  
  SMatrix<float,4,3> C = A * S;
  cout << " C: " << endl << C << endl;
  
  return 0;
#endif
}

int test7() { 
#ifdef XXX
  SVector<float,3>    xv(4,4,4);
  SVector<float,3>    yv(5,5,5);
  SVector<float,2>    zv(64,64);
  SMatrix<float,2,3>  x; 
  x.Place_in_row(xv,0,0);
  x.Place_in_row(xv,1,0);
  SMatrix<float,2,3>  y;  
  y.Place_in_row(yv,0,0);
  y.Place_in_row(yv,1,0);
  SMatrix<float,2,3>  z;
  z.Place_in_col(zv,0,0);
  z.Place_in_col(zv,0,1);
  z.Place_in_col(zv,0,2);

  // element wise multiplication
  cout << "x * y: " << endl << times(x, -y) << endl;

  x += z - y;
  cout << "x += z - y: " << endl << x << endl;

  // element wise square root
  cout << "sqrt(z): " << endl << sqrt(z) << endl;

  // element wise multiplication with constant
  cout << "2 * y: " << endl << 2 * y << endl;

  // a more complex expression
  cout << "fabs(-z + 3*x): " << endl << fabs(-z + 3*x) << endl;

  return 0;
#endif
}

int test8() { 
#ifdef XXX
  SMatrix<float,2,3> A;
  SVector<float,3>    av1(5.,15.,5.);
  SVector<float,3>    av2(15.,5.,15.);
  A.Place_in_row(av1,0,0);
  A.Place_in_row(av2,1,0);
  
  cout << "A: " << endl << A << endl;

  SVector<float,3>    x(1,2,3);
  SVector<float,3>    y(4,5,6);

  cout << "dot(x,y): " << Dot(x,y) << endl;

  cout << "mag(x): " << Mag(x) << endl;

  cout << "cross(x,y): " << Cross(x,y) << endl;

  cout << "unit(x): " << Unit(x) << endl;

  SVector<float,3>    z(4,16,64);
  cout << "x + y: " << x+y << endl;

  cout << "x * y: " << x * -y << endl;
  x += z - y;
  cout << "x += z - y: " << x << endl;

  // element wise square root
  cout << "sqrt(z): " << sqrt(z) << endl;

  // element wise multiplication with constant
  cout << "2 * y: " << 2 * y << endl;

  // a more complex expression
  cout << "fabs(-z + 3*x): " << fabs(-z + 3*x) << endl;

  SVector<float,4> a;
  SVector<float,2> b(1,2);
  a.Place_at(b,2);
  cout << "a: " << a << endl;
#endif


  return 0;
}

int test9() { 
  // test non mutating inversions
  SMatrix<double,3> A;
  A(0,0) = A(0,1) = A(1,0) = 1;
  A(1,1) = A(2,2) = 2;


  double det = 0.;
  A.Det2(det);
  cout << "Determinant: " << det << endl;

  SMatrix<double,3> Ainv = A.Inverse();
  cout << "A^-1: " << endl << Ainv << endl;

  // check if this is really the inverse:
  cout << "A^-1 * A: " << endl << Ainv * A << endl;

  return 0;
}

int test10() { 
  // test slices
  int iret = 0;
  double d[9] = { 1,2,3,4,5,6,7,8,9};
  SMatrix<double,3> A( d,d+9);

  cout << "A: " << A << endl;

  SVector<double,2> v23 = A.SubRow<2>( 0,1);    
  SVector<double,2> v69 = A.SubCol<2>( 2,1);    

  std::cout << " v23 =  " << v23 << " \tv69 = " << v69 << std::endl;
  iret |= compare( Dot(v23,v69),(2*6+3*9) ); 
  
  SMatrix<double,2,2> subA1 = A.Sub<2,2>( 1,0);
  SMatrix<double,2,3> subA2 = A.Sub<2,3>( 0,0);
  std::cout << " subA1 =  " << subA1 << " \nsubA2 = " << subA2 << std::endl;
  iret |= compare ( subA1(0,0), subA2(1,0)); 
  iret |= compare ( subA1(0,1), subA2(1,1)); 



  SVector<double,3> diag = A.Diagonal();
  std::cout << " diagonal =  " << diag << std::endl; 
  iret |= compare( Mag2(diag) , 1+5*5+9*9 ); 


  SMatrix<double,3> B = Transpose(A);
  std::cout << " B = " << B << std::endl;

#ifdef UNSUPPORTED_TEMPLATE_EXPRESSION
  // in this case  function is templated. Need to pass the 6 
  SVector<double,6> vU = A.UpperBlock<6>();
  SVector<double,6> vL = B.LowerBlock<6>();
#else 
  // standards
  SVector<double,6> vU = A.UpperBlock();
  SVector<double,6> vL = B.LowerBlock();
#endif
  std::cout << " vU =  " << vU << " \tvL = " << vL << std::endl;
  // need to test mag since order can change
  iret |= compare( Mag(vU), Mag(vL) ); 

  // test subvector
  SVector<double,3> subV = vU.Sub<3>(1);
  std::cout << " sub vU =  " << subV << std::endl;

  iret |= compare( vU[2], subV[1] ); 
  
  // test constructor from subVectors
  SMatrix<double,3> C(vU);
  SMatrix<double,3> D(vL,true);

  std::cout << " C =  " << C << std::endl;
  std::cout << " D =  " << D << std::endl;

  iret |= compare( C==D, true ); 

 
  return iret;
}


int test11() { 

  int iret = 0;
  double dSym[15] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  double d3[10] = {1,2,3,4,5,6,7,8,9,10};
  double d2[10] = {10,1,4,5,8,2,3,6,7,9};

  SVector<double,15> vsym(dSym,15);
  
  SMatrix<double,5,5> m1(vsym);
  SMatrix<double,2,5> m2(d2,d2+10);
  SMatrix<double,5,2> m3(d3,d3+10);
  SMatrix<double,5,5> I; 

  SMatrix<double,5,5> m32 = m3*m2;

  SMatrix<double,5,5> m4 = m1 -  m32 * m1; 
  SMatrix<double,5,5> m5 = m3*m2;
  // in smatrix this should not be done since a temporary object storing 
  //  m5 * m1 first is not computed 
  m5 =  m1 - m5 * m1;

  // this works probably becuse here multiplication is done first
  
  SMatrix<double,5,5> m6 = m3*m2;
  m6 =  - m6 * m1 + m1;

  
  std::cout << m4 << std::endl;
  std::cout << m5 << std::endl;
  //  std::cout << m6 << std::endl;

  // this is test will fail because operation is done at the same time
  iret |= compare( m4==m5, false ); 
  iret |= compare( m4==m6, true ); 



  return iret;
}


#define TEST(N)                                                                 \
  itest = N;                                                                    \
  if (test##N() == 0) std::cout << " Test " << itest << "  OK " << std::endl;   \
  else  {std::cout << " Test " << itest << "  Failed " << std::endl;             \
  return -1;}  



int main(void) {

  int itest;
  TEST(1);
  TEST(2);
  TEST(3);
  TEST(4);
  TEST(5);
  TEST(6);
  TEST(7);
  TEST(8);
  TEST(9);
  TEST(10);
  TEST(11);

  return 0;
}
