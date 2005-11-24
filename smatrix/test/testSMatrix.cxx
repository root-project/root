#include <cmath>
#include "Math/SVector.h"
#include "Math/SMatrix.h"

#include <iostream>

using namespace ROOT::Math;

using std::cout;
using std::endl;


#define XXX

int test1() { 

  SVector<float,3> x(4,5,6);
    SVector<float,2> y(2,3);
    cout << "x: " << x << endl;
    cout << "y: " << y << endl;
    
    SMatrix<float,4,3> A;
    SMatrix<float,2,2> B;
    
    A.Place_in_row(y, 1, 1);
    A.Place_in_col(x + 2, 1, 0);
    A.Place_at(B , 2, 1);
    cout << "A: " << endl << A << endl;
    
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

  A.Sinvert();
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

  return 0;
}
