#include <iostream>
#include <cmath>
#include "Math/SVector.h"
#include "Math/SMatrix.h"

using namespace ROOT::Math;

using std::cout;
using std::endl;

int main(void) {

  SVector<float,3> x(4,5,6);
  SVector<float,2> y(2,3);
  cout << "x: " << x << endl;
  cout << "y: " << y << endl;

  SMatrix<float,4,3> A;
  SMatrix<float,2,2> B;

  A.place_in_row(y, 1, 1);
  A.place_in_col(x + 2, 1, 0);
  A.place_at(B , 2, 1);
  cout << "A: " << endl << A << endl;

  return 0;

#ifdef XXX
  SMatrix<double,3> A;
  A(0,0) = A(1,0) = 1;
  A(0,1) = 3;
  A(1,1) = A(2,2) = 2;
  cout << "A: " << endl << A << endl;

  SVector<double,3> x = A.row(0);
  cout << "x: " << x << endl;

  SVector<double,3> y = A.col(1);
  cout << "y: " << y << endl;

  return 0;
#endif

#ifdef XXX
  SMatrix<double,3> A;
  A(0,0) = A(0,1) = A(1,0) = 1;
  A(1,1) = A(2,2) = 2;

  SMatrix<double,3> B = A; // save A in B
  cout << "A: " << endl << A << endl;

  double det = 0.;
  A.sdet(det);
  cout << "Determinant: " << det << endl;
  // WARNING: A has changed!!
  cout << "A again: " << endl << A << endl;

  A.sinvert();
  cout << "A^-1: " << endl << A << endl;

  // check if this is really the inverse:
  cout << "A^-1 * B: " << endl << A * B << endl;

  return 0;
#endif

#ifdef XXX
  SMatrix<double,3> A;
  A(0,0) = A(0,1) = A(1,0) = 1;
  A(1,1) = A(2,2) = 2;
  cout << " A: " << endl << A << endl;

  SVector<double,3> x(1,2,3);
  cout << "x: " << x << endl;

  // we add 1 to each component of x and A
  cout << " (x+1)^T * (A+1) * (x+1): " << product(x+1,A+1) << endl;

  return 0;
#endif

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

#ifdef XXX
  SMatrix<float,2,3>  x = 4.;
  SMatrix<float,2,3>  y = 5.;
  SMatrix<float,2,3>  z = 64.;

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

#ifdef XXX
  SMatrix<float,2,3> A = 15.;
  A(0,0) = A(1,1) = A(0,2) = 5.;
  
  cout << "A: " << endl << A << endl;

  SVector<float,3>    x(1,2,3);
  SVector<float,3>    y(4,5,6);

  cout << "dot(x,y): " << dot(x,y) << endl;

  cout << "mag(x): " << mag(x) << endl;

  cout << "cross(x,y): " << cross(x,y) << endl;

  cout << "unit(x): " << unit(x) << endl;

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
  a.place_at(b,2);
  cout << "a: " << a << endl;
#endif

  for(unsigned int i=0; i<1000000; ++i) {
#ifdef XXX
    VtVector a(1.,2.,3.);
    VtVector b(4.,5.,6.);
    VtVector c(8.,9.,10.);

    VtVector d = a*(-1) + b;
    VtVector d = a + b + c;
#endif
#ifdef XXX
    VtMatrix A(4,3);
    A(0,0) = A(1,1) = A(2,2) = 4.;
    A(2,3) = 1.;
    VtVector a(1,2,3);
    //    cout << " a: " << a << endl;
    //    cout << " A: " << A << endl;
    //    VtVector x(1,2,3,4);
    //    cout << " x: " << x << endl;
    //    VtVector y = x + A * a;
    VtVector y = A * a;
    cout << " y: " << y << endl;
    exit(0);
#endif
#ifdef XXX
    // ==========================================
    VtMatrix A(3,3);
    A(0,0) = A(1,1) = A(2,2) = 4.;
    //    A(2,3) = 1.;
    //    cout << " A: " << -A << endl;
    VtMatrix B(3,3);
    B(0,1) = B(1,0) = B(1,1) = B(0,2) = 1.;
    //    cout << " B: " << B << endl;
    //    cout << " B: " << B << endl;
    VtMatrix C(3,3);
    C(0,2) = C(1,2) = C(2,2) = 2.;
    //    cout << " C: " << C << endl;
    //    VtMatrix D = B + C + (-A);
    VtMatrix D = A + B + C + A + B + C;
    cout << " D: " << D << endl;
    exit(0);
#endif
#ifdef XXX
    VtSymMatrix VA(3,2.);
    VA(0,0) = VA(0,1) = VA(1,0) = 1;
    //    cout << " A: " << VA << endl;
    //    cout << " det A: " << VA.det() << endl;
    VA.det();
    VA.VtDsinv();
    cout << " A^-1: " << VA << endl;
#endif
#ifdef XXX
    //===================================
    VtSymMatrix A(3,3.);
    A(0,0) = A(0,1) = A(1,0) = 1;
    //    cout << " A: " << VA << endl;
    VtSymMatrix B(3,3);
    B(0,1) = B(1,0) = B(1,1) = B(2,2) = 1.;
    //    cout << " B: " << B << endl;
    VtSymMatrix C(3,3);
    C(0,1) = C(1,0) = C(2,1) = C(0,2) = 2.;
    cout << " C: " << C << endl;
    VtVector    x(1,2,3);
    //    cout << " x: " << x << endl;
    VtVector    y(4,5,6);
    //    cout << " y: " << y << endl;
    (A+B+C).product(x+y);
    cout << " (x+y)^T * (A+B+C) * (x+y): " << (A+B+C).product(x+y) << endl;
    exit(0);
#endif
#ifdef XXX
    //    cout << " c: " << c << endl;
    SVector<float,3> a;
    a[0] = 1.; a[1] = 2.; a[2] = 3.;
    SVector<float,3> b;
    b[0] = 4.; b[1] = 5.; b[2] = 6.;
    SVector<float,3> c;
    c[0] = 8.; c[1] = 9.; c[2] = 10.;

    SVector<float,3> d = a + b + c;
    cout << "d: " << d << endl;
    exit(0);

    cout << "d: " << d << endl;
    d -= b + c;
    cout << "d -= b + c: " << d << endl;
    cout << "dot(a,b): " << dot(a,b) << endl;
    cout << "dot(a+b,c+d): " << dot(a+b,c+d) << endl;
    cout << "dot(a*b,c+d): " << dot(a*b,c+d) << endl;
    cout << "dot(a*b+c,c+d): " << dot(a*b+c,c+d) << endl;

    cout << "mag2(a) " << mag2(a) << endl;
    cout << "mag(a)  " << mag(a) << endl;
    cout << "mag2(a+b+c)" << mag2(a+b+c) << endl;
    cout << "mag(a+b+c) " << mag(a+b+c) << endl;
    //    mag2(a);
    //    mag2(a+b+c);

    //    SVector<float,3> d = a + static_cast<float>(3.);
    //    SVector<float,3> d = sqr(a + b + -20);
    //    SVector<float,3> d = (-a+b) * 3;
    SVector<float,3> d = a * (b + c);
    d = unit(a + b + c);
    //    d.unit();
    //    cout << "d unit: " << d << endl;
    //    cout << "d = -a + b: " << d << endl;
    //    cout << "a: " << a << " b: " << b << endl;
    //    cout << "mag2(3*a): " << mag2(3*a) << endl;
    //    cout << "cross(3*a,b+c): " << cross(3*a,b+c) << endl;
    //    cout << "dot(-a,c+d+3): " << dot(-a,c+d+3) << endl;
    //    cout << "mag2(-a+d): " << mag2(-a+d) << " mag(d): " << mag(-a+d) << endl;

    SMatrix<float,3,3> A = 15.;
    SMatrix<float,3,3> B = A;
    A(0,0) = A(1,1) = A(2,0) = A(2,2) = 5.;
    SMatrix<float,3,3> C = fabs(A + -B + 2);
//      cout << "A: " << endl << A << endl;
      cout << "C: " << endl << C << endl;
#endif

#ifdef XXX
    SMatrix<float,4,3> A;
    A(0,0) = A(1,1) = A(2,2) = 4.;
    A(2,3) = 1.;
    //    cout << "A: " << endl << A << endl;
    SVector<float,4> x(1,2,3,4);
    //    cout << " x: " << x << endl;
    SVector<float,3> a(1,2,3);
    //    cout << " a: " << a << endl;
    SVector<float,4> y = x + A * a;
    //    SVector<float,4> y = A * a;
    cout << " y: " << y << endl;
    exit(0);
#endif
#ifdef XXX
    // =======================================
    SMatrix<float,3,3> A;
    A(0,0) = A(1,1) = A(2,2) = 4.;
    //    cout << "A: " << endl << A << endl;
    //    cout << "B: " << endl << B << endl;
    SMatrix<float,3,3> B;
    B(0,1) = B(1,0) = B(1,1) = B(0,2) = 1.;
    //    cout << "B: " << endl << B << endl;
    //    SMatrix<float,3,3> C = times(A,B); // component wise multiplication
    SMatrix<float,3,3> C;
    C(0,2) = C(1,2) = C(2,2) = 2.;
    //    cout << "C: " << endl << C << endl;
    //    SMatrix<float,3,3> D = -A + B + C;
    SMatrix<float,3,3> D = A + B + C + A + B + C;
    cout << "D: " << endl << D << endl;
    exit(0);
    SMatrix<float,3,2> E = 4.;
    cout << "E: " << endl << E << endl;
    //    D = A + E;
#endif

#ifdef XXX
    SVector<float,3> b(4,5,6);
    cout << " x: " << x << endl;

    //    SVector<float,4> y = x + (A * 2) * (a + 1);
    SVector<float,3> y = (x+1) * (A+1);
    cout << " y: " << y << endl;

    SMatrix<float,3> S;
    S(0,0) = S(1,0) = S(2,0) = 1.;
    cout << " S: " << endl << S << endl;
    SMatrix<float,4,3> C = A * S;
    cout << " C: " << endl << C << endl;
#endif
#ifdef XXX
    SMatrix<double,3> A;
    A(0,0) = A(0,1) = A(1,0) = 1;
    A(1,1) = A(2,2) = 2;
    //    cout << "A: " << endl << A << endl;
    double det = 0.;
    A.sdet(det);
    cout << "Determinant: " << det << endl;
    cout << "A again: " << endl << A << endl;
    exit(0);
    A.sinvert();
    cout << "A^-1: " << endl << A << endl;
    exit(0);
#endif
#ifdef XXX
    SVector<double,3> x(1,2,3);
    cout << "x: " << x << endl;
    cout << " x^T * A * x: " << product(x+1,A+1) << endl;
    // product(A,x);

    SMatrix<double,3> B = 1;
    cout << "B: " << endl << B << endl;
    A /= B + 1;
    cout << "A/=B: " << endl << A << endl;

    SVector<double,3> y(4,5,6);
    cout << "y: " << y << endl;
    y /= x;
    cout << "y/=x: " << y << endl;
    exit(0);
#endif
#ifdef XXX
    //===================================
    SMatrix<float,3> A;
    A(0,0) = A(0,1) = A(1,0) = 1;
    A(1,1) = A(2,2) = 3.;
    //    cout << " A: " << endl << VA << endl;
    SMatrix<float,3> B;
    B(0,1) = B(1,0) = B(1,1) = B(2,2) = 1.;
    B(0,0) = 3;
    //    cout << " B: " << endl << B << endl;
    SMatrix<float,3> C;
    C(0,1) = C(1,0) = C(2,1) = C(0,2) = 2.;
    C(0,0) = C(1,1) = C(2,2) = 3;
    //    cout << " C: " << endl << C << endl;
    SVector<float,3>    x(1,2,3);
    //    cout << " x: " << x << endl;
    SVector<float,3>    y(4,5,6);
    //    cout << " y: " << y << endl;
    product(A+B+C,x+y);
    cout << " (x+y)^T * (A+B+C) * (x+y): " << product(A+B+C,x+y) << endl;
    exit(0);
#endif
#ifdef XXX
    SVector<float,4> x;
    SVector<float,2> y(1,2);
    //    cout << "y: " << y << endl;
    x.place_at(y,2);
    cout << "x: " << x << endl;
    SMatrix<float,4,3> A;
    SMatrix<float,2,2> B = 1;
    //    SVector<float,2>   x(1,2);
    //    A.place_in_row(x+2,1,1);
    //    A.place_in_col(x,1,1);
    A.place_at(B,0,1);
    cout << "A: " << endl << A << endl;
    exit(0);
#endif
#ifdef XXX
    VtVector x(4);
    VtVector y(1,2);
    //    cout << "y: " << y << endl;
    x.place_at(y,2);
    cout << "x: " << x << endl;
    exit(0);
#endif
#ifdef XXX
    SMatrix<float,4,3> A;
    SMatrix<float,2,2> B = 1;
    //    SVector<float,2>   x(1,2);
    //    A.place_in_row(x+2,1,1);
    //    A.place_in_col(x,1,1);
    A.place_at(B,0,1);
    cout << "A: " << endl << A << endl;
    exit(0);
#endif
#ifdef XXX
    VtMatrix A(4,3);
    VtMatrix B(2,2);
    B(0,0) = B(1,1) = 2.;
    A.place_at(B,0,1);
    cout << "A: " << A << endl;
    exit(0);
#endif
#ifdef XXX
    SMatrix<float,3> C;
    C(0,1) = C(1,0) = C(2,1) = C(0,2) = 2.;
    C(0,0) = C(1,1) = C(2,2) = 3;
    //    cout << " C: " << endl << C << endl;
    float det = 0.;
    //    Dfact<SMatrix<float,3>,3,3>(C,det);
    C.det(det);
    cout << "Dfact(C): " << det << endl;
    cout << "C after: " << endl << C << endl;
    exit(0);
#endif
#ifdef XXX
    SMatrix<float,4> C;
    C(0,1) = C(1,0) = C(2,1) = C(0,2) = C(3,1) = C(2,3) = 2.;
    C(0,0) = C(1,1) = C(2,2) = C(3,3) = 3;
//      SMatrix<float,4> B(C);
//      cout << " C: " << endl << C << endl;
    Dinv<SMatrix<float,4>,4,4>(C);
    cout << "C after: " << endl << C << endl;
    SMatrix<float,4> D = B * C;
    //    cout.setf(ios::fixed);
    cout << "D = B * C: " << endl << D << endl;
    exit(0);
#endif
#ifdef XXX
    SMatrix<float,2> C;
    C(0,0) = C(1,1) = C(0,1) = 5.;
    C(1,0) = 1;
    SMatrix<float,2> B(C);
    cout << " C: " << endl << C << endl;
    if(Invert<0>::Dinv(C) ) {
      cout << "C after: " << endl << C << endl;
      SMatrix<float,2> D = C * B;
      cout << " D: " << endl << D << endl;
    } else { cerr << " inversion failed! " << endl; }
    exit(0);
#endif
#ifdef XXX
    SMatrix<float,3> C;
    C(0,1) = C(1,1) = C(0,2) = C(2,2) = 5.;
    C(0,0) = 10;
    SMatrix<float,3> B(C);
    cout << " C: " << endl << C << endl;
    if(Invert<3>::Dinv(C)) {
      cout << "C after: " << endl << C << endl;
      SMatrix<float,3> D = B * C;
      cout << "D: " << endl << D << endl;
    } else { cerr << " inversion failed! " << endl; }
    exit(0);
#endif
#ifdef XXX
    SMatrix<float,4> C;
    C(0,1) = C(1,0) = C(2,1) = C(0,2) = 2;
    C(3,1) = 2;
    C(2,3) = 2.;
    C(0,0) = C(1,1) = C(2,2) = C(3,3) = 3;
    //    Invert<4>::Dinv<SMatrix<float,4>, 4>(C);
    //    C.invert();
    SMatrix<float,4> B(C);
    cout << " C: " << endl << C << endl;
    //    if(Invert<4>::Dinv(C)) {
    if(C.invert()) {
      cout << "C after: " << endl << C << endl;
      SMatrix<float,4> D = B * C;
      cout.setf(ios::fixed);
      cout << "D: " << endl << D << endl;
    } else { cerr << " inversion failed! " << endl; }

    cout << "C+B: " << endl << C+B << endl;
    exit(0);
#endif
  }

  return 0;
}
