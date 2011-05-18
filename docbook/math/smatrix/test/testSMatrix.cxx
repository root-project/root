#include <cmath>
#include "Math/SVector.h"
#include "Math/SMatrix.h"

#include <iostream>
#include <vector>
#include <string>
#include <limits>


using namespace ROOT::Math;

using std::cout;
using std::endl;


//#define TEST_STATIC_CHECK  // for testing compiler failures (static check)

#define XXX

template<class T> 
int compare( T a, T b, const std::string & s="",double tol = 1) { 
  if (a == b) return 0; 
  double eps = tol*8.*std::numeric_limits<T>::epsilon();

  if (a == 0 && std::abs(b) < eps ) return 0; 
  if (b == 0 && std::abs(a) < eps ) return 0; 
  if (std::abs(a-b) < a*eps) return 0; 
  if ( s =="" ) 
    std::cout << "\nFailure " << a << " different than " << b << std::endl;
  else 
    std::cout << "\n" << s << " : Failure " << a << " different than " << b << std::endl;
  return 1;
}

int compare( int a, int b, const std::string & s="") { 
  if (a == b) return 0; 
  if ( s =="" ) 
    std::cout << "\nFailure " << a << " different than " << b << std::endl;
  else 
    std::cout << "\n" << s << " : Failure " << a << " different than " << b << std::endl;
  return 1;
}
int compare( bool a, bool b, const std::string & s="") { 
  return compare(static_cast<int>(a), static_cast<int>(b),s); 
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
  A.Place_in_col(x, 1, 0);
  A.Place_in_col(x + 2, 1, 0);
  A.Place_in_row(y + 3, 1, 1);
  

#ifndef _WIN32
  A.Place_at(B , 2, 1);
#else
  //Windows need template parameters
  A.Place_at<2,2>(B , 2, 1);
#endif
  cout << "A: " << endl << A << endl;

  SVector<float,3> z(x+2);
  z.Place_at(y, 1);
  z.Place_at(y+3, 1);
  cout << "z: " << endl << z << endl;

  
#ifdef TEST_STATIC_CHECK
  // create a vector of size 2 from 3 arguments
  SVector<float, 2> vbad(1,2,3);
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

  SVector<float, 4> sp1(m,4);
  SVector<float, 4> sp2(sm.begin(),sm.end());
  SMatrix<float, 2,2> sm2(vm.begin(),vm.end());

  if ( sp1 != sp2) { cout << "Test STL interface for SVector failed" << endl; return -1; }
  if ( sm2 != sm) { cout << "Test STL interface for SMatrix failed" << endl; return -1; }


  // test construction from identity
  SMatrix<float,3,3> i3 = SMatrixIdentity(); 

  cout << "3x3 Identity\n" << i3 << endl;

  SMatrix<float,2,3> i23 = SMatrixIdentity(); 
  cout << "2x3 Identity\n" << i23 << endl;

  SMatrix<float,3,3,MatRepSym<float,3> > is3 = SMatrixIdentity(); 
  cout << "Sym matrix Identity\n" << is3 << endl;


  // test operator = from identity
  A = SMatrixIdentity();
  cout << "4x3 Identity\n" << A << endl;

  std::vector<float> v(6);
  for (int i = 0; i <6; ++i) v[i] = double(i+1);
  SMatrix<float,3,3,MatRepSym<float,3> > s3(v.begin(), v.end() );
  cout << s3 << endl;

    
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
  A.Det(det);
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
  cout << " (x+1)^T * (A+1) * (x+1): " << Similarity(x+1,A+1) << endl;

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

  cout << "x\n" << x << "y\n" << y << "z\n" << z << endl;

  // element wise multiplication
  cout << "x * (- y) : " << endl << Times(x, -y) << endl;

  x += z - y;
  cout << "x += z - y: " << endl << x << endl;

  // element wise square root
  cout << "sqrt(z): " << endl << sqrt(z) << endl;

  // element wise multiplication with constant
  cout << "2 * y: " << endl << 2 * y << endl;

  // a more complex expression (failure on Win32)
#ifndef _WIN32
  //cout << "fabs(-z + 3*x): " << endl << fabs(-z + 3*x) << endl;
  cout << "fabs(3*x -z): " << endl << fabs(3*x -z) << endl;
#else 
  // doing directly gives internal compiler error on windows
  SMatrix<float,2,3>  ztmp = 3*x - z; 
  cout << " fabs(-z+3*x) " << endl << fabs(ztmp) << endl;
#endif

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

  cout << "x + y(0) " << (x+y)(0) << endl;

  cout << "x * -y: " << x * -y << endl;
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

  SVector<float,3> x2 = Square(x); 
  std::cout << x2 << std::endl;


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

  int ifail; 
  SMatrix<double,3> Ainv = A.Inverse(ifail);
  if (ifail) { 
    cout << "inversion failed\n";
    return -1;
  } 
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

  SVector<double,2> v23 = A.SubRow<SVector<double,2> >( 0,1);    
  SVector<double,2> v69 = A.SubCol<SVector<double,2> >( 2,1);    

  std::cout << " v23 =  " << v23 << " \tv69 = " << v69 << std::endl;
  iret |= compare( Dot(v23,v69),double(2*6+3*9) ); 
  
//    SMatrix<double,2,2> subA1 = A.Sub<2,2>( 1,0);
//    SMatrix<double,2,3> subA2 = A.Sub<2,3>( 0,0);
  SMatrix<double,2,2> subA1 = A.Sub< SMatrix<double,2,2> > ( 1,0);
  SMatrix<double,2,3> subA2 = A.Sub< SMatrix<double,2,3> > ( 0,0);
  std::cout << " subA1 =  " << subA1 << " \nsubA2 = " << subA2 << std::endl;
  iret |= compare ( subA1(0,0), subA2(1,0)); 
  iret |= compare ( subA1(0,1), subA2(1,1)); 



  SVector<double,3> diag = A.Diagonal();
  std::cout << " diagonal =  " << diag << std::endl; 
  iret |= compare( Mag2(diag) , double(1+5*5+9*9) ); 
  iret |= compare( A.Trace() , double(1+5+9) ); 


  SMatrix<double,3> B = Transpose(A);
  std::cout << " B = " << B << std::endl;

#ifdef UNSUPPORTED_TEMPLATE_EXPRESSION
  // in this case  function is templated. Need to pass the 6 
  SVector<double,6> vU = A.UpperBlock< SVector<double,6> >();
  SVector<double,6> vL = B.LowerBlock< SVector<double,6> >();
#else 
  // standards
  SVector<double,6> vU = A.UpperBlock();
  SVector<double,6> vL = B.LowerBlock();
#endif
  std::cout << " vU =  " << vU << " \tvL = " << vL << std::endl;
  // need to test mag since order can change
  iret |= compare( Mag(vU), Mag(vL) ); 

  // test subvector
  SVector<double,3> subV = vU.Sub< SVector<double,3> >(1);
  std::cout << " sub vU =  " << subV << std::endl;

  iret |= compare( vU[2], subV[1] ); 
  
  // test constructor from subVectors
  SMatrix<double,3> C(vU,false);
  SMatrix<double,3> D(vL,true);

//   std::cout << " C =  " << C << std::endl;
//   std::cout << " D =  " << D << std::endl;

  iret |= compare( static_cast<int>(C==D), 1 ); 

  SMatrix<double,3, 3, MatRepSym<double,3> > C2(vU,false);
  SMatrix<double,3, 3, MatRepSym<double,3> > D2(vL,true);

  iret |= compare( static_cast<int>(C==C2), 1 ); 
  iret |= compare( static_cast<int>(D==D2), 1 ); 


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
  //SMatrix<double,5,5> I; 

  SMatrix<double,5,5> m32 = m3*m2;

  SMatrix<double,5,5> m4 = m1 -  m32 * m1; 
  SMatrix<double,5,5> m5 = m3*m2;

  // now thanks to the IsInUse() function this should work 
  m5 =  m1 - m5 * m1;

  // this works probably becuse here multiplication is done first
  
  SMatrix<double,5,5> m6 = m3*m2;
  //m6 =  - m6 * m1 + m1;
  // this does not work if use reference in binary and unary operators , better this
  m6 =  - m32 * m1 + m1;

  
  std::cout << m4 << std::endl;
  std::cout << m5 << std::endl;
  std::cout << m6 << std::endl;

  // this is test will now work  
  iret |= compare( m4==m5, true ); 
  iret |= compare( m4==m6, true ); 



  return iret;
}


int test12() {
  // test of symmetric matrices

  int iret = 0; 
  SMatrix<double,2,2,MatRepSym<double,2> >  S; 
  S(0,0) = 1.233;
  S(0,1) = 2.33; 
  S(1,1) = 3.45; 
  std::cout << "S\n" << S << std::endl;

  double a = 3; 
  std::cout << "S\n" << a * S << std::endl;

  // test inversion
  int ifail1 = 0; 
  //int ifail2 = 0; 
  SMatrix<double,2,2,MatRepSym<double,2> > Sinv1 = S.Inverse (ifail1);  
  //SMatrix<double,2,2,MatRepSym<double,2> > Sinv2 = S.Sinverse(ifail2);
  //std::cout << "Inverse:  S-1 " <<  Sinv1 << "\nifail=" << ifail1 << std::endl;
  //std::cout << "Sinverse: S-1"  << Sinv2 << "\nifail=" << ifail2 << std::endl;

  SMatrix<double,2> IS1 = S*Sinv1;
  //SMatrix<double,2> IS2 = S*Sinv2;
  double d1 = std::sqrt( IS1(1,0)*IS1(1,0) + IS1(0,1)*IS1(0,1) ); 
  //double d2 = std::sqrt( IS2(1,0)*IS2(1,0) + IS2(0,1)*IS2(0,1) ); 


  iret |= compare( d1 < 1E-6,true,"inversion1" ); 
  //iret |= compare( d2 < 1E-6,true,"inversion2" ); 
  

  SMatrix<double,2,3  >  M1 =  SMatrixIdentity(); 
  M1(0,1) = 1; M1(1,0) = 1; M1(0,2) = 1; M1(1,2) = 1;
  SMatrix<double,2,3  >  M2 =  SMatrixIdentity(); 
  SMatrix<double,2,2,MatRepSym<double,2> >  S2=S; 
  // S2 -= M1*Transpose(M2);  // this should fails to compile
  SMatrix<double,2 > mS2(S2);  
  mS2 -= M1*Transpose(M2);
  //std::cout << "S2=S-M1*M2T\n" << mS2 << std::endl;
  iret |= compare( mS2(0,1), S(0,1)-1 ); 
  mS2 += M1*Transpose(M2); 
  iret |= compare( mS2(0,1), S(0,1) ); 
 
  //std::cout << "S2+=M1*M2T\n" << mS2 << std::endl;


  SMatrix<float,100,100,MatRepSym<float,100> >  mSym; 
  SMatrix<float,100 >  m; 
  //std::cout << " Symmetric matrix size: " << sizeof(mSym) << std::endl; 
  //std::cout << " Normal    matrix size: " << sizeof( m  ) << std::endl; 

  SMatrix<float,100,100,MatRepSym<float,100> >  mSym2; 
  //std::cout << " Symmetric matrix size: " << sizeof(mSym2) << std::endl; 


  return iret; 
} 

int test13() { 
  // test of operation with a constant; 

  int iret = 0; 

  int a = 2;
  float b = 3;


  SVector<double,2> v(1,2); 
  SVector<double,2> v2= v + a; 
  iret |= compare( v2[1], v[1]+a ); 
  SVector<double,2> v3= a + v; 
  iret |= compare( v3[1], v[1]+a ); 
  iret |= compare( v3[0], v2[0] ); 

  v2 = v - a; 
  iret |= compare( v2[1], v[1]-a ); 
  v3 = a - v; 
  iret |= compare( v3[1], a - v[1] ); 

  // test now with expression
  v2 = b*v + a; 
  iret |= compare( v2[1], b*v[1]+a ); 
  v3 = a + v*b; 
  iret |= compare( v3[1], b*v[1]+a ); 
  v2 = v*b - a; 
  iret |= compare( v2[1], b*v[1]-a ); 
  v3 = a - b*v; 
  iret |= compare( v3[1], a - b*v[1] ); 

  v2 = a * v/b; 
  iret |= compare( v2[1], a*v[1]/b ); 

  SVector<double,2> p(1,2); 
  SVector<double,2> q(3,4);
  v = p+q; 
  v2 = a*(p+q);
  iret |= compare( v2[1], a*v[1] ); 
  v3 = (p+q)*b;
  iret |= compare( v3[1], b*v[1] ); 
  v2 = (p+q)/b;
  iret |= compare( v2[1], v[1]/b ); 

  //std::cout << "tested vector -constant operations : v2 = " << v2 << " v3 = " << v3 << std::endl;

  // now test the matrix (normal)

  SMatrix<double,2,2> m; 
  m.Place_in_row(p,0,0);
  m.Place_in_row(q,1,0);

  SMatrix<double,2,2> m2,m3; 

  m2= m + a; 
  iret |= compare( m2(1,0), m(1,0)+a ); 
  m3= a + m; 
  iret |= compare( m3(1,0), m(1,0)+a ); 
  iret |= compare( m3(0,0), m2(0,0) ); 

  m2 = m - a; 
  iret |= compare( m2(1,0), m(1,0)-a ); 
  m3 = a - m; 
  iret |= compare( m3(1,0), a - m(1,0) ); 

  // test now with expression
  m2 = b*m + a; 
  iret |= compare( m2(1,0), b*m(1,0)+a ); 
  m3 = a + m*b; 
  iret |= compare( m3(1,0), b*m(1,0)+a ); 
  m2 = m*b - a; 
  iret |= compare( m2(1,0), b*m(1,0)-a ); 
  m3 = a - b*m; 
  iret |= compare( m3(1,0), a - b*m(1,0) ); 

  m2 = a * m/b; 
  iret |= compare( m2(1,0), a*m(1,0)/b ); 

  SMatrix<double,2> u = m; 
  SMatrix<double,2> w; 
  w(0,0) = 5; w(0,1) = 6; w(1,0)=7; w(1,1) = 8;

  m = u+w; 
  m2 = a*(u+w);
  iret |= compare( m2(1,0), a*m(1,0) ); 
  m3 = (u+w)*b;
  iret |= compare( m3(1,0), b*m(1,0) ); 
  m2 = (u+w)/b;
  iret |= compare( m2(1,0), m(1,0)/b ); 
  
  //std::cout << "tested general matrix -constant operations :\nm2 =\n" << m2 << "\nm3 =\n" << m3 << std::endl;

  // now test the symmetric matrix 

  SMatrix<double,2,2,MatRepSym<double,2> > s; 
  s(0,0) = 1; s(1,0) = 2; s(1,1) = 3; 

  SMatrix<double,2,2,MatRepSym<double,2> > s2,s3; 

  s2= s + a; 
  iret |= compare( s2(1,0), s(1,0)+a ); 
  s3= a + s; 
  iret |= compare( s3(1,0), s(1,0)+a ); 
  iret |= compare( s3(0,0), s2(0,0) ); 

  s2 = s - a; 
  iret |= compare( s2(1,0), s(1,0)-a ); 
  s3 = a - s; 
  iret |= compare( s3(1,0), a - s(1,0) ); 


  // test now with expression
  s2 = b*s + a; 
  iret |= compare( s2(1,0), b*s(1,0)+a ); 
  s3 = a + s*b; 
  iret |= compare( s3(1,0), b*s(1,0)+a ); 
  s2 = s*b - a; 
  iret |= compare( s2(1,0), b*s(1,0)-a ); 
  s3 = a - b*s; 
  iret |= compare( s3(1,0), a - b*s(1,0) ); 

  s2 = a * s/b; 
  iret |= compare( s2(1,0), a*s(1,0)/b ); 


  SMatrix<double,2,2,MatRepSym<double,2> > r = s; 
  SMatrix<double,2,2,MatRepSym<double,2> > t; 
  t(0,0) = 4; t(0,1) = 5; t(1,1) = 6;

  s = r+t; 
  s2 = a*(r+t);
  iret |= compare( s2(1,0), a*s(1,0),"a*(r+t)" ); 
  s3 = (t+r)*b;
  iret |= compare( s3(1,0), b*s(1,0), "(t+r)*b" ); 
  s2 = (r+t)/b;
  iret |= compare( s2(1,0), s(1,0)/b, "(r+t)/b" ); 

  //std::cout << "tested sym matrix -constant operations :\ns2 =\n" << s2 << "\ns3 =\n" << s3 << std::endl;


  return iret; 
}


int test14() { 
  // test place_at (insertion) of all type of matrices

  int iret = 0; 

  // test place at with sym matrices 

  SMatrix<double,2,2,MatRepSym<double,2> >  S; 
  S(0,0) = 1;
  S(0,1) = 2; 
  S(1,1) = 3; 

  double u[6] = {1,2,3,4,5,6};
  SMatrix<double,2,3,MatRepStd<double,2,3> >  U(u,u+6); 

  //place general matrix in general matrix  
  SMatrix<double,4,4> A; 
  A.Place_at(U,1,0); 
  //std::cout << "Test general matrix placed in general at 1,0 :\nA=\n" << A << std::endl; 
  iret |= compare( A(1,0),U(0,0) );
  iret |= compare( A(1,1),U(0,1) );
  iret |= compare( A(2,1),U(1,1) );
  iret |= compare( A(2,2),U(1,2) );

  A.Place_at(-2*U,1,0);
  iret |= compare( A(1,0),-2*U(0,0) );
  iret |= compare( A(1,1),-2*U(0,1) );
  iret |= compare( A(2,1),-2*U(1,1) );
  iret |= compare( A(2,2),-2*U(1,2) );


  //place symmetric in general (should work always)
  A.Place_at(S,0,0); 
  //std::cout << "Test symmetric matrix placed in general at 0,0:\nA=\n" << A << std::endl; 
  iret |= compare( A(0,0),S(0,0) );
  iret |= compare( A(1,0),S(0,1) );
  iret |= compare( A(1,1),S(1,1) );

  A.Place_at(2*S,0,0); 
  iret |= compare( A(0,0),2*S(0,0) );
  iret |= compare( A(1,0),2*S(0,1) );
  iret |= compare( A(1,1),2*S(1,1) );


  A.Place_at(S,2,0); 
  //std::cout << "A=\n" << A << std::endl; 
  iret |= compare( A(2,0),S(0,0) );
  iret |= compare( A(3,0),S(0,1) );
  iret |= compare( A(3,1),S(1,1) );


  SMatrix<double,3,3,MatRepSym<double,3> >  B; 

  //place symmetric in symmetric (OK for col=row) 
  B.Place_at(S,1,1);
  //std::cout << "Test symmetric matrix placed in symmetric at 1,1:\nB=\n" << B << std::endl; 
  iret |= compare( B(1,1),S(0,0) );
  iret |= compare( B(2,1),S(0,1) );
  iret |= compare( B(2,2),S(1,1) );

  B.Place_at(-S,0,0);
  //std::cout << "B=\n" << B << std::endl; 
  iret |= compare( B(0,0),-S(0,0) );
  iret |= compare( B(1,0),-S(0,1) );
  iret |= compare( B(1,1),-S(1,1) );


  //this should assert at run time
  //B.Place_at(S,1,0); 
  //B.Place_at(2*S,1,0); 

  //place general in symmetric should fail to compiler
#ifdef TEST_STATIC_CHECK
  B.Place_at(U,0,0);
  B.Place_at(-U,0,0);
#endif

  // test place vector in matrices
  SVector<double,2> v(1,2);

  A.Place_in_row(v,1,1); 
  iret |= compare( A(1,1),v[0] );
  iret |= compare( A(1,2),v[1] );
  A.Place_in_row(2*v,1,1); 
  iret |= compare( A(1,1),2*v[0] );
  iret |= compare( A(1,2),2*v[1] );

  A.Place_in_col(v,1,1); 
  iret |= compare( A(1,1),v[0] );
  iret |= compare( A(2,1),v[1] );
  A.Place_in_col(2*v,1,1); 
  iret |= compare( A(1,1),2*v[0] );
  iret |= compare( A(2,1),2*v[1] );
  
  //place vector in sym matrices  
  B.Place_in_row(v,0,1); 
  //std::cout << "B=\n" << B << std::endl;
  iret |= compare( B(0,1),v[0] );
  iret |= compare( B(1,0),v[0] );
  iret |= compare( B(2,0),v[1] );
  B.Place_in_row(2*v,1,1); 
  iret |= compare( B(1,1),2*v[0] );
  iret |= compare( B(2,1),2*v[1] );

  B.Place_in_col(v,1,0); 
  //std::cout << "B=\n" << B << std::endl;
  iret |= compare( B(0,1),v[0] );
  iret |= compare( B(1,0),v[0] );
  iret |= compare( B(0,2),v[1] );
  B.Place_in_col(2*v,1,1); 
  iret |= compare( B(1,1),2*v[0] );
  iret |= compare( B(1,2),2*v[1] );


  // test Sub 
  SMatrix<double,2,2,MatRepSym<double,2> > sB = B.Sub<SMatrix<double,2,2,MatRepSym<double,2> > > (1,1); 
  iret |= compare( sB(0,0),B(1,1) );
  iret |= compare( sB(1,0),B(1,2) );
  iret |= compare( sB(1,1),B(2,2) );

  SMatrix<double,2,3,MatRepStd<double,2,3> > sA = A.Sub<SMatrix<double,2,3,MatRepStd<double,2,3> > > (1,0); 
  iret |= compare( sA(0,0),A(1,0) );
  iret |= compare( sA(1,0),A(2,0) );
  iret |= compare( sA(1,1),A(2,1) );
  iret |= compare( sA(1,2),A(2,2) );

  sA = B.Sub<SMatrix<double,2,3,MatRepStd<double,2,3> > > (0,0); 
  iret |= compare( sA(0,0),B(0,0) );
  iret |= compare( sA(1,0),B(1,0) );
  iret |= compare( sA(0,1),B(0,1) );
  iret |= compare( sA(1,1),B(1,1) );
  iret |= compare( sA(1,2),B(1,2) );

  //this should run assert
  //  sA = A.Sub<SMatrix<double,2,3,MatRepStd<double,2,3> > > (0,2); 
  //  sB = B.Sub<SMatrix<double,2,2,MatRepSym<double,2> > > (0,1);

#ifdef TEST_STATIC_CHECK
  sB = A.Sub<SMatrix<double,2,2,MatRepSym<double,2> > > (0,0);
  SMatrix<double,5,2> tmp1 = A.Sub<SMatrix<double,5,2> >(0,0); 
  SMatrix<double,2,5> tmp2 = A.Sub<SMatrix<double,2,5> >(0,0); 
#endif


  // test setDiagonal
  
#ifdef TEST_STATIC_CHECK
  SVector<double,3> w(-1,-2,3);
  sA.SetDiagonal(w);
  sB.SetDiagonal(w);
#endif

  sA.SetDiagonal(v);
  iret |= compare( sA(1,1),v[1] );
  sB.SetDiagonal(v);
  iret |= compare( sB(0,0),v[0] );
  
  // test Trace
  iret |= compare( sA.Trace(), v[0]+v[1]);
  iret |= compare( sB.Trace(), v[0]+v[1]);
  SMatrix<double,3,2> sAt = Transpose(sA);
  iret |= compare( sAt.Trace(), v[0]+v[1]);


  return iret;
}



int test15() { 
  // test using iterators 
  int iret = 0; 

  double u[12] = {1,2,3,4,5,6,7,8,9,10,11,12}; 
  double w[6] = {1,2,3,4,5,6};

  SMatrix<double,3,4> A1(u,12);
  iret |= compare( A1(0,0),u[0] );
  iret |= compare( A1(1,2),u[6] );
  iret |= compare( A1(2,3),u[11] );
  //cout << A1 << endl;

  SMatrix<double,3,4> A2(w,6,true,true);
  iret |= compare( A2(0,0),w[0] );
  iret |= compare( A2(1,0),w[1] );
  iret |= compare( A2(2,0),w[3] );
  iret |= compare( A2(2,2),w[5] );
  //cout << A2 << endl;

  // upper diagonal (needs 9 elements)
  SMatrix<double,3,4> A3(u,9,true,false);
  iret |= compare( A3(0,0),u[0] );
  iret |= compare( A3(0,1),u[1] );
  iret |= compare( A3(0,2),u[2] );
  iret |= compare( A3(1,2),u[5] );
  iret |= compare( A3(2,3),u[8] );
  //cout << A3 << endl;


  //cout << "test sym matrix " << endl;
  SMatrix<double,3,3,MatRepSym<double,3> > S1(w,6,true); 
  iret |= compare( S1(0,0),w[0] );
  iret |= compare( S1(1,0),w[1] );
  iret |= compare( S1(1,1),w[2] );
  iret |= compare( S1(2,0),w[3] );
  iret |= compare( S1(2,1),w[4] );
  iret |= compare( S1(2,2),w[5] );

  SMatrix<double,3,3,MatRepSym<double,3> > S2(w,6,true,false); 
  iret |= compare( S2(0,0),w[0] );
  iret |= compare( S2(1,0),w[1] );
  iret |= compare( S2(2,0),w[2] );
  iret |= compare( S2(1,1),w[3] );
  iret |= compare( S2(2,1),w[4] );
  iret |= compare( S2(2,2),w[5] );

  // check retrieve
  double * pA1 = A1.begin();
  for ( int i = 0; i< 12; ++i) 
    iret |= compare( pA1[i],u[i] );

  double * pS1 = S1.begin();
  for ( int i = 0; i< 6; ++i) 
    iret |= compare( pS1[i],w[i] );


  // check with SetElements
  std::vector<double> vu(u,u+12);
  std::vector<double> vw(w,w+6);
  SMatrix<double,3,4> B1;
  B1.SetElements(vu.begin(),10); 
  iret |= compare( B1(0,0),u[0] );
  iret |= compare( B1(1,2),u[6] );
  iret |= compare( B1(2,3),0.0 );

  B1.SetElements(vu.begin(),vu.end()); 
  iret |= compare( B1(0,0),vu[0] );
  iret |= compare( B1(1,2),vu[6] );
  iret |= compare( B1(2,3),vu[11] );

  B1.SetElements(vw.begin(),vw.end(),true,true); 
  iret |= compare( B1(0,0),w[0] );
  iret |= compare( B1(1,0),w[1] );
  iret |= compare( B1(2,0),w[3] );
  iret |= compare( B1(2,2),w[5] );

  SVector<double,12> v1; 
  v1.SetElements(vu.begin(),vu.end() ); 
  for (unsigned int i = 0; i < v1.kSize; ++i) 
     iret |= compare( v1[i],vu[i] );

  // v1.SetElements(vw.begin(),vw.end() ); // this assert at run-time
  v1.SetElements(vw.begin(), vw.size() );
  for (unsigned int i = 0; i < vw.size(); ++i) 
     iret |= compare( v1[i],vw[i] );



  return iret;
}

int test16() { 
  // test IsInUse() function  to create automatically temporaries 
  int iret = 0; 

  double a[6] = {1,2,3,4,5,6};
  double w[9] = {10,2,3,4,50,6,7,8,90};
  
  SMatrix<double,3,3,MatRepSym<double,3> > A(a,a+6); 
  SMatrix<double,3,3,MatRepSym<double,3> > B;
//   SMatrix<double,3,3,MatRepSym<double,3> > C; 

  B = Transpose(A);
  A = Transpose(A);
  iret |= compare( A==B,true,"transp");
  
  SMatrix<double,3 > W(w,w+9); 
  SMatrix<double,3 > Y = W.Inverse(iret);
  SMatrix<double,3 > Z; 
  Z = W *  Y; 
  Y = W *  Y; 
#ifndef _WIN32
  // this fails on Windows (bad calculations)
  iret |= compare( Z==Y,true,"mult");
#else 
  for (int i = 0; i< 9; ++i) { 
    // avoid small value of a 
    double a = Z.apply(i); 
    double eps = std::numeric_limits<double>::epsilon();
    if (a < eps) a = 0; 
    iret |= compare(a,Y.apply(i),"index");
  }
#endif

  Z = (A+W)*(B+Y); 
  Y = (A+W)*(B+Y); 

  iret |= compare( Z==Y,true,"complex mult");


  // test of assign sym
//   AssignSym::Evaluate(A,  W * A * Transpose(W)  );
//   AssignSym::Evaluate(B,  W * A * Transpose(W)  );
//   iret |= compare( A==B,true,"assignsym");
  

  return iret; 
}


int test17() { 
  int iret =0;
  // tets tensor product
  SVector<double,2> v1(1,2); 
  SVector<double,3> v2(1,2,3); 

  SMatrix<double,2,3> m = TensorProd(v1,v2); 
  for (int i = 0; i < m.kRows ; ++i) 
     for (int j = 0; j < m.kCols ; ++j) 
        iret |= compare(m(i,j),v1(i)*v2(j) );
  //std::cout << "Tensor Product \n" << m << std::endl;

  SVector<double,4> a1(2,-1,3,4); 
  SVector<double,4> a2(5,6,1,-2); 
  
  SMatrix<double,4> A = TensorProd(a1,a2); 
  double r1 = Dot(a1, A * a2 ); 
  double r2 = Dot(a1, a1) * Dot(a2,a2 );
  iret |= compare(r1,r2,"tensor prod");

  A = TensorProd(2.*a1,a2); 
  r1 = Dot(a1, A * a2 )/2; 
  r2 = Dot(a1, a1) * Dot(a2,a2 );
  iret |= compare(r1,r2,"tensor prod");


  A = TensorProd(a1,2*a2); 
  r1 = Dot(a1, A * a2 )/2; 
  r2 = Dot(a1, a1) * Dot(a2,a2 );
  iret |= compare(r1,r2,"tensor prod");

  A = TensorProd(0.5*a1,2*a2); 
  r1 = Dot(a1, A * a2 ); 
  r2 = Dot(a1, a1) * Dot(a2,a2 );
  iret |= compare(r1,r2,"tensor prod");
 
  return iret;
}

// test inverison of large matrix (double) 
int test18() { 
  int iret =0;
  // data for a 7x7 sym matrix to invert
  SMatrix<double,7,7,MatRepSym<double,7> > S;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j <= i; ++j) { 
      if (i == j) 
	S(i,j) = 10*double(std::rand())/(RAND_MAX); // generate between 0,10
      else
	S(i,j) = 2*double(std::rand())/(RAND_MAX)-1; // generate between -1,1
    }
  }
  int ifail = 0;
  SMatrix<double,7,7,MatRepSym<double,7> > Sinv = S.Inverse(ifail);
  iret |= compare(ifail,0,"sym7x7 inversion");
  SMatrix<double,7> Id = S*Sinv; 
  for (int i = 0; i < 7; ++i)
    iret |= compare(Id(i,i),1.,"inv result");

  double sum = 0; 
  for (int i = 0; i < 7; ++i) 
    for (int j = 0; j <i; ++j)
      sum+= std::fabs(Id(i,j) );  // sum of off diagonal elements

  iret |= compare(sum < 1.E-10, true,"inv off diag");

  // now test inversion of general 
  SMatrix<double,7> M;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) { 
      if (i == j) 
	M(i,j) = 10*double(std::rand())/(RAND_MAX); // generate between 0,10
      else
	M(i,j) = 2*double(std::rand())/(RAND_MAX)-1; // generate between -1,1
    }
  }
  ifail = 0;
  SMatrix<double,7 > Minv = M.Inverse(ifail);
  iret |= compare(ifail,0,"7x7 inversion");
  Id = M*Minv; 
  for (int i = 0; i < 7; ++i)
    iret |= compare(Id(i,i),1.,"inv result");

  sum = 0; 
  for (int i = 0; i < 7; ++i) 
    for (int j = 0; j <i; ++j)
      sum+= std::fabs(Id(i,j) );  // sum of off diagonal elements

  iret |= compare(sum < 1.E-10, true,"inv off diag");
  
  
  return iret; 
}

// test inversion of large matrices (float)
int test19() { 
  int iret =0;
  // data for a 7x7 sym matrix to invert
  SMatrix<float,7,7,MatRepSym<float,7> > S;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j <= i; ++j) { 
      if (i == j) 
	S(i,j) = 10*float(std::rand())/(RAND_MAX); // generate between 0,10
      else
	S(i,j) = 2*float(std::rand())/(RAND_MAX)-1; // generate between -1,1
    }
  }
  int ifail = 0;
  SMatrix<float,7,7,MatRepSym<float,7> > Sinv = S.Inverse(ifail);
  iret |= compare(ifail,0,"sym7x7 inversion");
  SMatrix<float,7> Id = S*Sinv; 

  //std::cout << S << "\n" << Sinv << "\n" << Id << "\n";

  for (int i = 0; i < 7; ++i)
    iret |= compare(Id(i,i),float(1.),"inv sym result");

  double sum = 0; 
  for (int i = 0; i < 7; ++i) 
    for (int j = 0; j <i; ++j)
      sum+= std::fabs(Id(i,j) );  // sum of off diagonal elements

  iret |= compare(sum < 1.E-5, true,"inv sym off diag");

  // now test inversion of general 
  SMatrix<float,7> M;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) { 
      if (i == j) 
	M(i,j) = 10*float(std::rand())/(RAND_MAX); // generate between 0,10
      else
	M(i,j) = 2*float(std::rand())/(RAND_MAX)-1; // generate between -1,1
    }
  }
  ifail = 0;
  SMatrix<float,7 > Minv = M.Inverse(ifail);
  iret |= compare(ifail,0,"7x7 inversion");
  Id = M*Minv; 

  //std::cout << M << "\n" << Minv << "\n" << Id << "\n";

  for (int i = 0; i < 7; ++i)
    iret |= compare(Id(i,i),float(1.),"inv result");

  sum = 0; 
  for (int i = 0; i < 7; ++i) 
    for (int j = 0; j <i; ++j)
      sum+= std::fabs(Id(i,j) );  // sum of off diagonal elements

  iret |= compare(sum < 1.E-5, true,"inv off diag");
  
  
  return iret; 
}


int test20() { 
// test operator += , -= 
  int iret =0;
  //std::cout.precision(18); 


  double d1[6]={1,2,3,4,5,6};
  double d2[6]={1,2,5,3,4,6};

  SMatrix<double,2>    m1_0(d1,4);
  SMatrix<double,2 >   m2_0(d2,4);
  SMatrix<double,2>    m1 = m1_0;
  SMatrix<double,2 >   m2 = m2_0;
  SMatrix<double,2>    m3;
 

  m3 = m1+m2; 
  m1+= m2; 
 
  if (iret) std::cout << "m1+= m2" << m1  << std::endl;

  iret |= compare(m1==m3,true); 

  m3 = m1 + 3;
  m1+= 3;
  iret |= compare(m1==m3,true); 
  if (iret)std::cout << "m1 + 3\n" << m1 << " \n  " << m3  << std::endl;

  m3 = m1 - m2; 
  m1-= m2; 
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1-= m2\n" << m1 << " \n  " << m3  << std::endl;
  
  m3 = m1 - 3; 
  m1-= 3; 
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1-= 3\n" << m1 << " \n " << m3  << std::endl;


  m3 = m1*2;
  m1*= 2; 
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1*= 2\n" << m1 << "\n" << m3  << std::endl;

  // matrix multiplication (*= works only for squared matrix mult.) 
  m3 = m1*m2; 
  m1*= m2; 
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1*= m2\n" << m1 << " \n " << m3  << std::endl;

  m3 = m1/2;
  m1/= 2; 
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1/=2\n" << m1 << " \n " << m3  << std::endl;

  // test mixed with a scalar 
  double a = 2; 
  m3 = m2 + a*m1; 
  m2 += a*m1; 
  iret |= compare(m2==m3,true); 
  if (iret) std::cout << "m2 += a*m1\n" << m2 << "\n  " << m3  << std::endl;


  // more complex op (passing expressions)

  m1 = m1_0;
  m2 = m2_0;


  m3 = m1 + (m1 * m2);
  m1 += m1 * m2;
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1 += m1*m2\n" << m1 << "\n  " << m3  << std::endl;

  m3 = m1 - (m1 * m2);
  m1 -= m1 * m2;
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1 -= m1*m2\n" << m1 << " \n " << m3  << std::endl;

  m3 = m1 * (m1 * m2);
  m1 *= m1 * m2;
  iret |= compare(m1==m3,true); 
  if (iret) std::cout << "m1 *= m1*m2\n" << m1 << "\n  " << m3  << std::endl;

  // test operation involving 2 expressions
  // (check bug 35076)

  // reset initial matrices to avoid numerical problems
  m1 = m1_0;
  m2 = m2_0;

  m3 = m1+m2;
  SMatrix<double,2>    m4;
  SMatrix<double,2>    m5;
  m4 = (m1*m2) + (m1*m3);
  m5 = m1*m2;
  m5 += m1*m3;
  iret |= compare(m4==m5,true); 
  if (iret) std::cout << "m5 = m1*m3\n" << m4 << "\n  " << m5  << std::endl;


  m4 = (m1*m2) - (m1*m3);
  m5 = m1*m2;
  m5 -= m1*m3;
  iret |= compare(m4==m5,true); 
  if (iret) std::cout << "m5 -= m1*m3\n" << m4 << "\n  " << m5  << std::endl;


  m4 = (m1+m2) * (m1-m3);
  m5 = m1+m2;
  m5 = m5 * (m1-m3);
  iret |= compare(m4==m5,true); 

  if (iret) std::cout << "m5= m5*(m1-m3) \n"  << m4 << " \n " << m5  << std::endl;


  // test with vectors 
  SVector<double,4>    v1(d1,4);
  SVector<double,4 >   v2(d2,4);
  SVector<double,4 >   v3;

  v3 = v1+v2;
  v1 += v2; 
  iret |= compare(v1==v3,true); 

  v3 = v1 + 2; 
  v1 += 2;
  iret |= compare(v1==v3,true); 

  v3 = v1+ (v1 + v2);
  v1 +=  v1 + v2;
  iret |= compare(v1==v3,true); 

  v3 = v1 - v2;
  v1 -= v2; 
  iret |= compare(v1==v3,true); 

  v3 = v1 - 2; 
  v1 -= 2;
  iret |= compare(v1==v3,true); 

  v3 = v1 - (v1 + v2);
  v1 -=  v1 + v2;
  iret |= compare(v1==v3,true); 

  v3 = v1 * 2;
  v1 *= 2;
  iret |= compare(v1==v3,true); 

  v3 = v1 / 2;
  v1 /= 2;
  iret |= compare(v1==v3,true); 

  // test with symmetric matrices 

  //double d1[6]={1,2,3,4,5,6};
  SMatrix<double,3,3,MatRepSym<double,3> >   ms1(d1,d1+6);
  SMatrix<double,3,3,MatRepSym<double,3> >   ms2(d1,d1+6,true, false);
  SMatrix<double,3,3,MatRepSym<double,3> >   ms3; 
  SMatrix<double,3,3,MatRepSym<double,3> >   ms4; 

  // using complex expressions
  ms3 = ms1 + (ms1 + ms2);
  ms1 += ms1 + ms2;
  iret |= compare(ms1==ms3,true); 

  ms3 = ms1 - (ms1 + ms2);
  ms1 -= ms1 + ms2;
  iret |= compare(ms1==ms3,true); 


  a = 2; 
  ms3 = ms1 + a*ms2; 

  ms4 = ms1; 
  ms4 += a*ms2; 

  iret |= compare(ms3==ms4,true); 

  ms3 = ms1 - a*ms2; 
  ms4 = ms1; 
  ms4 -= a*ms2; 
  iret |= compare(ms3==ms4,true); 

  return iret; 
}

int test21() { 

   // test global matrix function (element-wise operations)

  int iret =0;

  double d1[4]={4,6,3,4};
  double d2[4]={2,3,1,4};

  SMatrix<double,2>    m1(d1,4);
  SMatrix<double,2 >   m2(d2,4);
  SMatrix<double,2>    m3;

  // test element-wise multiplication
  m3 = Times(m1,m2);
  for (int i = 0; i < 4; ++i)
     iret |= compare(m3.apply(i),m1.apply(i)*m2.apply(i)); 

  // matrix division is element-wise division
  m3 = Div(m1,m2);
  for (int i = 0; i < 4; ++i)
     iret |= compare(m3.apply(i),m1.apply(i)/m2.apply(i)); 


  return iret; 

}


int test22() { 

   // test conversion to scalar for size 1 matrix and vectors 

  int iret =0;
  SMatrix<double,1> m1(2); 
  iret |= compare(m1(0,0),2.); 

  SVector<double,1> v1;
  v1 = 2; 
  iret |= compare(m1(0,0),2.); 

  return iret; 
}

int test23() { 
   // test cholesky inversion and solving 
   int iret = 0; 

   double m[] = { 100, .15, 2.3, 0.01, .01,  1.}; 
   SMatrix<double, 3, 3, MatRepSym<double, 3> >  smat(m, m+6); 

   //std::cout << "Perform inversion  of matrix \n" << smat << std::endl; 

   int ifail = 0; 
   SMatrix<double, 3, 3, MatRepSym<double, 3> > imat = smat.InverseChol(ifail); 
   iret |= compare(ifail==0, true, "inversion");

   // test max deviations from identity for m = imat * smat

   SMatrix<double, 3> mid = imat * smat; 
   int n = 3; 
   double prod = 1; 
   double vmax = 0; 
   for (int i = 0; i < n; ++i) { 
      for (int j = 0; j < n; ++j) { 
         if (i == j) 
            prod *= mid(i,i); 
         else { 
            if (std::abs (mid(i,j)) > vmax ) vmax = std::abs( mid(i,j) ); 
         }
      }
   } 
   iret |= compare(prod, 1., "max dev diagonal");
   iret |= compare(vmax, 0., "max dev offdiag ",10);

   // test now solving of linear system
   SVector<double, 3> vec(1,2,3); 

   SVector<double, 3> x = SolveChol( smat, vec, ifail); 

   //std::cout << "linear system solution " << x << std::endl;

   iret |= compare( (ifail==0), true, "solve chol"); 

   SVector<double, 3> v2 = smat * x;  

   for (int i = 0; i < 3; ++i) 
      iret |= compare(v2[i], vec[i], "v2 ==vec"); 
   
   return iret; 

}

int test24() { 
   // add transpose test
   // see bug #65531
   double a[9] = { 1,-2,3,4,-5,6,-7,8,9};
   double b[9] = { 1,-1,0,0,2,0,-1,0,3};

   SMatrix<double,3> A(a,a+9); 
   SMatrix<double,3> B(b,b+9); 

   SMatrix<double,3> R = A * B * Transpose(A);

   SMatrix<double,3> temp1 = A * B; 
   SMatrix<double,3> R1 = temp1 * Transpose(A);

   SMatrix<double,3> temp2 = B * Transpose(A); 
   SMatrix<double,3> R2 = A * temp2;

   int iret = 0; 
   iret |= compare(R1 == R2, true);
   iret |= compare(R == R1,true);
   return iret; 

}

#define TEST(N)                                                                 \
  itest = N;                                                                    \
  if (test##N() == 0) std::cerr << " Test " << itest << "  OK " << std::endl; \
  else { std::cerr << " Test " << itest << "  FAILED " << std::endl;    \
     iret +=1; };




int testSMatrix() {

  int iret = 0; 
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
  TEST(12);
  TEST(13);
  TEST(14);
  TEST(15);
  TEST(16);
  TEST(17);
  TEST(18);
  TEST(19);
  TEST(20);
  TEST(21);
  TEST(22);
  TEST(23);
  TEST(24);

  return iret;
}

int main() { 
   int ret = testSMatrix();
   if (ret)  std::cerr << "test SMatrix:\t  FAILED !!! " << std::endl; 
   else   std::cerr << "test SMatrix: \t OK " << std::endl;
   return ret; 
}
