/**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , FNAL LCG ROOT MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// RotationApplication.cpp
//
// Created by:  M. Fischler, Aug 10, 2005
//
// Tests that each Rotation produces correct results when applied to 
// each form of vector in each oordinate system, and incidently that products
// of rotations work properly.
//
// The strategy is to build up sequences of rotations about the X, Y, and Z 
// axes, such that we can easily determine the correct vector answer.
//
// =================================================================

#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/Polar3D.h"
#include "Math/GenVector/CylindricalEta3D.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/EulerAngles.h"
#include "Math/GenVector/Quaternion.h"
#include "Math/GenVector/RotationZYX.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

#include "Math/GenVector/VectorUtil.h"

#include "Math/Vector3Dfwd.h"

#include "CoordinateTraits.h"
#include "RotationTraits.h"

#include <iostream>
#include <limits>
#include <cmath>
#include <vector>

using std::sin;
using std::cos;

//#define TRACE2
//#define TRACE1
//#define DEBUG

using namespace ROOT::Math;

#ifndef __CINT__

template <typename T1, typename T2 > 
struct Precision { 
  enum { result = std::numeric_limits<T1>::digits <= std::numeric_limits<T2>::digits   }; 
}; 

template <typename T1, typename T2, bool>
struct LessPreciseType {
  typedef T1 type;
};
template <typename T1, typename T2>
struct LessPreciseType<T1, T2, false> {
  typedef T2 type;
};


const double machEpsilon = pow (2.0, -52.0);

template <typename Scalar1, typename Scalar2>
int
closeEnough (Scalar1 s1, Scalar2 s2, std::string const & coord, double ticks) {
  int ret = 0;
  int pr = std::cout.precision(18);
  Scalar1 eps1 = std::numeric_limits<Scalar1>::epsilon();
  Scalar2 eps2 = std::numeric_limits<Scalar2>::epsilon();
  typedef typename LessPreciseType<Scalar1, Scalar2,Precision<Scalar1,Scalar2>::result>::type Scalar;
  Scalar epsilon = (eps1 >= eps2) ? eps1 : eps2;
  Scalar ss1 (s1);
  Scalar ss2 (s2);
  Scalar diff = ss1 - ss2;
  if (diff < 0) diff = -diff;
  if ( diff > ticks*epsilon ) {
    ret=3;
    std::cout << "\n\n????????\n\nAbsolute discrepancy in " << coord << "(): "
              << ss1 << " != " << ss2 << "\n"
              << "   (Allowed discrepancy is " << ticks
	      << " ticks = " << ticks*epsilon
              << ")\nDifference is " << diff/epsilon << " ticks\n";
  }
  std::cout.precision (pr);
  return ret;
}

template <class V1, class V2>
int compare3D (const V1 & v1, const V2 & v2, double ticks) {
  int ret =0;
  typedef typename V1::CoordinateType CoordType1;
  typedef typename V2::CoordinateType CoordType2;

  ret |= closeEnough ( v1.x(),     v2.x(),     "x"     ,ticks);
  ret |= closeEnough ( v1.y(),     v2.y(),     "y"     ,ticks);
  ret |= closeEnough ( v1.z(),     v2.z(),     "z"     ,ticks);
 
  if (ret != 0) {
    std::cout << "Discrepancy detected (see above) is between:\n  "
              << CoordinateTraits<CoordType1>::name() << " and\n  "
              << CoordinateTraits<CoordType2>::name() << "\n"
              << "with v = (" << v1.x() << ", " << v1.y() << ", " << v1.z()
              << ")\nv1 is " << v1 
	      << "\nv2 is " << v2 << "\n\n";
  }

  return ret;
}

template <class V> struct correctedTicks {
  double operator()(double ticks, const V& /*v */ , const XYZVector & /* ans */) 
  { 
    return ticks; 
  }
};

double correctTicks (double ticks,double z,double r) {
  double e = ticks*std::fabs( z*z / (r*r-z*z) );
  if (e < ticks) return ticks;
  return e;
}

template<> struct
correctedTicks< DisplacementVector3D < CylindricalEta3D<double> > > {
  double operator()(double ticks, 
  	const DisplacementVector3D< CylindricalEta3D<double> >& v,
	const XYZVector & ans) {
  double t1 = correctTicks (ticks,   v.z(),   v.r());
  double t2 = correctTicks (ticks, ans.z(), ans.r());
  return t1 > t2 ? t1 : t2;
  }
};
  
template<> struct
correctedTicks< PositionVector3D < CylindricalEta3D<double> > > {
  double operator()(double ticks, 
  	const PositionVector3D< CylindricalEta3D<double> >& v,
	const XYZVector & ans) {
  double t1 = correctTicks (ticks,   v.z(),   v.r());
  double t2 = correctTicks (ticks, ans.z(), ans.r());
  return t1 > t2 ? t1 : t2;
  }
};
  
template <class R, class V>
int testApplication(const R& r,const V& v,const XYZVector &answer,double t) {

  typedef typename V::CoordinateType CoordType;

  int ret = 0;
  correctedTicks<V> ct;
  double ticks = ct(t, v, answer);
  #ifdef DEBUG
  std::cout <<">>>>> Testing application of " 
            << RotationTraits<R>::name() << " " << r << "\non " 
            << CoordinateTraits<typename V::CoordinateType>::name() << 
	    v << " ticks = " << ticks; 
  #endif
 	#ifdef TRACE2
   	std::cout << "  about to do V rv = r(v) - \n";
	#endif
  V rv = r(v);
 	#ifdef TRACE2
   	std::cout << "ok ";
	#endif
        // comparison here should be == but need to use 10 ticks to be sure for 32 bits machines
        // when results are flushed in memory and approximated
        if ( compare3D(rv, r*v, 10 ) != 0) { 
     std::cout << "Inconsistency between R(v) and R*v for R = " 
               << RotationTraits<R>::name() << " " << r 
               << "\nacting on "  << CoordinateTraits<CoordType>::name() << v << "\n";
     ret |= 9;
  }
 	#ifdef TRACE2
   	std::cout << "+ also did rv != r*v ";
	#endif
  if ( closeEnough(v.r(), rv.r(), "r", ticks) != 0 ) {
    std::cout << "Radius change  between R(v) and R*v for R = "
              << RotationTraits<R>::name() << " " << r 
              << "\nacting on "   
	      << CoordinateTraits<CoordType>::name()
    	      << v << "\n";
    ret |= 17;
  }  
 	#ifdef TRACE2
   	std::cout << "\n---- about to do compare3D ----";
	#endif
  ret |= compare3D (rv, answer, ticks);
 	#ifdef TRACE2
   	std::cout << " done \n";
	#endif
  #ifdef DEBUG
  if (ret == 0) std::cout << " OK\n";
  #endif 
  return ret;
}


XYZVector rxv ( double phi, XYZVector v ) {
  double c = cos(phi);
  double s = sin(phi);
  return XYZVector ( v.x(), c*v.y()-s*v.z(), s*v.y()+c*v.z() );
}

XYZVector ryv ( double phi, XYZVector v ) {
  double c = cos(phi);
  double s = sin(phi);
  return XYZVector ( c*v.x()+s*v.z(), v.y(), -s*v.x()+c*v.z() );
}

XYZVector rzv ( double phi, XYZVector v ) {
  double c = cos(phi);
  double s = sin(phi);
  return XYZVector ( c*v.x()-s*v.y(), s*v.x()+c*v.y(), v.z() );
}

enum XYZ { X, Y, Z } ;

struct TestRotation {
  std::vector<XYZ> xyz;
  std::vector<double> phi;
  TestRotation (std::vector<XYZ> const & xyz_, std::vector<double> const & phi_)
  	: xyz(xyz_), phi(phi_) {}
};


Rotation3D rrr (TestRotation const & t) {
  	#ifdef TRACE1
   	std::cout << "---- rrr ----";
	#endif
  Rotation3D r;
  for (unsigned int i=0; i<t.xyz.size(); ++i) {
    switch ( t.xyz[i] ) {
      case X:  
        r = r*RotationX(t.phi[i]);
	break;
      case Y:
        r = r*RotationY(t.phi[i]);
	break;
      case Z:
        r = r*RotationZ(t.phi[i]);
	break;
    }
  }
 	#ifdef TRACE1
   	std::cout << " done\n";
	#endif 
  return r;
}

XYZVector ans ( TestRotation const & t,	XYZVector const & v_in) {
  XYZVector v(v_in);
  for (int i=t.xyz.size()-1; i >= 0; --i) {
    switch ( t.xyz[i] ) {
      case X:  
        v = rxv ( t.phi[i],v );
	break;
      case Y:
        v = ryv ( t.phi[i],v );
	break;
      case Z:
        v = rzv ( t.phi[i],v );
	break;
    }
  }
  return v;
}

const double pi = 3.1415926535897932385;

std::vector<TestRotation> 
makeTestRotations () {
  	#ifdef TRACE1
   	std::cout << "---- makeTestRotations ----";
	#endif
  std::vector<TestRotation> t;  
  std::vector<XYZ>    xyz;
  std::vector<double> phi;
  


  xyz.clear();      phi.clear();
  xyz.push_back(X); phi.push_back( pi/2 );
  t.push_back(TestRotation(xyz,phi));
 
  xyz.clear();      phi.clear();
  xyz.push_back(Y); phi.push_back( pi/2 );
  t.push_back(TestRotation(xyz,phi));
 
  xyz.clear();      phi.clear();
  xyz.push_back(Z); phi.push_back( pi/2 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(X); phi.push_back( -pi/6 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(Y); phi.push_back( pi/6 );
  t.push_back(TestRotation(xyz,phi));
 
  xyz.clear();      phi.clear();
  xyz.push_back(Z); phi.push_back( pi/3 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(X); phi.push_back( -pi/6 );
  xyz.push_back(Y); phi.push_back(  pi/3 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(X); phi.push_back( -pi/6 );
  xyz.push_back(Y); phi.push_back(  pi/4 );
  xyz.push_back(Z); phi.push_back( -pi/5 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(Y); phi.push_back( pi );
  xyz.push_back(X); phi.push_back( -pi/2 );
  xyz.push_back(Z); phi.push_back( -pi/1.5 );
  xyz.push_back(Y); phi.push_back( -pi/3 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(Z); phi.push_back(  1.3 );
  xyz.push_back(Y); phi.push_back( -1.1 );
  xyz.push_back(X); phi.push_back(  0.4 );
  xyz.push_back(Y); phi.push_back(  0.7 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(X); phi.push_back(  1.3 );
  xyz.push_back(Z); phi.push_back( -1.1 );
  xyz.push_back(Y); phi.push_back(  0.4 );
  xyz.push_back(Z); phi.push_back(  0.7 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(Y); phi.push_back(  1.3 );
  xyz.push_back(X); phi.push_back( -1.1 );
  xyz.push_back(Z); phi.push_back(  0.4 );
  xyz.push_back(X); phi.push_back(  0.7 );
  t.push_back(TestRotation(xyz,phi));

  xyz.clear();      phi.clear();
  xyz.push_back(Z); phi.push_back(  .03 );
  xyz.push_back(Y); phi.push_back( -.05 );
  xyz.push_back(X); phi.push_back(  0.04 );
  xyz.push_back(Y); phi.push_back(  0.07 );
  xyz.push_back(Z); phi.push_back( -0.02 );
  t.push_back(TestRotation(xyz,phi));

  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif
  return t;
}

std::vector<XYZVector> makeTestVectors () {
  	#ifdef TRACE1
   	std::cout << "---- makeTestVectors ----";
	#endif
  std::vector<XYZVector> vs;
  vs.push_back(XYZVector ( 1, 0, 0 ));
  vs.push_back(XYZVector ( 0, 1, 0 ));
  vs.push_back(XYZVector ( 0, 0, 1 ));
  vs.push_back(XYZVector ( -1, 0, 0 ));
  vs.push_back(XYZVector ( 0, -1, 0 ));
  vs.push_back(XYZVector ( 0, 0, -1 ));
  vs.push_back(XYZVector ( 1, 2, 3 ));
  vs.push_back(XYZVector ( 2, -1, 3 ));
  vs.push_back(XYZVector ( -3, 1, -2 ));
  vs.push_back(XYZVector ( 0, .00001, -2 ));

  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif
  return vs;
}

template <class R, class V>
int doTest (TestRotation const & testRotation, XYZVector const & testVector,
		double ticks) {
  	#ifdef TRACE1
   	std::cout << "---- doTest ----";
	#endif
  int ret = 0;
  R r ( rrr(testRotation) );
  V v(testVector);
  XYZVector rv = ans (testRotation, testVector);
  ret |= testApplication (r, v, rv, ticks);  
  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif

	if (ret == 0) std::cout << ".";

  return ret;
}

template <class R, class C>
int doTestL (TestRotation const & testRotation, XYZVector const & testVector,
		double ticks) {
  	#ifdef TRACE1
   	std::cout << "---- doTestL ----";
	#endif
  int ret = 0;
  R r ( rrr(testRotation) );
  LorentzVector<C> v;
  double x = testVector.X();
  double y = testVector.Y();
  double z = testVector.Z();
  double t = std::sqrt (x*x + y*y + z*z + 1);
  v.SetXYZT ( x, y, z, t );  
  XYZVector rv = ans (testRotation, testVector);
  ret |= testApplication (r, v, rv, ticks); 
  LorentzVector<C> rv2 = r(v);
  ret |= closeEnough (t, rv2.E(), "t", ticks); 
  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif
  return ret;
}

struct ForeignVector {
  typedef Cartesian3D<> CoordinateType;
  XYZVector v;
  template <class V> 
  explicit ForeignVector (V const & v_) : v(v_) {}
  ForeignVector (double xx, double yy, double zz) : v(xx,yy,zz) {}
  double x() const { return v.x(); }
  double y() const { return v.y(); }
  double z() const { return v.z(); }
  double r() const { return v.r(); }
  bool operator==(ForeignVector const & rhs) {return v == rhs.v;}
  bool operator!=(ForeignVector const & rhs) {return v != rhs.v;}
};
std::ostream & operator<< (std::ostream& os, const ForeignVector& v) {
  return os << v.v;
}


template <class R>
int doTestOfR (TestRotation const & testRotation, XYZVector const & testVector){
  	#ifdef TRACE1
   	std::cout << "---- doTestofR ----\n";
	#endif
  int ret = 0;
  const double ticks = 100;  // move from 32 to 100 
   #ifdef DEBUG
  std::cout << ">>>>> DisplacementVector3D< Cartesian3D<double> \n";
  #endif
  ret |= doTest <R, DisplacementVector3D< Cartesian3D<double> > >
  		(testRotation,testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> DisplacementVector3D< Polar3D<double> \n";
  #endif
  ret |= doTest <R, DisplacementVector3D< Polar3D<double> > >
  		(testRotation,testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> DisplacementVector3D< CylindricalEta3D<double> \n";
  #endif
  ret |= doTest <R, DisplacementVector3D< CylindricalEta3D<double> > >
  		(testRotation,testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> PositionVector3D< Cartesian3D<double> \n";
  #endif
  ret |= doTest <R, PositionVector3D< Cartesian3D<double> > >
  		(testRotation,testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> PositionVector3D< Polar3D<double> \n";
  #endif
  ret |= doTest <R, PositionVector3D< Polar3D<double> > >
  		(testRotation,testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> PositionVector3D< CylindricalEta3D<double> \n";
  #endif
  ret |= doTest <R, PositionVector3D< CylindricalEta3D<double> > >
  		(testRotation,testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> ForeignVector\n";
  #endif
  ret |= doTest <R, ForeignVector >
  		(testRotation,testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> LorentzVector<PxPyPzE4D<double> >\n";
  #endif
  ret |= doTestL <R, PxPyPzE4D<double> >
  		(testRotation,testVector,ticks);
  	#ifdef TRACE1
   	std::cout << " ---- doTestofR ---- done\n";
	#endif

	if (ret == 0) std::cout << ".";

  // TODO - other 4D coordinates

  return ret;
}


int exerciseTestCase (TestRotation const & testRotation, 
		      XYZVector const & testVector)      
{

  std::cout << ">>>>> Rotation Tests of " << testVector << "\t\t: " ; 

  	#ifdef TRACE1
   	std::cout << "---- exerciseTestCase ----";
	#endif
  int ret = 0;
  ret |= doTestOfR <Rotation3D>  (testRotation,testVector);
  ret |= doTestOfR <AxisAngle>   (testRotation,testVector);
  ret |= doTestOfR <EulerAngles> (testRotation,testVector);
  ret |= doTestOfR <Quaternion>  (testRotation,testVector);
  ret |= doTestOfR <RotationZYX> (testRotation,testVector);
  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif

  if (ret == 0) 
    std::cout << "\t OK\n"; 
  else {
    std::cout << "\t Failed!\n "; 
    std::cerr << "\n>>>>> Rotation Tests of " << testVector << "\t\t:\t FAILED \n";
  }

  return ret;
}

// ===== Axial test section =============

template <class R, class V>
int doTestA (XYZVector const & testVector, double ticks) {
  	#ifdef TRACE1
   	std::cout << "---- doTestA ----";
	#endif
  int ret = 0;
  V v(testVector);
  XYZVector rv;
  for (double angle = -4.0;  angle < 4.0; angle += .15) {
    RotationX rx (angle);
    rv = rxv (angle, testVector);
    ret |= testApplication (rx, v, rv, ticks);  
    RotationY ry (angle);
    rv = ryv (angle, testVector);
    ret |= testApplication (ry, v, rv, ticks);  
    RotationZ rz (angle);
    rv = rzv (angle, testVector);
    ret |= testApplication (rz, v, rv, ticks);  
  }
  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif
	if (ret == 0) std::cout << ".";
  return ret;
}

template <class R, class C>
int doTestLA (XYZVector const & testVector, double ticks) {
  	#ifdef TRACE1
   	std::cout << "---- doTestLA ----";
	#endif
  int ret = 0;
  LorentzVector<C> v;
  double x = testVector.X();
  double y = testVector.Y();
  double z = testVector.Z();
  double t = std::sqrt (x*x + y*y + z*z + 1);
  v.SetXYZT ( x, y, z, t );  
  XYZVector rv;
  for (double angle = -4.0;  angle < 4.0; angle += .15) {
    //std::cout << "\n============ angle is " << angle << "\n";
    RotationX rx (angle);
    rv = rxv (angle, testVector);
    ret |= testApplication (rx, v, rv, ticks);  
    RotationY ry (angle);
    rv = ryv (angle, testVector);
    ret |= testApplication (ry, v, rv, ticks);  
    RotationZ rz (angle);
    rv = rzv (angle, testVector);
    ret |= testApplication (rz, v, rv, ticks);  
  }
  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif

	if (ret == 0) std::cout << ".";
  return ret;
}

template <class R>
int doTestOfAxial (XYZVector const & testVector){
  	#ifdef TRACE1
   	std::cout << "---- doTestOfAxial ----\n";
	#endif
  int ret = 0;
  const double ticks = 32;
  #ifdef DEBUG
  std::cout << ">>>>> DisplacementVector3D< Cartesian3D<double> \n";
  #endif
  ret |= doTestA <R, DisplacementVector3D< Cartesian3D<double> > >
  		(testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> DisplacementVector3D< Polar3D<double> \n";
  #endif
  ret |= doTestA <R, DisplacementVector3D< Polar3D<double> > >
  		(testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> DisplacementVector3D< CylindricalEta3D<double> \n";
  #endif
  ret |= doTestA <R, DisplacementVector3D< CylindricalEta3D<double> > >
  		(testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> PositionVector3D< Cartesian3D<double> \n";
  #endif
  ret |= doTestA <R, PositionVector3D< Cartesian3D<double> > >
  		(testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> PositionVector3D< Polar3D<double> \n";
  #endif
  ret |= doTestA <R, PositionVector3D< Polar3D<double> > > (testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> PositionVector3D< CylindricalEta3D<double> \n";
  #endif
  ret |= doTestA <R, PositionVector3D< CylindricalEta3D<double> > >
  		(testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> ForeignVector\n";
  #endif
  ret |= doTestA <R, ForeignVector > (testVector,ticks);
  #ifdef DEBUG
  std::cout << ">>>>> LorentzVector<PxPyPzE4D<double> >\n";
  #endif
  ret |= doTestLA <R, PxPyPzE4D<double> > (testVector,ticks);
  	#ifdef TRACE1
   	std::cout << " ---- doTestofR ---- done\n";
	#endif
  // TODO - other 4D coordinates

	if (ret == 0) std::cout << ".";

   return ret;
}

int exerciseAxialTest (XYZVector const & testVector)      
{

  	#ifdef TRACE1
   	std::cout << "---- exerciseAxialTest ----";
	#endif

	std::cout << ">>>>> Axial Rotation Tests of " << testVector << "\t\t: "; 

  int ret = 0;
  ret |= doTestOfAxial <RotationX> (testVector);
  ret |= doTestOfAxial <RotationY> (testVector);
  ret |= doTestOfAxial <RotationZ> (testVector);
  	#ifdef TRACE1
   	std::cout << " done\n";
	#endif

  if (ret == 0) 
    std::cout << "\t OK\n"; 
  else {
    std::cout << "\t Failed!\n "; 
    std::cerr << "\n>>>>> Axial Rotation Tests of " << testVector << "\t\t:\t FAILED \n";
  }

  return ret;
}


#endif // endif on __CINT__

// ======================================


int rotationApplication () {
  int ret = 0;
  std::vector<TestRotation> testRotations = makeTestRotations();
  std::vector<XYZVector>    testVectors   = makeTestVectors();
  for ( std::vector<TestRotation>::const_iterator n =  testRotations.begin();
        n !=  testRotations.end(); ++n ) {
    for ( std::vector<XYZVector>::const_iterator m =  testVectors.begin();
      	m !=  testVectors.end(); ++m ) {
      ret |= exerciseTestCase (*n, *m);
    }
  }
  for ( std::vector<XYZVector>::const_iterator vp =  testVectors.begin();
        vp !=  testVectors.end(); ++vp ) {
    ret |= exerciseAxialTest (*vp);
  }

  return ret;
}

int main() { 
   int ret =  rotationApplication();
   if (ret)  std::cerr << "test FAILED !!! " << std::endl; 
   else   std::cout << "test OK " << std::endl;
   return ret; 
}
