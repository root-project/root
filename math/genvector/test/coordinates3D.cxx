// $Id $
//
// Tests that each form of vector has all the properties that stem from
// owning and forwarding to a coordinates instance, and that they give proper
// results.
//
// 3D vectors have:
//
// accessors x(), y(), z(), r(), theta(), phi(), rho(), eta()
//
// =================================================================

#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/Polar3D.h"
#include "Math/GenVector/CylindricalEta3D.h"
#include "Math/GenVector/Cylindrical3D.h"
#include "Math/GenVector/etaMax.h"

#include "Math/Vector3Dfwd.h"

#include "CoordinateTraits.h"

#include <iostream>
#include <limits>
#include <cmath>
#include <vector>

//#define TRACE1
#define DEBUG

using namespace ROOT::Math;




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


template <typename Scalar1, typename Scalar2>
int
closeEnough ( Scalar1 s1, Scalar2 s2, std::string const & coord, double ticks ) {
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
  if (ss1 == 0 || ss2 == 0) { // TODO - the ss2==0 makes a big change??
    if ( diff > ticks*epsilon ) {
      ret=3;
      std::cout << "\nAbsolute discrepancy in " << coord << "(): "
                << ss1 << " != " << ss2 << "\n"
                << "   (Allowed discrepancy is " << ticks*epsilon
                << ")\nDifference is " << diff/epsilon << " ticks\n";
    }
    std::cout.precision (pr);
    return ret;
  }
  // infinity dicrepancy musy be checked with max precision
  long double sd1(ss1); 
  long double sd2(ss2); 
  if ( (sd1 + sd2 == sd1) != (sd1 + sd2 == sd2) ) {
    ret=5;
    std::cout << "\nInfinity discrepancy in " << coord << "(): "
	      << sd1 << " != " << sd2 << "\n";
    std::cout.precision (pr);
    return ret;
  }
  Scalar denom = ss1 > 0 ? ss1 : -ss1;
  if ((diff/denom > ticks*epsilon) && (diff > ticks*epsilon)) {
    ret=9;
    std::cout << "\nDiscrepancy in " << coord << "(): "
              << ss1 << " != " << ss2 << "\n"
              << "   (Allowed discrepancy is " << ticks*epsilon*ss1
              << ")\nDifference is " << (diff/denom)/epsilon << " ticks\n";
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
  ret |= closeEnough ( v1.rho(),   v2.rho(),   "rho"   ,ticks);
  // case in phi that difference is close to 2pi
  typedef typename  V2::Scalar Scalar; 
  Scalar phi2 = v2.phi();
  if (std::abs(v1.phi()- phi2 ) > ROOT::Math::Pi() ) { 
     if (phi2<0) 
        phi2 += 2.*ROOT::Math::Pi();
     else 
        phi2 -= 2*ROOT::Math::Pi();
  }
  ret |= closeEnough ( v1.phi(),   phi2,   "phi"   ,ticks);
  ret |= closeEnough ( v1.r(),     v2.r(),     "r"     ,ticks);
  ret |= closeEnough ( v1.theta(), v2.theta(), "theta" ,ticks);
  ret |= closeEnough ( v1.mag2(),  v2.mag2(),  "mag2"  ,ticks);
  ret |= closeEnough ( v1.perp2(), v2.perp2(), "perp2" ,ticks);
  if ( v1.rho() > 0 && v2.rho() > 0 ) { // eta can legitimately vary if rho == 0
    ret |= closeEnough ( v1.eta(), v2.eta(),   "eta"   ,ticks);
  }

  if (ret != 0) {
    std::cout << "\nDiscrepancy detected (see above) is between:\n  "
              << CoordinateTraits<CoordType1>::name() << " and\n  "
              << CoordinateTraits<CoordType2>::name() << "\n"
              << "with v = (" << v1.x() << ", " << v1.y() << ", " << v1.z()
              << ")\n\n\n";
  }
  else { 
#ifdef DEBUG
    std::cout << ".";
#endif
  }


  return ret;
}

template <class C>
int test3D ( const DisplacementVector3D<C> & v, double ticks ) {

#ifdef DEBUG
  std::cout <<"\n>>>>> Testing 3D from " << v << " ticks = " << ticks << std::endl;
#endif

  int ret = 0;
  DisplacementVector3D< Cartesian3D<double> > vxyz_d (v.x(), v.y(), v.z());

  double r = std::sqrt (v.x()*v.x() + v.y()*v.y() + v.z()*v.z());
  double rho = std::sqrt (v.x()*v.x() + v.y()*v.y());
  double z   = v.z();
//   double theta = r>0 ? std::acos ( z/r ) : 0;
//   if (std::abs( std::abs(z) - r) < 10*r* std::numeric_limits<double>::epsilon() ) 
   double  theta = std::atan2( rho, z );  // better when theta is small or close to pi
  
  double phi = rho>0 ? std::atan2 (v.y(), v.x()) : 0;
  DisplacementVector3D< Polar3D<double> > vrtp_d ( r, theta, phi );

  double eta;
  if (rho != 0) {
    eta = -std::log(std::tan(theta/2));
        #ifdef TRACE1
        std::cout <<  ":::: rho != 0\n"
                  <<  ":::: theta = << " << theta
                  <<"/n:::: tan(theta/2) = " << std::tan(theta/2)
                  <<"\n:::: eta = " << eta << "\n";
        #endif
  } else if (v.z() == 0) {
    eta = 0;
        #ifdef TRACE1
        std::cout <<  ":::: v.z() == 0\n"
                  <<"\n:::: eta = " << eta << "\n";
        #endif
  } else if (v.z() > 0) {
    eta = v.z() + etaMax<long double>();
        #ifdef TRACE1
        std::cout <<  ":::: v.z() > 0\n"
                  <<  ":::: etaMax =  " << etaMax<long double>()
                  <<"\n:::: eta = " << eta << "\n";
        #endif
  } else {
    eta = v.z() - etaMax<long double>();
        #ifdef TRACE1
        std::cout <<  ":::: v.z() < 0\n"
                  <<  ":::: etaMax =  " << etaMax<long double>()
                  <<"\n:::: eta = " << eta << "\n";
        #endif
  }

#ifdef DEBUG
  std::cout << "           Testing DisplacementVector3D :  ";
#endif

  DisplacementVector3D< CylindricalEta3D<double> > vrep_d ( rho, eta, phi );

  ret |= compare3D( vxyz_d, vrtp_d, ticks);
  ret |= compare3D( vrtp_d, vxyz_d, ticks);
  ret |= compare3D( vxyz_d, vrep_d, ticks);
  ret |= compare3D( vrtp_d, vrep_d, ticks);

  DisplacementVector3D< Cylindrical3D<double> >   vrzp_d ( rho, z, phi );

  ret |= compare3D( vxyz_d, vrzp_d, ticks);
  ret |= compare3D( vrtp_d, vrzp_d, ticks);
  ret |= compare3D( vrep_d, vrzp_d, ticks);

  DisplacementVector3D< Cartesian3D<float> >      vxyz_f (v.x(), v.y(), v.z());
  DisplacementVector3D< Polar3D<float> >          vrtp_f ( r, theta, phi );
  DisplacementVector3D< CylindricalEta3D<float> > vrep_f ( rho, eta, phi );

  ret |= compare3D( vxyz_d, vxyz_f, ticks);
  ret |= compare3D( vxyz_d, vrep_f, ticks);
  ret |= compare3D( vrtp_d, vrep_f, ticks);

  ret |= compare3D( vxyz_f, vxyz_f, ticks);
  ret |= compare3D( vxyz_f, vrep_f, ticks);
  ret |= compare3D( vrtp_f, vrep_f, ticks);

#ifdef DEBUG
  if (ret == 0) std::cout << "\t OK\n";
  else { 
     std::cout << "\t FAIL\n";
     std::cerr << "\n>>>>> Testing DisplacementVector3D from " << v << " ticks = " << ticks 
               << "\t:\t FAILED\n";
  }
  std::cout << "           Testing PositionVector3D     :   ";
#endif


  PositionVector3D< Cartesian3D     <double> > pxyz_d; pxyz_d = vxyz_d;
  PositionVector3D< Polar3D         <double> > prtp_d; prtp_d = vrtp_d;
  PositionVector3D< CylindricalEta3D<double> > prep_d; prep_d = vrep_d;
  PositionVector3D< Cylindrical3D   <double> > przp_d; przp_d = vrzp_d;

  ret |= compare3D( pxyz_d, prtp_d, ticks);
  ret |= compare3D( vxyz_d, prep_d, ticks);
  ret |= compare3D( vrtp_d, prep_d, ticks);
  ret |= compare3D( vxyz_d, przp_d, ticks);

  PositionVector3D< Cartesian3D<float> >      pxyz_f (v.x(), v.y(), v.z());
  PositionVector3D< Polar3D<float> >          prtp_f ( r, theta, phi );
  PositionVector3D< CylindricalEta3D<float> > prep_f ( rho, eta, phi );

  ret |= compare3D( vxyz_d, pxyz_f, ticks);
  ret |= compare3D( vxyz_d, prep_f, ticks);
  ret |= compare3D( vrtp_d, prep_f, ticks);

  ret |= compare3D( vxyz_f, pxyz_f, ticks);
  ret |= compare3D( vxyz_f, prep_f, ticks);
  ret |= compare3D( vrtp_f, prep_f, ticks);

#ifdef DEBUG
  if (ret == 0) std::cout << "\t\t OK\n";
  else { 
     std::cout << "\t FAIL\n";
     std::cerr << "\n>>>>> Testing PositionVector3D from     " << v << " ticks = " << ticks 
               << "\t:\t FAILED\n";
  }
#endif
  return ret;
}



int coordinates3D () {
  int ret = 0;

  ret |= test3D (XYZVector ( 0.0, 0.0, 0.0 )     ,2 );
  ret |= test3D (XYZVector ( 1.0, 2.0, 3.0 )     ,6 );
  ret |= test3D (XYZVector ( -1.0, 2.0, 3.0 )    ,6 );
  ret |= test3D (XYZVector ( 1.0, -2.0, 3.0 )    ,6 );
  ret |= test3D (XYZVector ( 1.0, 2.0, -3.0 )    ,6 );
  ret |= test3D (XYZVector ( -1.0, -2.0, 3.0 )   ,6 );
  ret |= test3D (XYZVector ( -1.0, 2.0, -3.0 )   ,6 );
  ret |= test3D (XYZVector ( 1.0, -2.0, -3.0 )   ,6 );
  ret |= test3D (XYZVector ( -1.0, -2.0, -3.0 )  ,6 );
  ret |= test3D (XYZVector ( 8.0, 0.0, 0.0 )     ,6 );
  ret |= test3D (XYZVector ( -8.0, 0.0, 0.0 )    ,12 );
  ret |= test3D (XYZVector ( 0.0, 9.0, 0.0 )     ,6 );
  ret |= test3D (XYZVector ( 0.0, -9.0, 0.0 )    ,6 );
// rho == 0 tests the beyon-eta-max cases of cylindricalEta
  ret |= test3D (XYZVector ( 0.0, 0.0, 7.0 )     ,8 );
  ret |= test3D (XYZVector ( 0.0, 0.0, -7.0 )    ,8 );
// Larger ratios among coordinates presents a precision challenge
  ret |= test3D (XYZVector ( 16.0, 0.02, .01 )   ,10 );
  ret |= test3D (XYZVector ( -16.0, 0.02, .01 )  ,10 );
  ret |= test3D (XYZVector ( -.01, 16.0, .01 )   ,2000 ); 
  ret |= test3D (XYZVector ( -.01, -16.0, .01 )  ,2000 ); 
  ret |= test3D (XYZVector ( 1.0, 2.0, 30.0 )    ,10 );  
  	// NOTE -- these larger erros are likely the results of treating
	//         the vector in a ctor or assignment as foreign... 
	// NO -- I'm fouling up the value of x() !!!!!
// As we push to higher z with zero rho, some accuracy loss is expected
  ret |= test3D (XYZVector ( 0.0, 0.0, 15.0 )    ,30 );
  ret |= test3D (XYZVector ( 0.0, 0.0, -15.0 )   ,30 );
  ret |= test3D (XYZVector ( 0.0, 0.0, 35.0 )    ,30 );
  ret |= test3D (XYZVector ( 0.0, 0.0, -35.0 )   ,30 );
// When z is big compared to rho, it is very hard to get precision in polar/eta:
  ret |= test3D (XYZVector ( 0.01, 0.02, 16.0 )  ,10 );
  ret |= test3D (XYZVector ( 0.01, 0.02, -16.0 ) ,40000 );
// test case when eta is large 
  ret |= test3D (XYZVector ( 1.E-8, 1.E-8, 10.0 )  , 20 );
// when z is neg error is larger in eta when calculated from polar 
// since we have a larger error in theta which is closer to pi 
  ret |= test3D (XYZVector ( 1.E-8, 1.E-8, -10.0 ) ,2.E9 );   

  // small value of z
  ret |= test3D (XYZVector ( 10., 10., 1.E-8 ) ,1.0E6 );   
  ret |= test3D (XYZVector ( 10., 10., -1.E-8 ) ,1.0E6 );   


  return ret;
}

int main() { 
   int ret = coordinates3D();
   if (ret)  std::cerr << "test FAILED !!! " << std::endl; 
   else   std::cout << "test OK " << std::endl;
   return ret;
}
