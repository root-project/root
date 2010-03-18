// $Id $
// 
// Tests that each form of 4-vector has all the properties that stem from
// owning and forwarding to a 4D coordinates instance 
//
// 6/28/05 m fischler	
//         from contents of test_coordinates.h by L. Moneta.
//
// =================================================================


#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/Polar3D.h"
#include "Math/GenVector/CylindricalEta3D.h"
#include "Math/GenVector/etaMax.h"

#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/PxPyPzM4D.h"
#include "Math/GenVector/PtEtaPhiE4D.h"
#include "Math/GenVector/PtEtaPhiM4D.h"
#include "Math/GenVector/LorentzVector.h"

#include "Math/Vector4Dfwd.h"  // for typedefs definitions

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
  Scalar1 eps1 = std::numeric_limits<Scalar1>::epsilon();
  Scalar2 eps2 = std::numeric_limits<Scalar2>::epsilon();
  typedef typename LessPreciseType<Scalar1, Scalar2,Precision<Scalar1,Scalar2>::result>::type Scalar;
  Scalar epsilon = (eps1 >= eps2) ? eps1 : eps2;
  int pr = std::cout.precision(18);
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
int compare4D (const V1 & v1, const V2 & v2, double ticks) {
  int ret =0;  
  typedef typename V1::CoordinateType CoordType1;
  typedef typename V2::CoordinateType CoordType2;
  
  ret |= closeEnough ( v1.x(),     v2.x(),     "x"     ,ticks); 
  ret |= closeEnough ( v1.y(),     v2.y(),     "y"     ,ticks); 
  ret |= closeEnough ( v1.z(),     v2.z(),     "z"     ,ticks); 
  ret |= closeEnough ( v1.t(),     v2.t(),     "t"     ,ticks); 
  ret |= closeEnough ( v1.rho(),   v2.rho(),   "rho"   ,ticks); 
  ret |= closeEnough ( v1.phi(),   v2.phi(),   "phi"   ,ticks); 
  ret |= closeEnough ( v1.P(),     v2.P(),     "p"     ,ticks); 
  ret |= closeEnough ( v1.theta(), v2.theta(), "theta" ,ticks); 
  ret |= closeEnough ( v1.perp2(), v2.perp2(), "perp2" ,ticks); 
  ret |= closeEnough ( v1.M2(),    v2.M2(),    "m2"    ,ticks); 
  ret |= closeEnough ( v1.M(),     v2.M(),     "m"     ,ticks); 
  ret |= closeEnough ( v1.Mt(),    v2.Mt(),    "mt"     ,ticks); 
  ret |= closeEnough ( v1.Et(),    v2.Et(),    "et"     ,ticks); 
  if ( v1.rho() > 0 && v2.rho() > 0 ) { // eta can legitimately vary if rho == 0
    ret |= closeEnough ( v1.eta(), v2.eta(),   "eta"   ,ticks); 
  }
  
  if (ret != 0) {
    std::cout << "Discrepancy detected (see above) is between:\n  "
              << CoordinateTraits<CoordType1>::name() << " and\n  "
              << CoordinateTraits<CoordType2>::name() << "\n"
	      << "with v = (" << v1.x() << ", " << v1.y() << ", " 
	      << v1.z() << ", " << v1.t() << ")\n";
  }
  else { 
    std::cout << ".";
  }

  return ret;
}




template <class C>
int test4D ( const LorentzVector<C> & v, double ticks ) {

#ifdef DEBUG
   std::cout <<"\n>>>>> Testing LorentzVector from " << XYZTVector(v) << " ticks = " << ticks << "\t: ";
#endif

  int ret = 0;
  LorentzVector< PxPyPzE4D<double> > vxyzt_d (v.x(), v.y(), v.z(), v.t());

  //double m = std::sqrt ( v.t()*v.t() - v.x()*v.x() - v.y()*v.y() - v.z()*v.z());
  //double r = std::sqrt (v.x()*v.x() + v.y()*v.y() + v.z()*v.z());
  double rho = std::sqrt (v.x()*v.x() + v.y()*v.y());
  double theta = std::atan2( rho, v.z() );  // better tahn using acos
  //double theta = r>0 ? std::acos ( v.z()/r ) : 0;
  double phi = rho>0 ? std::atan2 (v.y(), v.x()) : 0;
    
  double eta;
  if (rho != 0) {
    eta = -std::log(std::tan(theta/2)); 
	#ifdef TRACE1
	std::cout <<  ":::: rho != 0\n" 
	          <<  ":::: theta = " <<  theta 
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


  LorentzVector< PtEtaPhiE4D<double> > vrep_d ( rho, eta, phi, v.t() );
  ret |= compare4D( vxyzt_d, vrep_d, ticks);

  LorentzVector< PtEtaPhiM4D<double> > vrepm_d ( rho, eta, phi, v.M() );
  ret |= compare4D( vxyzt_d, vrep_d, ticks);

  LorentzVector< PxPyPzM4D  <double> > vxyzm_d ( v.x(), v.y(), v.z(), v.M() );
  ret |= compare4D( vrep_d, vxyzm_d, ticks);
  
  LorentzVector< PxPyPzE4D<float> >      vxyzt_f (v.x(), v.y(), v.z(), v.t());
  ret |= compare4D( vxyzt_d, vxyzt_f, ticks);

  LorentzVector< PtEtaPhiE4D<float> > vrep_f ( rho, eta, phi, v.t() );
  ret |= compare4D( vxyzt_f, vrep_f, ticks);

  LorentzVector< PtEtaPhiM4D<float> > vrepm_f ( rho, eta, phi, v.M() );
  ret |= compare4D( vxyzt_f, vrepm_f, ticks);

  LorentzVector< PxPyPzM4D<float> >      vxyzm_f (v.x(), v.y(), v.z(), v.M());
  ret |= compare4D( vrep_f, vxyzm_f, ticks);

  if (ret == 0) std::cout << "\t OK\n";
  else { 
     std::cout << "\t FAIL\n";
     std::cerr << "\n>>>>> Testing LorentzVector from " << XYZTVector(v) << " ticks = " << ticks 
               << "\t:\t FAILED\n";
  }
  return ret; 
}


int coordinates4D (bool testAll = false) {
  int ret = 0;

  ret |= test4D (XYZTVector ( 0.0, 0.0, 0.0, 0.0 )     , 1 );
  ret |= test4D (XYZTVector ( 1.0, 2.0, 3.0, 4.0 )     ,10 );
  ret |= test4D (XYZTVector ( -1.0, -2.0, 3.0, 4.0 )   ,10 );
  // test for large eta values (which was giving inf before  Jun 07)
  ret |= test4D (XYZTVector ( 1.E-8, 1.E-8, 10.0, 100.0 )   ,10 );
  // for z < 0 precision in eta is worse since theta is close to Pi 
  ret |= test4D (XYZTVector ( 1.E-8, 1.E-8, -10.0, 100.0 )   ,1000000000 );

  // test cases with zero mass

  // tick should be p /sqrt(eps) ~ 4 /sqrt(eps)
  ret |= test4D (PxPyPzMVector ( 1., 2., 3., 0.)  ,  4./std::sqrt(std::numeric_limits<double>::epsilon()) );

  // this test fails in some machines (skip by default) 
  if (!testAll) return ret;  

  // take a factor 1.5 in ticks to be conservative
  ret |= test4D (PxPyPzMVector ( 1., 1., 100., 0.)  ,  150./std::sqrt(std::numeric_limits<double>::epsilon()) );
  // need a larger  a factor here
  ret |= test4D (PxPyPzMVector ( 1.E8, 1.E8, 1.E8, 0.)  ,  1.E9/std::sqrt(std::numeric_limits<double>::epsilon()) );
  // if use 1 here fails
  ret |= test4D (PxPyPzMVector ( 1.E-8, 1.E-8, 1.E-8, 0.)  ,  2.E-8/std::sqrt(std::numeric_limits<double>::epsilon()) );

  return ret;
}

int main() { 
   int ret = coordinates4D();
   if (ret)  std::cerr << "test FAILED !!! " << std::endl; 
   else   std::cout << "test OK " << std::endl;
   return ret;
}
