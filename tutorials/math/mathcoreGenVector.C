// Example macro testing available methods and operation of the GenVector classes.
// The results are compared and check at the
// numerical precision levels.
// Some small discrepancy can appear when the macro
// is executed on different architectures where it has been calibrated (Power PC G5)
// The macro is divided in 4 parts:
//    - testVector3D          :  tests of the 3D Vector classes
//    - testPoint3D           :  tests of the 3D Point classes
//    - testLorentzVector     :  tests of the 4D LorentzVector classes
//    - testVectorUtil        :  tests of the utility functions of all the vector classes
//
// To execute the macro type in:
//
// root[0]: .x  mathcoreGenVector.C
//Author: Lorenzo Moneta

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TMath.h"

#include "Math/Point3D.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/EulerAngles.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/Quaternion.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"
#include "Math/GenVector/RotationZYX.h"
#include "Math/GenVector/LorentzRotation.h"
#include "Math/GenVector/Boost.h"
#include "Math/GenVector/BoostX.h"
#include "Math/GenVector/BoostY.h"
#include "Math/GenVector/BoostZ.h"
#include "Math/GenVector/Transform3D.h"
#include "Math/GenVector/Plane3D.h"
#include "Math/GenVector/VectorUtil.h"

using namespace ROOT::Math;

int ntest = 0;
int nfail = 0;
int ok = 0;


int compare( double v1, double v2, const char* name, double Scale = 1.0) {
  ntest = ntest + 1;

  // numerical double limit for epsilon
  double eps = Scale* 2.22044604925031308e-16;
  int iret = 0;
  double delta = v2 - v1;
  double d = 0;
  if (delta < 0 ) delta = - delta;
  if (v1 == 0 || v2 == 0) {
    if  (delta > eps ) {
      iret = 1;
    }
  }
  // skip case v1 or v2 is infinity
  else {
     d = v1;

    if ( v1 < 0) d = -d;
    // add also case when delta is small by default
    if ( delta/d  > eps && delta > eps )
      iret =  1;
  }

  if (iret == 0)
    std::cout << ".";
  else {
    int pr = std::cout.precision (18);
    int discr;
    if (d != 0)
       discr = int(delta/d/eps);
    else
       discr = int(delta/eps);

    std::cout << "\nDiscrepancy in " << name << "() : " << v1 << " != " << v2 << " discr = " << discr
              << "   (Allowed discrepancy is " << eps  << ")\n";
    std::cout.precision (pr);
    nfail = nfail + 1;
  }
  return iret;
}




int testVector3D() {


  std::cout << "\n************************************************************************\n "
       << " Vector 3D Test"
       << "\n************************************************************************\n";

  //CINT cannot autoload classes known only via a typedef (here XYZVector)
  gSystem->Load("libGenVector");

  XYZVector v1(0.01, 0.02, 16);
  //XYZVector v1(1.0, 2.0, 3.0);

//   XYZVector v1(1.0, 2.0, 30.0);

//   double R = sqrt (v1.X()*v1.X() + v1.Y()*v1.Y() + v1.Z()*v1.Z());
//   // this formula in not precise enough
//   //  double Theta = R>0 ? acos ( v1.Z()/r ) : 0;
//   double Rho = sqrt (v1.X()*v1.X() + v1.Y()*v1.Y());
//   double Theta = v1.Z() == 0 || Rho == 0 ? 0 : atan2( Rho, v1.Z() );
//   double Phi = Rho>0 ? atan2 (v1.Y(), v1.X()) : 0;

  std::cout << "Test Cartesian-Polar :          " ;

  Polar3DVector v2(v1.R(), v1.Theta(), v1.Phi() );

  ok = 0;
  ok+= compare(v1.X(), v2.X(), "x");
  ok+= compare(v1.Y(), v2.Y(), "y");
  ok+= compare(v1.Z(), v2.Z(), "z");
  ok+= compare(v1.Phi(), v2.Phi(), "phi");
  ok+= compare(v1.Theta(), v2.Theta(), "theta");
  ok+= compare(v1.R(), v2.R(), "r");
  ok+= compare(v1.Eta(), v2.Eta(), "eta");
  ok+= compare(v1.Rho(), v2.Rho(), "rho");

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Cartesian-CylindricalEta : ";

  RhoEtaPhiVector v3( v1.Rho(), v1.Eta(), v1.Phi() );

  ok = 0;
  ok+= compare(v1.X(), v3.X(), "x");
  ok+= compare(v1.Y(), v3.Y(), "y");
  ok+= compare(v1.Z(), v3.Z(), "z");
  ok+= compare(v1.Phi(), v3.Phi(), "phi");
  ok+= compare(v1.Theta(), v3.Theta(), "theta");
  ok+= compare(v1.R(), v3.R(), "r");
  ok+= compare(v1.Eta(), v3.Eta(), "eta");
  ok+= compare(v1.Rho(), v3.Rho(), "rho");

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Cartesian-Cylindrical :    ";

  RhoZPhiVector v4( v1.Rho(), v1.Z(), v1.Phi() );

  ok = 0;
  ok+= compare(v1.X(), v4.X(), "x");
  ok+= compare(v1.Y(), v4.Y(), "y");
  ok+= compare(v1.Z(), v4.Z(), "z");
  ok+= compare(v1.Phi(), v4.Phi(), "phi");
  ok+= compare(v1.Theta(), v4.Theta(), "theta");
  ok+= compare(v1.R(), v4.R(), "r");
  ok+= compare(v1.Eta(), v4.Eta(), "eta");
  ok+= compare(v1.Rho(), v4.Rho(), "rho");

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Operations :               " ;

  ok = 0;
  double Dot = v1.Dot(v2);
  ok+= compare( Dot, v1.Mag2(),"dot"  );
  XYZVector vcross = v1.Cross(v2);
  ok+= compare( vcross.R(), 0,"cross"  );

  //std::cout << "\nTest Unit & scaling : " ;

  XYZVector vscale1 = v1*10;
  XYZVector vscale2 = vscale1/10;
  ok+= compare( v1.R(), vscale2.R(), "scale");

  XYZVector vu = v1.Unit();
  ok+= compare(v2.Phi(),vu.Phi(),"unit Phi");
  ok+= compare(v2.Theta(),vu.Theta(),"unit Theta");
  ok+= compare(1.0,vu.R(),"unit ");

  XYZVector q1 = v1;
  // RhoEtaPhiVector q2 = v1;  ! copy onstructor between different vector does not work yet)
  RhoEtaPhiVector q2(1.0,1.0,1.0);

  XYZVector q3 = q1 + q2;
  XYZVector q4 = q3 - q2;

  ok+= compare( q4.X(), q1.X(), "op X"  );
  ok+= compare( q4.Y(), q1.Y(), "op Y" );
  ok+= compare( q4.Z(), q1.Z(), "op Z" );

  // test operator ==
  XYZVector        w1 = v1;
  Polar3DVector    w2 = v2;
  RhoEtaPhiVector  w3 = v3;
  RhoZPhiVector    w4 = v4;
  ok+= compare( w1 == v1, static_cast<double>(true), "== XYZ");
  ok+= compare( w2 == v2, static_cast<double>(true), "== Polar");
  ok+= compare( w3 == v3, static_cast<double>(true), "== RhoEtaPhi");
  ok+= compare( w4 == v4, static_cast<double>(true), "== RhoZPhi");


  if (ok == 0) std::cout << "\t OK " << std::endl;


  //test setters

  std::cout << "Test Setters :                  " ;

  q2.SetXYZ(q1.X(), q1.Y(), q1.Z() );

  ok+= compare( q2.X(), q1.X(), "setXYZ X"  );
  ok+= compare( q2.Y(), q1.Y(), "setXYZ Y" );
  ok+= compare( q2.Z(), q1.Z(), "setXYZ Z" );

  q2.SetCoordinates( 2.0*q1.Rho(), q1.Eta(), q1.Phi() );
  XYZVector q1s = 2.0*q1;
  ok+= compare( q2.X(), q1s.X(), "set X"  );
  ok+= compare( q2.Y(), q1s.Y(), "set Y" );
  ok+= compare( q2.Z(), q1s.Z(), "set Z" );


  if (ok == 0) std::cout << "\t\t OK " << std::endl;

  std::cout << "Test Linear Algebra conversion: " ;

  XYZVector vxyz1(1.,2.,3.);

  TVectorD vla1(3);
  vxyz1.Coordinates().GetCoordinates(vla1.GetMatrixArray() );

  TVectorD vla2(3);
  vla2[0] = 1.; vla2[1] = -2.; vla2[2] = 1.;

  XYZVector vxyz2;
  vxyz2.SetCoordinates(&vla2[0]);

  ok = 0;
  double prod1 =  vxyz1.Dot(vxyz2);
  double prod2 = vla1*vla2;
  ok+= compare( prod1, prod2, "la test" );

  if (ok == 0) std::cout << "\t\t OK " << std::endl;

  return ok;
}



int testPoint3D() {

  std::cout << "\n************************************************************************\n "
       << " Point 3D Tests"
       << "\n************************************************************************\n";



  //XYZPoint p1(0.00001, 0.00001, 30000000000.0);
  XYZPoint p1(1.0, 2.0, 3.0);

  std::cout << "Test Cartesian-Polar :          ";

  Polar3DPoint p2(p1.R(), p1.Theta(), p1.Phi() );

  ok = 0;
  ok+= compare(p1.x(), p2.X(), "x");
  ok+= compare(p1.y(), p2.Y(), "y");
  ok+= compare(p1.z(), p2.Z(), "z");
  ok+= compare(p1.phi(), p2.Phi(), "phi");
  ok+= compare(p1.theta(), p2.Theta(), "theta");
  ok+= compare(p1.r(), p2.R(), "r");
  ok+= compare(p1.eta(), p2.Eta(), "eta");
  ok+= compare(p1.rho(), p2.Rho(), "rho");

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test Polar-CylindricalEta :     ";

  RhoEtaPhiPoint p3( p2.Rho(), p2.Eta(), p2.Phi() );

  ok = 0;
  ok+= compare(p2.X(), p3.X(), "x");
  ok+= compare(p2.Y(), p3.Y(), "y");
  ok+= compare(p2.Z(), p3.Z(), "z",3);
  ok+= compare(p2.Phi(), p3.Phi(), "phi");
  ok+= compare(p2.Theta(), p3.Theta(), "theta");
  ok+= compare(p2.R(), p3.R(), "r");
  ok+= compare(p2.Eta(), p3.Eta(), "eta");
  ok+= compare(p2.Rho(), p3.Rho(), "rho");

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test operations :               ";

  //std::cout << "\nTest Dot and Cross products with Vectors : ";
  Polar3DVector vperp(1.,p1.Theta() + TMath::PiOver2(),p1.Phi() );
  double Dot = p1.Dot(vperp);
  ok+= compare( Dot, 0.0,"dot", 10  );

  XYZPoint vcross = p1.Cross(vperp);
  ok+= compare( vcross.R(), p1.R(),"cross mag"  );
  ok+= compare( vcross.Dot(vperp), 0.0,"cross dir"  );

  XYZPoint pscale1 = 10*p1;
  XYZPoint pscale2 = pscale1/10;
  ok+= compare( p1.R(), pscale2.R(), "scale");

  // test operator ==
  ok+= compare( p1 == pscale2, static_cast<double>(true), "== Point");


  //RhoEtaPhiPoint q1 = p1;  ! constructor yet not working in CINT
  RhoEtaPhiPoint q1; q1 = p1;
  q1.SetCoordinates(p1.Rho(),2.0, p1.Phi() );

  Polar3DVector v2(p1.R(), p1.Theta(),p1.Phi());


  //#ifdef WHEN_CINT_FIXED
  RhoEtaPhiPoint q3 = q1 + v2;
  // point -point in vector does not work yet
  RhoEtaPhiPoint q4 = q3 - v2;
  ok+= compare( q4.X(), q1.X(), "PV op X"  );
  ok+= compare( q4.Y(), q1.Y(), "PV op Y" );
  ok+= compare( q4.Z(), q1.Z(), "PV op Z" ,2);
  //#endif

  if (ok == 0) std::cout << "\t OK " << std::endl;


//   RhoEtaPhiVector v4 = q3 - q1;
//   ok+= compare( v4.X(), v2.X(), "op X"  );
//   ok+= compare( v4.Y(), v2.Y(), "op Y" );
//   ok+= compare( v4.Z(), v2.Z(), "op Z" );

  return ok;

}




int testLorentzVector() {

  std::cout << "\n************************************************************************\n "
       << " Loorentz Vector Tests"
       << "\n************************************************************************\n";



  //XYZTVector v1(0.00001, 0.00001, 30000000000.0);
  XYZTVector v1(1.0, 2.0, 3.0, 4.0);


  std::cout << "Test XYZT - PtEtaPhiE Vectors:  ";

  PtEtaPhiEVector v2( v1.Rho(), v1.Eta(), v1.Phi(), v1.E() );

  ok = 0;
  ok+= compare(v1.Px(), v2.X(), "x");
  ok+= compare(v1.Py(), v2.Y(), "y");
  ok+= compare(v1.Pz(), v2.Z(), "z", 2);
  ok+= compare(v1.E(), v2.T(), "e");
  ok+= compare(v1.Phi(), v2.Phi(), "phi");
  ok+= compare(v1.Theta(), v2.Theta(), "theta");
  ok+= compare(v1.Pt(), v2.Pt(), "pt");
  ok+= compare(v1.M(), v2.M(), "mass", 5);
  ok+= compare(v1.Et(), v2.Et(), "et");
  ok+= compare(v1.Mt(), v2.Mt(), "mt", 3);

  if (ok == 0) std::cout << "\t OK " << std::endl;


  std::cout << "Test XYZT - PtEtaPhiM Vectors:  ";

  PtEtaPhiMVector v3( v1.Rho(), v1.Eta(), v1.Phi(), v1.M() );

  ok = 0;
  ok+= compare(v1.Px(), v3.X(), "x");
  ok+= compare(v1.Py(), v3.Y(), "y");
  ok+= compare(v1.Pz(), v3.Z(), "z", 2);
  ok+= compare(v1.E(), v3.T(), "e");
  ok+= compare(v1.Phi(), v3.Phi(), "phi");
  ok+= compare(v1.Theta(), v3.Theta(), "theta");
  ok+= compare(v1.Pt(), v3.Pt(), "pt");
  ok+= compare(v1.M(), v3.M(), "mass", 5);
  ok+= compare(v1.Et(), v3.Et(), "et");
  ok+= compare(v1.Mt(), v3.Mt(), "mt", 3);

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test PtEtaPhiE - PxPyPzM Vect.: ";

  PxPyPzMVector v4( v3.X(), v3.Y(), v3.Z(), v3.M() );

  ok = 0;
  ok+= compare(v4.Px(), v3.X(), "x");
  ok+= compare(v4.Py(), v3.Y(), "y");
  ok+= compare(v4.Pz(), v3.Z(), "z",2);
  ok+= compare(v4.E(), v3.T(), "e");
  ok+= compare(v4.Phi(), v3.Phi(), "phi");
  ok+= compare(v4.Theta(), v3.Theta(), "theta");
  ok+= compare(v4.Pt(), v3.Pt(), "pt");
  ok+= compare(v4.M(), v3.M(), "mass",5);
  ok+= compare(v4.Et(), v3.Et(), "et");
  ok+= compare(v4.Mt(), v3.Mt(), "mt",3);

  if (ok == 0) std::cout << "\t OK " << std::endl;

  std::cout << "Test operations :               ";
  //std::cout << "\nTest Dot product : " ;

  ok = 0;
  double Dot = v1.Dot(v2);
  ok+= compare( Dot, v1.M2(),"dot" , 10 );

  //std::cout << "\nTest scaling : " ;

  XYZTVector vscale1 = v1*10;
  XYZTVector vscale2 = vscale1/10;
  ok+= compare( v1.M(), vscale2.M(), "scale");


  XYZTVector q1 = v1;
  // RhoEtaPhiVector q2 = v1;  ! copy onstructor between different vector does not work yet)
  PtEtaPhiEVector  q2(1.0,1.0,1.0,5.0);

  XYZTVector q3 = q1 + q2;
  XYZTVector q4 = q3 - q2;

  ok+= compare( q4.x(), q1.X(), "op X"  );
  ok+= compare( q4.y(), q1.Y(), "op Y" );
  ok+= compare( q4.z(), q1.Z(), "op Z" );
  ok+= compare( q4.t(), q1.E(), "op E" );

  // test operator ==
  XYZTVector        w1 = v1;
  PtEtaPhiEVector   w2 = v2;
  PtEtaPhiMVector   w3 = v3;
  PxPyPzMVector     w4 = v4;
  ok+= compare( w1 == v1, static_cast<double>(true), "== PxPyPzE");
  ok+= compare( w2 == v2, static_cast<double>(true), "== PtEtaPhiE");
  ok+= compare( w3 == v3, static_cast<double>(true), "== PtEtaPhiM");
  ok+= compare( w4 == v4, static_cast<double>(true), "== PxPyPzM");

  // test gamma beta and boost
  XYZVector b = q1.BoostToCM();
  double beta = q1.Beta();
  double gamma = q1.Gamma();

  ok += compare( b.R(), beta, "beta" );
  ok += compare( gamma, 1./sqrt( 1 - beta*beta ), "gamma");


  if (ok == 0) std::cout << "\t OK " << std::endl;

  //test setters

  std::cout << "Test Setters :                  " ;

  q2.SetXYZT(q1.Px(), q1.Py(), q1.Pz(), q1.E() );

  ok+= compare( q2.X(), q1.X(), "setXYZT X"  );
  ok+= compare( q2.Y(), q1.Y(), "setXYZT Y" );
  ok+= compare( q2.Z(), q1.Z(), "setXYZT Z" ,2);
  ok+= compare( q2.T(), q1.E(), "setXYZT E" );

  q2.SetCoordinates( 2.0*q1.Rho(), q1.Eta(), q1.Phi(), 2.0*q1.E() );
  XYZTVector q1s = q1*2.0;
  ok+= compare( q2.X(), q1s.X(), "set X"  );
  ok+= compare( q2.Y(), q1s.Y(), "set Y" );
  ok+= compare( q2.Z(), q1s.Z(), "set Z" ,2);
  ok+= compare( q2.T(), q1s.T(),  "set E" );


  if (ok == 0) std::cout << "\t OK " << std::endl;


  return ok;
}


int testVectorUtil() {


  std::cout << "\n************************************************************************\n "
       << " Utility Function Tests"
       << "\n************************************************************************\n";

  std::cout << "Test Vector utility functions : ";


  XYZVector v1(1.0, 2.0, 3.0);
  Polar3DVector v2pol(v1.R(), v1.Theta()+TMath::PiOver2(), v1.Phi() + 1.0);
  // mixedmethods not yet impl.
  XYZVector v2; v2 = v2pol;

  ok = 0;
  ok += compare( VectorUtil::DeltaPhi(v1,v2), 1.0, "deltaPhi Vec");

  RhoEtaPhiVector v2cyl(v1.Rho(), v1.Eta() + 1.0, v1.Phi() + 1.0);
  v2 = v2cyl;


  ok += compare( VectorUtil::DeltaR(v1,v2), sqrt(2.0), "DeltaR Vec");

  XYZVector vperp = v1.Cross(v2);
  ok += compare( VectorUtil::CosTheta(v1,vperp), 0.0, "costheta Vec");
  ok += compare( VectorUtil::Angle(v1,vperp), TMath::PiOver2(), "angle Vec");

  if (ok == 0) std::cout << "\t\t OK " << std::endl;


  std::cout << "Test Point utility functions :  ";


  XYZPoint p1(1.0, 2.0, 3.0);
  Polar3DPoint p2pol(p1.R(), p1.Theta()+TMath::PiOver2(), p1.Phi() + 1.0);
  // mixedmethods not yet impl.
  XYZPoint p2; p2 = p2pol;

  ok = 0;
  ok += compare( VectorUtil::DeltaPhi(p1,p2), 1.0, "deltaPhi Point");

  RhoEtaPhiPoint p2cyl(p1.Rho(), p1.Eta() + 1.0, p1.Phi() + 1.0);
  p2 = p2cyl;
  ok += compare( VectorUtil::DeltaR(p1,p2), sqrt(2.0), "DeltaR Point");

  XYZPoint pperp(vperp.X(), vperp.Y(), vperp.Z());
  ok += compare( VectorUtil::CosTheta(p1,pperp), 0.0, "costheta Point");
  ok += compare( VectorUtil::Angle(p1,pperp), TMath::PiOver2(), "angle Point");

  if (ok == 0) std::cout << "\t\t OK " << std::endl;


  std::cout << "LorentzVector utility funct.:   ";


  XYZTVector q1(1.0, 2.0, 3.0,4.0);
  PtEtaPhiEVector q2cyl(q1.Pt(), q1.Eta()+1.0, q1.Phi() + 1.0, q1.E() );
  // mixedmethods not yet impl.
  XYZTVector q2; q2 = q2cyl;

  ok = 0;
  ok += compare( VectorUtil::DeltaPhi(q1,q2), 1.0, "deltaPhi LVec");
  ok += compare( VectorUtil::DeltaR(q1,q2), sqrt(2.0), "DeltaR LVec");

  XYZTVector qsum = q1+q2;
  ok += compare( VectorUtil::InvariantMass(q1,q2), qsum.M(), "InvMass");

  if (ok == 0) std::cout << "\t\t OK " << std::endl;

  return ok;

}



int testRotation() {


  std::cout << "\n************************************************************************\n "
       << " Rotation and Transformation Tests"
       << "\n************************************************************************\n";

  std::cout << "Test Vector Rotations :         ";
  ok = 0;

  XYZPoint v(1.,2,3.);

  double pi = TMath::Pi();
  // initiate rotation with some non -trivial angles to test all matrix
  EulerAngles r1( pi/2.,pi/4., pi/3 );
  Rotation3D  r2(r1);
  // only operator= is in CINT for the other rotations
  Quaternion  r3; r3 = r2;
  AxisAngle   r4; r4 = r3;
  RotationZYX r5; r5 = r2;

  XYZPoint v1 = r1 * v;
  XYZPoint v2 = r2 * v;
  XYZPoint v3 = r3 * v;
  XYZPoint v4 = r4 * v;
  XYZPoint v5 = r5 * v;

  ok+= compare(v1.X(), v2.X(), "x",2);
  ok+= compare(v1.Y(), v2.Y(), "y",2);
  ok+= compare(v1.Z(), v2.Z(), "z",2);

  ok+= compare(v1.X(), v3.X(), "x",2);
  ok+= compare(v1.Y(), v3.Y(), "y",2);
  ok+= compare(v1.Z(), v3.Z(), "z",2);

  ok+= compare(v1.X(), v4.X(), "x",5);
  ok+= compare(v1.Y(), v4.Y(), "y",5);
  ok+= compare(v1.Z(), v4.Z(), "z",5);

  ok+= compare(v1.X(), v5.X(), "x",2);
  ok+= compare(v1.Y(), v5.Y(), "y",2);
  ok+= compare(v1.Z(), v5.Z(), "z",2);

  // test with matrix
  double rdata[9];
  r2.GetComponents(rdata, rdata+9);
  TMatrixD m(3,3,rdata);
  double vdata[3];
  v.GetCoordinates(vdata);
  TVectorD q(3,vdata);
  TVectorD q2 = m*q;

  XYZPoint v6;
  v6.SetCoordinates( q2.GetMatrixArray() );

  ok+= compare(v1.X(), v6.X(), "x");
  ok+= compare(v1.Y(), v6.Y(), "y");
  ok+= compare(v1.Z(), v6.Z(), "z");


  if (ok == 0) std::cout << "\t OK " << std::endl;
  else  std::cout << std::endl;

  std::cout << "Test Axial Rotations :          ";
  ok = 0;

  RotationX rx( pi/3);
  RotationY ry( pi/4);
  RotationZ rz( 4*pi/5);

  Rotation3D r3x(rx);
  Rotation3D r3y(ry);
  Rotation3D r3z(rz);

  Quaternion qx; qx = rx;
  Quaternion qy; qy = ry;
  Quaternion qz; qz = rz;

  RotationZYX rzyx( rz.Angle(), ry.Angle(), rx.Angle() );

  XYZPoint vrot1 = rx * ry * rz * v;
  XYZPoint vrot2 = r3x * r3y * r3z * v;

  ok+= compare(vrot1.X(), vrot2.X(), "x");
  ok+= compare(vrot1.Y(), vrot2.Y(), "y");
  ok+= compare(vrot1.Z(), vrot2.Z(), "z");

  vrot2 = qx * qy * qz * v;

  ok+= compare(vrot1.X(), vrot2.X(), "x",2);
  ok+= compare(vrot1.Y(), vrot2.Y(), "y",2);
  ok+= compare(vrot1.Z(), vrot2.Z(), "z",2);

  vrot2 = rzyx * v;

  ok+= compare(vrot1.X(), vrot2.X(), "x");
  ok+= compare(vrot1.Y(), vrot2.Y(), "y");
  ok+= compare(vrot1.Z(), vrot2.Z(), "z");

  // now inverse (first x then y then z)
  vrot1 = rz * ry * rx * v;
  vrot2 = r3z * r3y * r3x * v;

  ok+= compare(vrot1.X(), vrot2.X(), "x");
  ok+= compare(vrot1.Y(), vrot2.Y(), "y");
  ok+= compare(vrot1.Z(), vrot2.Z(), "z");


  XYZPoint vinv1 = rx.Inverse()*ry.Inverse()*rz.Inverse()*vrot1;

  ok+= compare(vinv1.X(), v.X(), "x",2);
  ok+= compare(vinv1.Y(), v.Y(), "y");
  ok+= compare(vinv1.Z(), v.Z(), "z");

  if (ok == 0) std::cout << "\t OK " << std::endl;
  else  std::cout << std::endl;


  std::cout << "Test Rotations by a PI angle :  ";
  ok = 0;

  double b[4] = { 6,8,10,3.14159265358979323 };
  AxisAngle  arPi(b,b+4 );
  Rotation3D rPi(arPi);
  AxisAngle  a1; a1 = rPi;
  ok+= compare(arPi.Axis().X(), a1.Axis().X(),"x");
  ok+= compare(arPi.Axis().Y(), a1.Axis().Y(),"y");
  ok+= compare(arPi.Axis().Z(), a1.Axis().Z(),"z");
  ok+= compare(arPi.Angle(), a1.Angle(),"angle");

  EulerAngles ePi; ePi=rPi;
  EulerAngles e1; e1=Rotation3D(a1);
  ok+= compare(ePi.Phi(), e1.Phi(),"phi");
  ok+= compare(ePi.Theta(), e1.Theta(),"theta");
  ok+= compare(ePi.Psi(), e1.Psi(),"ps1");

  if (ok == 0) std::cout << "\t\t OK " << std::endl;
  else  std::cout << std::endl;

  std::cout << "Test Inversions :               ";
  ok = 0;


  EulerAngles s1 = r1.Inverse();
  Rotation3D  s2 = r2.Inverse();
  Quaternion  s3 = r3.Inverse();
  AxisAngle   s4 = r4.Inverse();
  RotationZYX s5 = r5.Inverse();


  // euler angles not yet impl.
  XYZPoint p = s2 * r2 * v;

  ok+= compare(p.X(), v.X(), "x",10);
  ok+= compare(p.Y(), v.Y(), "y",10);
  ok+= compare(p.Z(), v.Z(), "z",10);


  p = s3 * r3 * v;

  ok+= compare(p.X(), v.X(), "x",10);
  ok+= compare(p.Y(), v.Y(), "y",10);
  ok+= compare(p.Z(), v.Z(), "z",10);

  p = s4 * r4 * v;
  // axis angle inversion not very precise
  ok+= compare(p.X(), v.X(), "x",1E9);
  ok+= compare(p.Y(), v.Y(), "y",1E9);
  ok+= compare(p.Z(), v.Z(), "z",1E9);

  p = s5 * r5 * v;

  ok+= compare(p.X(), v.X(), "x",10);
  ok+= compare(p.Y(), v.Y(), "y",10);
  ok+= compare(p.Z(), v.Z(), "z",10);


  Rotation3D r6(r5);
  Rotation3D s6 = r6.Inverse();

  p = s6 * r6 * v;

  ok+= compare(p.X(), v.X(), "x",10);
  ok+= compare(p.Y(), v.Y(), "y",10);
  ok+= compare(p.Z(), v.Z(), "z",10);

  if (ok == 0) std::cout << "\t OK " << std::endl;
  else  std::cout << std::endl;

  // test Rectify

  std::cout << "Test rectify :                  ";
  ok = 0;

  XYZVector u1(0.999498,-0.00118212,-0.0316611);
  XYZVector u2(0,0.999304,-0.0373108);
  XYZVector u3(0.0316832,0.0372921,0.998802);
  Rotation3D rr(u1,u2,u3);
  // check orto-normality
  XYZPoint vrr = rr* v;
  ok+= compare(v.R(), vrr.R(), "R",1.E9);

  if (ok == 0) std::cout << "\t\t OK " << std::endl;
  else  std::cout << std::endl;

  std::cout << "Test Transform3D :              ";
  ok = 0;

  XYZVector d(1.,-2.,3.);
  Transform3D t(r2,d);

  XYZPoint pd = t * v;
  // apply directly rotation
  XYZPoint vd = r2 * v + d;

  ok+= compare(pd.X(), vd.X(), "x");
  ok+= compare(pd.Y(), vd.Y(), "y");
  ok+= compare(pd.Z(), vd.Z(), "z");

  // test with matrix
  double tdata[12];
  t.GetComponents(tdata);
  TMatrixD mt(3,4,tdata);
  double vData[4]; // needs a vector of dim 4
  v.GetCoordinates(vData);
  vData[3] = 1;
  TVectorD q0(4,vData);

  TVectorD qt = mt*q0;

  ok+= compare(pd.X(), qt(0), "x");
  ok+= compare(pd.Y(), qt(1), "y");
  ok+= compare(pd.Z(), qt(2), "z");


  // test inverse

  Transform3D tinv = t.Inverse();

  p = tinv * t * v;

  ok+= compare(p.X(), v.X(), "x",10);
  ok+= compare(p.Y(), v.Y(), "y",10);
  ok+= compare(p.Z(), v.Z(), "z",10);

  // test costruct inverse from translation first

  //Transform3D tinv2( -d, r2.Inverse() );
  //Transform3D tinv2 =  r2.Inverse() * Translation3D(-d) ;
  Transform3D tinv2 ( r2.Inverse(), r2.Inverse() *( -d) ) ;
  p = tinv2 * t * v;

  ok+= compare(p.X(), v.X(), "x",10);
  ok+= compare(p.Y(), v.Y(), "y",10);
  ok+= compare(p.Z(), v.Z(), "z",10);

  // test from only rotation and only translation
  Transform3D ta( EulerAngles(1.,2.,3.) );
  Transform3D tb( XYZVector(1,2,3) );
  Transform3D tc(  Rotation3D(EulerAngles(1.,2.,3.)) ,  XYZVector(1,2,3) );
  Transform3D td(  ta.Rotation(), ta.Rotation()  * XYZVector(1,2,3) ) ;

  ok+= compare( tc == tb*ta, static_cast<double>(true), "== Rot*Tra");
  ok+= compare( td == ta*tb, static_cast<double>(true), "== Rot*Tra");


  if (ok == 0) std::cout << "\t OK " << std::endl;
  else  std::cout << std::endl;

  std::cout << "Test Plane3D :                  ";
  ok = 0;

  // test transfrom a 3D plane


  XYZPoint p1(1,2,3);
  XYZPoint p2(-2,-1,4);
  XYZPoint p3(-1,3,2);
  Plane3D plane(p1,p2,p3);

  XYZVector n = plane.Normal();
  // normal is perpendicular to vectors on the planes obtained from subracting the points
  ok+= compare(n.Dot(p2-p1), 0.0, "n.v12",10);
  ok+= compare(n.Dot(p3-p1), 0.0, "n.v13",10);
  ok+= compare(n.Dot(p3-p2), 0.0, "n.v23",10);

  Plane3D plane1 = t(plane);

  // transform the points
  XYZPoint pt1 = t(p1);
  XYZPoint pt2 = t(p2);
  XYZPoint pt3 = t(p3);
  Plane3D plane2(pt1,pt2,pt3);

  XYZVector n1 = plane1.Normal();
  XYZVector n2 = plane2.Normal();


  ok+= compare(n1.X(), n2.X(), "a",10);
  ok+= compare(n1.Y(), n2.Y(), "b",10);
  ok+= compare(n1.Z(), n2.Z(), "c",10);
  ok+= compare(plane1.HesseDistance(), plane2.HesseDistance(), "d",10);

  // check distances
  ok += compare(plane1.Distance(pt1), 0.0, "distance",10);

  if (ok == 0) std::cout << "\t OK " << std::endl;
  else  std::cout << std::endl;

  std::cout << "Test LorentzRotation :          ";
  ok = 0;

  XYZTVector lv(1.,2.,3.,4.);

  // test from rotx (using boosts and 3D rotations not yet impl.)
  // rx,ry and rz already defined
  Rotation3D r3d = rx*ry*rz;

  LorentzRotation rlx(rx);
  LorentzRotation rly(ry);
  LorentzRotation rlz(rz);

  LorentzRotation rl0 = rlx*rly*rlz;
  LorentzRotation rl1( r3d);

//   cout << rl << endl;
//   cout << rl0 << endl;
//   int eq = rl0 == rl;
//   cout << eq << endl;
//   double d1[16];
//   double d2[16];
//   rl.GetComponents(d1,d1+16);
//   rl0.GetComponents(d2,d2+16);
//   for (int i = 0; i < 16; ++i)
//     ok+= compare(d1[i], d2[i], "i",1);

  //ok+= compare( rl == rl2, static_cast<double>(true), " LorenzRot");


  //  cout << Rotation3D(rx) << endl;

  XYZTVector lv0 = rl0 * lv;

  XYZTVector lv1 = rl1 * lv;

  XYZTVector lv2 = r3d * lv;


  ok+= compare(lv1== lv2,true,"V0==V2");
  ok+= compare(lv1== lv2,true,"V1==V2");

  double rlData[16];
  rl0.GetComponents(rlData);
  TMatrixD ml(4,4,rlData);
  //  ml.Print();
  double lvData[4];
  lv.GetCoordinates(lvData);
  TVectorD ql(4,lvData);

  TVectorD qlr = ml*ql;

  ok+= compare(lv1.X(), qlr(0), "x");
  ok+= compare(lv1.Y(), qlr(1), "y");
  ok+= compare(lv1.Z(), qlr(2), "z");
  ok+= compare(lv1.E(), qlr(3), "t");

  // test inverse

  lv0 = rl0 * rl0.Inverse() * lv;

  ok+= compare(lv0.X(), lv.X(), "x");
  ok+= compare(lv0.Y(), lv.Y(), "y");
  ok+= compare(lv0.Z(), lv.Z(), "z");
  ok+= compare(lv0.E(), lv.E(), "t");

  if (ok == 0) std::cout << "\t OK " << std::endl;
  else  std::cout << std::endl;

  // test Boosts

  std::cout << "Test Boost :                    ";
  ok = 0;


  Boost bst( 0.3,0.4,0.5);   //  boost (must be <= 1)


  XYZTVector lvb = bst ( lv );

  LorentzRotation rl2 (bst);

  XYZTVector lvb2 = rl2 (lv);


  // test with lorentz rotation
  ok+= compare(lvb.X(), lvb2.X(), "x");
  ok+= compare(lvb.Y(), lvb2.Y(), "y");
  ok+= compare(lvb.Z(), lvb2.Z(), "z");
  ok+= compare(lvb.E(), lvb2.E(), "t");
  ok+= compare(lvb.M(), lv.M(), "m",50); // m must stay constant


  // test inverse
  lv0 = bst.Inverse() * lvb;

  ok+= compare(lv0.X(), lv.X(), "x",5);
  ok+= compare(lv0.Y(), lv.Y(), "y",5);
  ok+= compare(lv0.Z(), lv.Z(), "z",3);
  ok+= compare(lv0.E(), lv.E(), "t",3);

  XYZVector brest = lv.BoostToCM();
  bst.SetComponents( brest.X(), brest.Y(), brest.Z() );

  XYZTVector lvr = bst * lv;

  ok+= compare(lvr.X(), 0.0, "x",10);
  ok+= compare(lvr.Y(), 0.0, "y",10);
  ok+= compare(lvr.Z(), 0.0, "z",10);
  ok+= compare(lvr.M(), lv.M(), "m",10);


  if (ok == 0) std::cout << "\t OK " << std::endl;
  else  std::cout << std::endl;

  return ok;
}


void mathcoreGenVector() {

#ifdef __CINT__
  gSystem->Load("libMathCore");
  using namespace ROOT::Math;
#endif

  testVector3D();
  testPoint3D();
  testLorentzVector();
  testVectorUtil();
  testRotation();

  std::cout << "\n\nNumber of tests " << ntest << " failed = " << nfail << std::endl;
}

