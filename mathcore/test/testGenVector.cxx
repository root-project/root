
#include "Math/Vector3D.h"
#include "Math/Point3D.h"
#include "Math/EulerAngles.h"

#include "Math/Transform3D.h"

#include "Math/Rotation3D.h"
#include "Math/RotationX.h"
#include "Math/RotationY.h"
#include "Math/RotationZ.h"
#include "Math/Quaternion.h"
#include "Math/AxisAngle.h"
#include "Math/EulerAngles.h"

#include "Math/LorentzRotation.h"

#include "Math/VectorUtil.h"
#ifndef NO_SMATRIX
#include "Math/SMatrix.h"
#endif


using namespace ROOT::Math;
using namespace ROOT::Math::VectorUtil;



typedef  DisplacementVector3D<Cartesian3D<double>, GlobalCoordinateSystemTag>  GlobalXYZVector; 
typedef  DisplacementVector3D<Cartesian3D<double>, LocalCoordinateSystemTag>   LocalXYZVector; 
typedef  DisplacementVector3D<Polar3D<double>, GlobalCoordinateSystemTag>      GlobalPolar3DVector; 



typedef  PositionVector3D<Cartesian3D<double>, GlobalCoordinateSystemTag>   GlobalXYZPoint; 
typedef  PositionVector3D<Cartesian3D<double>, LocalCoordinateSystemTag>    LocalXYZPoint; 
typedef  PositionVector3D<Polar3D<double>, GlobalCoordinateSystemTag>       GlobalPolar3DPoint; 
typedef  PositionVector3D<Polar3D<double>, LocalCoordinateSystemTag>       LocalPolar3DPoint; 


//#define TEST_COMPILE_ERROR


int compare( double v1, double v2, const std::string & name = "", double scale = 1.0) {
  //  ntest = ntest + 1; 

  // numerical double limit for epsilon 
  double eps = scale* std::numeric_limits<double>::epsilon();
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
    std::cout << "\nDiscrepancy in " << name << "() : " << v1 << " != " << v2 << " discr = " << int(delta/d/eps) 
              << "   (Allowed discrepancy is " << eps  << ")\n";
    std::cout.precision (pr);
    //nfail = nfail + 1;
  }
  return iret; 
}

int testVector3D() { 

  int iret = 0;

  std::cout << "testing Vector3D   \t:\t"; 

  // test the vector tags 

  GlobalXYZVector vg(1.,2.,3.); 
  GlobalXYZVector vg2(vg); 

  GlobalPolar3DVector vpg(vg);

  iret |= compare(vpg.R(), vg2.R() );
 
//   std::cout << vg2 << std::endl;

  double r = vg.Dot(vpg); 
  iret |= compare(r, vg.Mag2() );

  GlobalXYZVector vcross = vg.Cross(vpg); 
  iret |= compare(vcross.R(), 0.0,"cross",10 );

//   std::cout << vg.Dot(vpg) << std::endl;
//   std::cout << vg.Cross(vpg) << std::endl;


  
 

  GlobalXYZVector vg3 = vg + vpg;  
  iret |= compare(vg3.R(), 2*vg.R() );

  GlobalXYZVector vg4 = vg - vpg;  
  iret |= compare(vg4.R(), 0.0,"diff",10 );




#ifdef TEST_COMPILE_ERROR
   LocalXYZVector vl; vl = vg; 
  LocalXYZVector vl2(vg2);
  LocalXYZVector vl3(vpg);
  vg.Dot(vl);
  vg.Cross(vl);
  vg3 = vg + vl;
  vg4 = vg - vl;
#endif


  if (iret == 0) std::cout << "\t\t\t\t\tOK\n"; 
  else std::cout << "\t\t\t\tFAILED\n";


  return iret;
}



int testPoint3D() { 

  int iret = 0;

  std::cout << "testing Point3D    \t:\t"; 

  // test the vector tags 

  GlobalXYZPoint pg(1.,2.,3.); 
  GlobalXYZPoint pg2(pg); 

  GlobalPolar3DPoint ppg(pg);

  iret |= compare(ppg.R(), pg2.R() );
  //std::cout << pg2 << std::endl;
  



  GlobalXYZVector vg(pg);

  double r = pg.Dot(vg); 
  iret |= compare(r, pg.Mag2() );

  GlobalPolar3DVector vpg(pg);
  GlobalXYZPoint pcross = pg.Cross(vpg);
  iret |= compare(pcross.R(), 0.0,"cross",10 );

  GlobalPolar3DPoint pg3 = ppg + vg; 
  iret |= compare(pg3.R(), 2*pg.R() );

  GlobalXYZVector vg4 = pg - ppg; 
  iret |= compare(vg4.R(), 0.0,"diff",10 );


#ifdef TEST_COMPILE_ERROR
  LocalXYZPoint pl; pl = pg; 
  LocalXYZVector pl2(pg2);
  LocalXYZVector pl3(ppg);
  pl.Dot(vg);
  pl.Cross(vg);
  pg3 = ppg + pg;
  pg3 = ppg + pl;
  vg4 = pg - pl;
#endif

  // operator - 
  XYZPoint q1(1.,2.,3.);
  XYZPoint q2 = -1.* q1; 
  XYZVector v2 = -XYZVector(q1);
  iret |= compare(XYZVector(q2) == v2,true,"reflection");


  if (iret == 0) std::cout << "\t\t\t\t\tOK\n"; 
  else std::cout << "\t\t\t\tFAILED\n"; 

  return iret;
}


// missing LV test

int testRotations3D() {  

  int iret=0; 
  std::cout << "testing 3D Rotations\t:\t"; 


  Rotation3D rot = RotationZ(1.) * RotationY(2) * RotationX(3); 
  GlobalXYZVector vg(1.,2.,3); 
  GlobalXYZPoint  pg(1.,2.,3); 
  GlobalPolar3DVector vpg(vg);  

  //  GlobalXYZVector vg2 = rot.operator()<Cartesian3D,GlobalCoordinateSystemTag, GlobalCoordinateSystemTag> (vg); 
  GlobalXYZVector vg2 = rot(vg); 
  iret |= compare(vg2.R(), vg.R(),"rot3D" );

  GlobalXYZPoint pg2 = rot(pg); 
  iret |= compare(pg2.X(), vg2.X(),"x diff");
  iret |= compare(pg2.Y(), vg2.Y(),"y diff");
  iret |= compare(pg2.Z(), vg2.Z(),"z diff");


  Quaternion qrot(rot); 

  pg2 = qrot(pg); 
  iret |= compare(pg2.X(), vg2.X(),"x diff",10);
  iret |= compare(pg2.Y(), vg2.Y(),"y diff",10);
  iret |= compare(pg2.Z(), vg2.Z(),"z diff",10);

  GlobalPolar3DVector vpg2 = qrot * vpg; 

  iret |= compare(vpg2.X(), vg2.X(),"x diff",10 );
  iret |= compare(vpg2.Y(), vg2.Y(),"y diff",10 );
  iret |= compare(vpg2.Z(), vg2.Z(),"z diff",10 );

  AxisAngle arot(rot); 
  pg2 = arot(pg); 
  iret |= compare(pg2.X(), vg2.X(),"x diff",10 );
  iret |= compare(pg2.Y(), vg2.Y(),"y diff",10 );
  iret |= compare(pg2.Z(), vg2.Z(),"z diff",10 );

  vpg2 = arot (vpg);
  iret |= compare(vpg2.X(), vg2.X(),"x diff",10 );
  iret |= compare(vpg2.Y(), vg2.Y(),"y diff",10 );
  iret |= compare(vpg2.Z(), vg2.Z(),"z diff",10 );
  
  EulerAngles erot(rot); 

  vpg2 = erot (vpg); 
  iret |= compare(vpg2.X(), vg2.X(),"x diff",10 );
  iret |= compare(vpg2.Y(), vg2.Y(),"y diff",10 );
  iret |= compare(vpg2.Z(), vg2.Z(),"z diff",10 );

  GlobalXYZVector vrx = RotationX(3) * vg; 
  GlobalXYZVector vry = RotationY(2) * vrx; 
  vpg2 = RotationZ(1) * GlobalPolar3DVector (vry); 
  iret |= compare(vpg2.X(), vg2.X(),"x diff",10 );
  iret |= compare(vpg2.Y(), vg2.Y(),"y diff",10 );
  iret |= compare(vpg2.Z(), vg2.Z(),"z diff",10 );

  // test Get/SetComponents
  XYZVector v1,v2,v3; 
  rot.GetComponents(v1,v2,v3);
  const Rotation3D rot2(v1,v2,v3); 
  //rot2.SetComponents(v1,v2,v3);
  double r1[9],r2[9]; 
  rot.GetComponents(r1,r1+9);
  rot2.GetComponents(r2,r2+9);
  for (int i = 0; i < 9; ++i) {
     iret |= compare(r1[i],r2[i],"Get/SetComponents"); 
  }
  // operator == fails for numerical precision
  //iret |= compare( (rot2==rot),true,"Get/SetComponens");

  // test get/set with a matrix
#ifndef NO_SMATRIX
  SMatrix<double,3> mat; 
  rot2.GetRotationMatrix(mat);
  rot.SetRotationMatrix(mat); 
  iret |= compare( (rot2==rot),true,"Get/SetRotMatrix");
#endif

  //test inversion
  Rotation3D rotInv = rot.Inverse();
  rot.Invert(); // invert in place
  bool comp = (rotInv == rot ); 
  iret |= compare(comp,true,"inversion");

  // rotation and scaling of points
  XYZPoint q1(1.,2,3); double a = 3;
  XYZPoint qr1 =  rot( a * q1);
  XYZPoint qr2 =  a * rot( q1);
  iret |= compare(qr1.X(), qr2.X(),"x diff",10 );
  iret |= compare(qr1.Y(), qr2.Y(),"y diff",10 );
  iret |= compare(qr1.Z(), qr2.Z(),"z diff",10 );


  if (iret == 0) std::cout << "\tOK\n"; 
  else std::cout << "\t FAILED\n";

  return iret; 
}


int testTransform3D() { 


  std::cout << "testing 3D Transform\t:\t"; 
  int iret = 0; 

  EulerAngles r(1.,2.,3.);

  GlobalPolar3DVector v(1.,2.,3.);
  GlobalXYZVector w(v);

  Transform3D t1( v );
  GlobalXYZPoint pg; 
  t1.Transform( LocalXYZPoint(), pg ); 
  iret |= compare(pg.X(), v.X(),"x diff",10 );
  iret |= compare(pg.Y(), v.Y(),"y diff",10 );
  iret |= compare(pg.Z(), v.Z(),"z diff",10 );


  Transform3D t2( r, v );

  GlobalPolar3DVector vr = r.Inverse()*v; 

//   std::cout << GlobalXYZVector(v) << std::endl; 
//   std::cout << GlobalXYZVector(vr) << std::endl; 
//   std::cout << GlobalXYZVector (r(v)) << std::endl; 
//   std::cout << GlobalXYZVector (r(vr)) << std::endl; 
//   std::cout << vr << std::endl; 
//   std::cout << r(vr) << std::endl; 

  

//   std::cout << r << std::endl; 
//   std::cout << r.Inverse() << std::endl; 
//   std::cout << r * r.Inverse() << std::endl; 
//   std::cout << Rotation3D(r) * Rotation3D(r.Inverse()) << std::endl; 
//   std::cout << Rotation3D(r) * Rotation3D(r).Inverse() << std::endl; 






  Transform3D t3( vr, r );

  //std::cout << t2 << std::endl; 
  //std::cout << t3 << std::endl; 

  XYZPoint p1(-1.,2.,-3);
  
  XYZPoint p2 = t2 (p1); 
  Polar3DPoint p3 = t3 ( Polar3DPoint(p1) ); 
  iret |= compare(p3.X(), p2.X(),"x diff",10 );
  iret |= compare(p3.Y(), p2.Y(),"y diff",10 );
  iret |= compare(p3.Z(), p2.Z(),"z diff",10 );

  GlobalXYZVector v1(1.,2.,3.);
  LocalXYZVector v2;  t2.Transform (v1, v2); 
  GlobalPolar3DVector v3;  t3.Transform ( GlobalPolar3DVector(v1), v3 ); 

  iret |= compare(v3.X(), v2.X(),"x diff",10 );
  iret |= compare(v3.Y(), v2.Y(),"y diff",10 );
  iret |= compare(v3.Z(), v2.Z(),"z diff",10 );

  XYZPoint q1(1,2,3);
  XYZPoint q2(-1,-2,-3);
  XYZPoint q3 = q1 +  XYZVector(q2);
  //std::cout << q3 << std::endl; 
  XYZPoint qt3 = t3(q3); 
  //std::cout << qt3 << std::endl; 
  XYZPoint qt1 = t3(q1);
  XYZVector vt2 = t3( XYZVector(q2) );
  XYZPoint qt4 = qt1 + vt2; 
  iret |= compare(qt3.X(), qt4.X(),"x diff",10 );
  iret |= compare(qt3.Y(), qt4.Y(),"y diff",10 );
  iret |= compare(qt3.Z(),  qt4.Z(),"z diff",10 );
     //std::cout << qt4 << std::endl; 

  // this fails
//  double a = 3;
  //XYZPoint q4 = a*q1;
//   std::cout << t3( a * q1) << std::endl;
//   std::cout << a * t3(q1) << std::endl;

  // test get/set with a matrix
#ifndef NO_SMATRIX
  SMatrix<double,3,4> mat; 
  t3.GetTransformMatrix(mat);
  Transform3D t3b;  t3b.SetTransformMatrix(mat); 
  iret |= compare( (t3==t3b),true,"Get/SetTransformMatrix");

  // test LR 
  Boost b(0.2,0.4,0.8); 
  LorentzRotation lr(b); 
  SMatrix<double,4> mat4; 
  lr.GetRotationMatrix(mat4);
  LorentzRotation lr2;  lr2.SetRotationMatrix(mat4); 
  iret |= compare( (lr==lr2),true,"Get/SetLRotMatrix");
#endif

  if (iret == 0) std::cout << "\t\t\t\tOK\n"; 
  else std::cout << "\t\t\t\tFAILED\n"; 

  return iret; 
}


int testVectorUtil() { 

  std::cout << "testing VectorUtil  \t:\t"; 
   int iret = 0;

   // test new perp functions
   XYZVector v(1.,2.,3.); 

   XYZVector vx = ProjVector(v,XYZVector(3,0,0) );
   iret |= compare(vx.X(), v.X(),"x",1 );
   iret |= compare(vx.Y(), 0,"y",1 );
   iret |= compare(vx.Z(), 0,"z",1 );

   XYZVector vpx = PerpVector(v,XYZVector(2,0,0) );
   iret |= compare(vpx.X(), 0,"x",1 );
   iret |= compare(vpx.Y(), v.Y(),"y",1 );
   iret |= compare(vpx.Z(), v.Z(), "z",1 );

   double perpy = Perp(v, XYZVector(0,2,0) );  
   iret |= compare(perpy, std::sqrt( v.Mag2() - v.y()*v.y()),"perpy" );

   XYZPoint  u(1,1,1); 
   XYZPoint  un = u/u.R(); 
   

   XYZVector vl = ProjVector(v,u);
   XYZVector vl2 = XYZVector(un) * ( v.Dot(un ) ); 

   iret |= compare(vl.X(), vl2.X(),"x",1 );
   iret |= compare(vl.Y(), vl2.Y(),"y",1 );
   iret |= compare(vl.Z(), vl2.Z(),"z",1 );

   XYZVector vp = PerpVector(v,u);
   XYZVector vp2 = v - XYZVector ( un * ( v.Dot(un ) ) ); 
   iret |= compare(vp.X(), vp2.X(),"x",10 );
   iret |= compare(vp.Y(), vp2.Y(),"y",10 );
   iret |= compare(vp.Z(), vp2.Z(),"z",10 );

   double perp = Perp(v,u);
   iret |= compare(perp, vp.R(),"perp",1 );
   double perp2 = Perp2(v,u);
   iret |= compare(perp2, vp.Mag2(),"perp2",1 );

  if (iret == 0) std::cout << "\t\t\t\tOK\n"; 
  else std::cout << "\t\t\t\t\t\tFAILED\n"; 
  return iret; 

}

int main() { 

  int iret = 0; 
  iret |= testVector3D(); 
  iret |= testPoint3D(); 

  iret |= testRotations3D(); 

  iret |= testTransform3D();

  iret |= testVectorUtil();


  if (iret !=0) std::cout << "\nTest GenVector FAILED!!!!!!!!!\n";
  return iret; 
}
