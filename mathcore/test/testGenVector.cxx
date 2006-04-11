#include "Math/Transform3D.h"
#include "Math/Vector3D.h"
#include "Math/Point3D.h"
#include "Math/EulerAngles.h"

#include "Math/Rotation3D.h"
#include "Math/RotationX.h"
#include "Math/RotationY.h"
#include "Math/RotationZ.h"
#include "Math/Quaternion.h"
#include "Math/AxisAngle.h"
#include "Math/EulerAngles.h"

#include <iostream>

using namespace ROOT::Math;



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

  iret |= compare(vpg.R(), vg.R() );
 
//   std::cout << vg2 << std::endl;

  double r = vg.Dot(vpg); 
  iret |= compare(r, vg.Mag2() );

  GlobalXYZVector vcross = vg.Cross(vpg); 
  iret |= compare(vcross.R(), 0.0,"cross",10 );

//   std::cout << vg.Dot(vpg) << std::endl;
//   std::cout << vg.Cross(vpg) << std::endl;


  
  LocalXYZVector vl;

  GlobalXYZVector vg3 = vg + vpg;  
  iret |= compare(vg3.R(), 2*vg.R() );

  GlobalXYZVector vg4 = vg - vpg;  
  iret |= compare(vg4.R(), 0.0,"diff",10 );



#ifdef TEST_COMPILE_ERROR
  vl = vg; 
  LocalXYZVector vl2(vg2);
  LocalXYZVector vl3(vpg);
  vg.Dot(vl);
  vg.Cross(vl);
  vg3 = vg + vl;
  vg4 = vg - vl;
#endif


  if (iret == 0) std::cout << "\t\t\tOK\n"; 
  else std::cout << "\t\tFAILED\n";


  return iret;
}



int testPoint3D() { 

  int iret = 0;

  std::cout << "testing Point3D    \t:\t"; 

  // test the vector tags 

  GlobalXYZPoint pg(1.,2.,3.); 
  GlobalXYZPoint pg2(pg); 

  GlobalPolar3DPoint ppg(pg);

  iret |= compare(ppg.R(), pg.R() );
  //std::cout << pg2 << std::endl;
  
  LocalXYZPoint pl;


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
  pl = pg; 
  LocalXYZVector pl2(pg2);
  LocalXYZVector pl3(ppg);
  pl.Dot(vg);
  pl.Cross(vg);
  pg3 = ppg + pg;
  pg3 = ppg + pl;
  vg4 = pg - pl;
#endif

  if (iret == 0) std::cout << "\t\t\tOK\n"; 
  else std::cout << "\t\tFAILED\n"; 

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



  if (iret == 0) std::cout << "\tOK\n"; 
  else std::cout << "\t\FAILED\n";

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



  if (iret == 0) std::cout << "\t\tOK\n"; 
  else std::cout << "\t\tFAILED\n"; 

  return 0; 
}


int main() { 

  int iret = 0; 
  iret |= testVector3D(); 
  iret |= testPoint3D(); 

  iret |= testRotations3D(); 

  iret |= testTransform3D();


  if (iret !=0) std::cout << "\nTest GenVector FAILED!!!!!!!!!\n";
  return iret; 
}
