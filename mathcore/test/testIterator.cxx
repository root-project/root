
#include "Math/Vector3D.h"
#include "Math/Point3D.h"
#include "Math/Vector4D.h"
#include "Math/EulerAngles.h"

#include "Math/Transform3D.h"
#include "Math/LorentzRotation.h"
#include "Math/Boost.h"

#include "Math/Rotation3D.h"
#include "Math/RotationX.h"
#include "Math/RotationY.h"
#include "Math/RotationZ.h"
#include "Math/Quaternion.h"
#include "Math/AxisAngle.h"
#include "Math/EulerAngles.h"

#include "Math/VectorUtil.h"

#include <iostream>
#include <iterator> 
#include <vector>

using namespace ROOT::Math;
using namespace ROOT::Math::VectorUtil;

template <class Rot> 
void printRot(const Rot & rot) {
   std::cout << "rot:  ( " ;
   std::ostream_iterator<double> oi(std::cout,"  ");
   rot.GetComponents(oi); 
   std::cout << ") " << std::endl;
}   
template<class V> 
void printVec(const V & v ) {
   std::cout << "vec : ( " ;
   std::ostream_iterator<double> oi(std::cout,"  ");
   v.GetCoordinates(oi); 
   std::cout << ") " << std::endl;
}   
int main() { 

   std::vector<double> data(16); 
   XYZVector v(1.,2.,3);
   printVec(v); 
   XYZPoint p(v); printVec(p);
   XYZTVector q(1.,2,3,4); printVec(q); 

   AxisAngle ar(v,3.); printRot(ar); 
   EulerAngles er(ar); printRot(er); 
   Quaternion qr(er); printRot(qr); 
   Rotation3D rr(qr) ; printRot(rr); 
 
   Transform3D t(rr,v); printRot(t); 

   
   Boost b(0.9*v/(v.R())); printRot(b); 
   LorentzRotation lr(rr); printRot(lr);
   LorentzRotation lr2(b); printRot(lr2);

}
