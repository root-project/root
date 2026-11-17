#include "Math/BoostX.h"
#include "Math/Boost.h"
#include "Math/BoostY.h"
#include "Math/BoostZ.h"
#include "Math/LorentzRotation.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/PtEtaPhiM4D.h"
#include "Math/PxPyPzM4D.h"
#include "Math/VectorUtil.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

#include <iterator>

using namespace ROOT::Math;

bool AlmostEq(const XYZTVector &v1, const XYZTVector &v2)
{
   const double eps = 0.00000001;
   return std::abs(v2.X() - v1.X()) < eps && std::abs(v2.Y() - v1.Y()) < eps && std::abs(v2.Z() - v1.Z()) < eps &&
          std::abs(v2.T() - v1.T()) < eps;
}

// A coordinate system passed directly to a boost is a foreign 4-vector whose constructor does not take
// (x, y, z, t) (#19635).
template <class B, class C>
int TestCoordinateSystem(const B &b, const char *name, const XYZTVector &v)
{
   C c = LorentzVector<C>(v).Coordinates();
   C bc = b(c);
   if (AlmostEq(XYZTVector(bc.Px(), bc.Py(), bc.Pz(), bc.E()), b(v)))
      return 0;
   std::cout << name << " acting on a coordinate system gives a wrong result" << std::endl;
   return 1;
}

template <class B>
int TestCoordinateSystems(const B &b, const char *name, const XYZTVector &v)
{
   return TestCoordinateSystem<B, PtEtaPhiE4D<double>>(b, name, v) +
          TestCoordinateSystem<B, PtEtaPhiM4D<double>>(b, name, v) +
          TestCoordinateSystem<B, PxPyPzM4D<double>>(b, name, v);
}

int main()
{

   BoostX bx(0.8);
   std::cout << "BoostX - beta : " << bx.Beta() << "  gamma : " << bx.Gamma() << std::endl;

   XYZTVector v(1., 2., 3., 4.);

   XYZTVector vb1 = bx(v);
   XYZTVector vb2 = VectorUtil::boostX(v, 0.8);

   int nFailedTests = 0;

   if (!AlmostEq(vb1, vb2)) {
      std::cout << "BoostX test failed" << std::endl;
      int pr = std::cout.precision(18);
      std::cout << vb1 << std::endl;
      std::cout << vb2 << std::endl;
      std::cout.precision(pr);
      nFailedTests++;
   }

   // Polar3DVector bv(1.,2.,0.8);
   Polar3DVector bv(0.99999, 1., 2);
   std::cout << "BoostVector " << XYZVector(bv) << " beta boost = " << XYZVector(bv).R() << std::endl;
   Boost b(bv);
   std::cout << "Boost Components : ";
   std::ostream_iterator<double> oi(std::cout, "\t");
   b.GetComponents(oi);
   std::cout << std::endl;

   vb1 = b(v);
   vb2 = VectorUtil::boost(v, bv);
   if (!AlmostEq(vb1, vb2)) {
      std::cout << "Boost test failed" << std::endl;
      int pr = std::cout.precision(18);
      std::cout << vb1 << std::endl;
      std::cout << vb2 << std::endl;
      std::cout.precision(pr);
      nFailedTests++;
   }

   nFailedTests += TestCoordinateSystems(bx, "BoostX", v);
   nFailedTests += TestCoordinateSystems(BoostY(0.6), "BoostY", v);
   nFailedTests += TestCoordinateSystems(BoostZ(-0.7), "BoostZ", v);
   nFailedTests += TestCoordinateSystems(b, "Boost", v);
   nFailedTests += TestCoordinateSystems(LorentzRotation(b), "LorentzRotation", v);

   return nFailedTests;
}
