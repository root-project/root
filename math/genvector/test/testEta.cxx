// Tests of Eta() for the coordinate systems that compute eta from rho and z.

#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>

using namespace ROOT::Math;

namespace {

// Error of eta against the reference, in units of the spacing of T at the reference.
template <typename T>
long double ErrorInUlp(T eta, long double reference)
{
   const T r = std::fabs(static_cast<T>(reference));
   const long double ulp = static_cast<long double>(std::nextafter(r, std::numeric_limits<T>::infinity())) - r;
   return std::fabs(eta - reference) / ulp;
}

// Calls check(eta(+z), eta(-z), rho, z) for each coordinate system,
// on a grid of rho and z > 0 that covers z / rho from 3e-5 to 7e8.
template <typename T, typename Check>
void ForEachEta(Check check)
{
   for (int i = 0; i < 34; ++i) {
      const T rho = T(0.01) * std::pow(T(1.37), T(i));
      const T x = T(0.6) * rho;
      const T y = T(0.8) * rho;
      for (int j = 0; j < 108; ++j) {
         const T z = T(0.01) * std::pow(T(1.21), T(j));
         SCOPED_TRACE(testing::Message() << "rho = " << rho << ", z = " << z);

         const DisplacementVector3D<Cylindrical3D<T>> c1(rho, z, T(1)), c2(rho, -z, T(1));
         check(c1.Eta(), c2.Eta(), c1.Rho(), z);

         const DisplacementVector3D<Cartesian3D<T>> v1(x, y, z), v2(x, y, -z);
         check(v1.Eta(), v2.Eta(), v1.Rho(), z);

         const T e = T(2) * (rho + z);
         const LorentzVector<PxPyPzE4D<T>> p1(x, y, z, e), p2(x, y, -z, e);
         check(p1.Eta(), p2.Eta(), p1.Pt(), z);

         const LorentzVector<PxPyPzM4D<T>> m1(x, y, z, T(1)), m2(x, y, -z, T(1));
         check(m1.Eta(), m2.Eta(), m1.Pt(), z);
      }
   }
}

} // namespace

// Eta is an odd function of z, bit for bit.
TEST(GenVectorEta, OddInZ)
{
   ForEachEta<float>([](float etaPlus, float etaMinus, float, float) { EXPECT_EQ(etaMinus, -etaPlus); });
   ForEachEta<double>([](double etaPlus, double etaMinus, double, double) { EXPECT_EQ(etaMinus, -etaPlus); });
}

// Eta of a float vector is within 2 ULP of asinh(z / rho), for both signs of z.
TEST(GenVectorEta, FloatPrecision)
{
   ForEachEta<float>([](float etaPlus, float etaMinus, float rho, float z) {
      const long double reference = std::asinh(static_cast<long double>(z) / static_cast<long double>(rho));
      EXPECT_LE(ErrorInUlp(etaPlus, reference), 2);
      EXPECT_LE(ErrorInUlp(etaMinus, -reference), 2);
   });
}
