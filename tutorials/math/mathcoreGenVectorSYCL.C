/// \file
/// \ingroup tutorial_math
/// \notebook -nodraw
/// Example macro testing available methods and operation of the GenVectorX classes.
///
/// \note This tutorial requires ROOT to be built with **SYCL support** and **GenVectorX** enabled.
/// Configure CMake with:
///   `-Dexperimental_adaptivecpp=ON -Dexperimental_genvectorx=ON`
///
/// The results are compared and checked
/// The macro is divided in 3 parts:
///    - testVector3D          :  tests of the 3D Vector classes
///    - testPoint3D           :  tests of the 3D Point classes
///    - testLorentzVector     :  tests of the 4D LorentzVector classes
///
/// \macro_code
///
/// \author Devajith Valaparambil Sreeramaswamy (CERN)

#define ROOT_MATH_ARCH MathSYCL
#define ROOT_MATH_SYCL

#include "MathX/Vector3D.h"
#include "MathX/Point3D.h"

#include "MathX/Vector2D.h"
#include "MathX/Point2D.h"

#include "MathX/EulerAngles.h"

#include "MathX/Transform3D.h"
#include "MathX/Translation3D.h"

#include "MathX/Rotation3D.h"
#include "MathX/RotationX.h"
#include "MathX/RotationY.h"
#include "MathX/RotationZ.h"
#include "MathX/Quaternion.h"
#include "MathX/AxisAngle.h"
#include "MathX/RotationZYX.h"

#include "MathX/LorentzRotation.h"
#include "MathX/PtEtaPhiM4D.h"
#include "MathX/LorentzVector.h"

#include "MathX/VectorUtil.h"

#include <sycl/sycl.hpp>

#include <vector>

using namespace ROOT::ROOT_MATH_ARCH;
using namespace ROOT::ROOT_MATH_ARCH::VectorUtil;

typedef DisplacementVector3D<Cartesian3D<double>, GlobalCoordinateSystemTag> GlobalXYZVector;
typedef DisplacementVector3D<Cartesian3D<double>, LocalCoordinateSystemTag> LocalXYZVector;
typedef DisplacementVector3D<Polar3D<double>, GlobalCoordinateSystemTag> GlobalPolar3DVector;

typedef PositionVector3D<Cartesian3D<double>, GlobalCoordinateSystemTag> GlobalXYZPoint;
typedef PositionVector3D<Cartesian3D<double>, LocalCoordinateSystemTag> LocalXYZPoint;
typedef PositionVector3D<Polar3D<double>, GlobalCoordinateSystemTag> GlobalPolar3DPoint;
typedef PositionVector3D<Polar3D<double>, LocalCoordinateSystemTag> LocalPolar3DPoint;

int ntest = 0;
int nfail = 0;
int ok = 0;

int compare(double v1, double v2, double scale = 1.0)
{
   ntest = ntest + 1;

   // numerical double limit for epsilon
   double eps = scale * std::numeric_limits<double>::epsilon();
   int iret = 0;
   double delta = v2 - v1;
   double d = 0;
   if (delta < 0)
      delta = -delta;
   if (v1 == 0 || v2 == 0) {
      if (delta > eps) {
         iret = 1;
      }
   }
   // skip case v1 or v2 is infinity
   else {
      d = v1;

      if (v1 < 0)
         d = -d;
      // add also case when delta is small by default
      if (delta / d > eps && delta > eps)
         iret = 1;
   }

   return iret;
}

template <class Transform>
bool IsEqual(const Transform &t1, const Transform &t2, unsigned int size)
{
   // size should be an enum of the Transform class
   std::vector<double> x1(size);
   std::vector<double> x2(size);
   t1.GetComponents(x1.begin(), x1.end());
   t2.GetComponents(x2.begin(), x2.end());
   bool ret = true;
   unsigned int i = 0;
   while (ret && i < size) {
      // from TMath::AreEqualRel(x1,x2,2*eps)
      bool areEqual =
         std::abs(x1[i] - x2[i]) < std::numeric_limits<double>::epsilon() * (std::abs(x1[i]) + std::abs(x2[i]));
      ret &= areEqual;
      i++;
   }
   return ret;
}

int testVector3D()
{
   std::cout << "\n************************************************************************\n " << " Vector 3D Test"
             << "\n************************************************************************\n";

   sycl::buffer<int, 1> ok_buf(&ok, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   {
      queue.submit([&](sycl::handler &cgh) {
         auto ok_device = ok_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testVector3D>([=]() {
            // test the vector tags

            GlobalXYZVector vg(1., 2., 3.);
            GlobalXYZVector vg2(vg);
            GlobalPolar3DVector vpg(vg);

            ok_device[0] += compare(vpg.R(), vg2.R());

            //   std::cout << vg2 << std::endl;

            double r = vg.Dot(vpg);
            ok_device[0] += compare(r, vg.Mag2());

            GlobalXYZVector vcross = vg.Cross(vpg);
            ok_device[0] += compare(vcross.R(), 0.0, 10);

            //   std::cout << vg.Dot(vpg) << std::endl;
            //   std::cout << vg.Cross(vpg) << std::endl;

            GlobalXYZVector vg3 = vg + vpg;
            ok_device[0] += compare(vg3.R(), 2 * vg.R());

            GlobalXYZVector vg4 = vg - vpg;
            ok_device[0] += compare(vg4.R(), 0.0, 10);
         });
      });
   }

   if (ok == 0)
      std::cout << "\t\t OK " << std::endl;

   return ok;
}

int testPoint3D()
{
   std::cout << "\n************************************************************************\n " << " Point 3D Tests"
             << "\n************************************************************************\n";

   sycl::buffer<int, 1> ok_buf(&ok, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   {
      queue.submit([&](sycl::handler &cgh) {
         auto ok_device = ok_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testPoint3D>([=]() {
            // test the vector tags

            GlobalXYZPoint pg(1., 2., 3.);
            GlobalXYZPoint pg2(pg);

            GlobalPolar3DPoint ppg(pg);

            ok_device[0] += compare(ppg.R(), pg2.R());
            // std::cout << pg2 << std::endl;

            GlobalXYZVector vg(pg);

            double r = pg.Dot(vg);
            ok_device[0] += compare(r, pg.Mag2());

            GlobalPolar3DVector vpg(pg);
            GlobalXYZPoint pcross = pg.Cross(vpg);
            ok_device[0] += compare(pcross.R(), 0.0, 10);

            GlobalPolar3DPoint pg3 = ppg + vg;
            ok_device[0] += compare(pg3.R(), 2 * pg.R());

            GlobalXYZVector vg4 = pg - ppg;
            ok_device[0] += compare(vg4.R(), 0.0, 10);

            // operator -
            XYZPoint q1(1., 2., 3.);
            XYZPoint q2 = -1. * q1;
            XYZVector v2 = -XYZVector(q1);
            ok_device[0] += compare(XYZVector(q2) == v2, true);
         });
      });
   }

   if (ok == 0)
      std::cout << "\t OK " << std::endl;

   return ok;
}

int testLorentzVector()
{
   std::cout << "\n************************************************************************\n "
             << " Lorentz Vector Tests"
             << "\n************************************************************************\n";

   sycl::buffer<int, 1> ok_buf(&ok, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto ok_device = ok_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testRotations3D>([=]() {
            LorentzVector<PtEtaPhiM4D<float>> v1(1, 2, 3, 4);
            LorentzVector<PtEtaPhiM4D<float>> v2(5, 6, 7, 8);
            ok_device[0] += compare(v1.DeltaR(v2), 4.60575f);

            LorentzVector<PtEtaPhiM4D<float>> v = v1 + v2;
            ok_device[0] += compare(v.M(), 62.03058f);
         });
      });
   }

   if (ok == 0)
      std::cout << "\tOK\n";
   else
      std::cout << "\t FAILED\n";

   return ok;
}

void mathcoreGenVectorSYCL()
{

   testVector3D();
   testPoint3D();
   testLorentzVector();

   std::cout << "\n\nNumber of tests " << ntest << " failed = " << nfail << std::endl;
}
