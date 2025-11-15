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

#include "MathX/GenVectorX/AccHeaders.h"

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

int compare(double v1, double v2, double scale = 1.0)
{
   //  ntest = ntest + 1;

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

   //    if (iret == 0)
   //       std::cout << ".";
   //    else {
   //       int pr = std::cout.precision(18);
   //       std::cout << "\nDiscrepancy in " << name << "() : " << v1 << " != " << v2 << " discr = " << int(delta / d /
   //       eps)
   //                 << "   (Allowed discrepancy is " << eps << ")\n";
   //       std::cout.precision(pr);
   //       // nfail = nfail + 1;
   //    }
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
   int iret_host = 0;

   std::cout << "testing Vector3D   \t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testVector3D>([=]() {
            // test the vector tags

            GlobalXYZVector vg(1., 2., 3.);
            GlobalXYZVector vg2(vg);
            GlobalPolar3DVector vpg(vg);

            iret[0] |= compare(vpg.R(), vg2.R());

            //   std::cout << vg2 << std::endl;

            double r = vg.Dot(vpg);
            iret[0] |= compare(r, vg.Mag2());

            GlobalXYZVector vcross = vg.Cross(vpg);
            iret[0] |= compare(vcross.R(), 0.0, 10);

            //   std::cout << vg.Dot(vpg) << std::endl;
            //   std::cout << vg.Cross(vpg) << std::endl;

            GlobalXYZVector vg3 = vg + vpg;
            iret[0] |= compare(vg3.R(), 2 * vg.R());

            GlobalXYZVector vg4 = vg - vpg;
            iret[0] |= compare(vg4.R(), 0.0, 10);

#ifdef TEST_COMPILE_ERROR
            LocalXYZVector vl;
            vl = vg;
            LocalXYZVector vl2(vg2);
            LocalXYZVector vl3(vpg);
            vg.Dot(vl);
            vg.Cross(vl);
            vg3 = vg + vl;
            vg4 = vg - vl;
#endif
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\t\t\t\t\tOK\n";
   else
      std::cout << "\t\t\t\tFAILED\n";

   return iret_host;
}

int testPoint3D()
{

   int iret_host = 0;

   std::cout << "testing Point3D    \t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testPoint3D>([=]() {
            // test the vector tags

            GlobalXYZPoint pg(1., 2., 3.);
            GlobalXYZPoint pg2(pg);

            GlobalPolar3DPoint ppg(pg);

            iret[0] |= compare(ppg.R(), pg2.R());
            // std::cout << pg2 << std::endl;

            GlobalXYZVector vg(pg);

            double r = pg.Dot(vg);
            iret[0] |= compare(r, pg.Mag2());

            GlobalPolar3DVector vpg(pg);
            GlobalXYZPoint pcross = pg.Cross(vpg);
            iret[0] |= compare(pcross.R(), 0.0, 10);

            GlobalPolar3DPoint pg3 = ppg + vg;
            iret[0] |= compare(pg3.R(), 2 * pg.R());

            GlobalXYZVector vg4 = pg - ppg;
            iret[0] |= compare(vg4.R(), 0.0, 10);

#ifdef TEST_COMPILE_ERROR
            LocalXYZPoint pl;
            pl = pg;
            LocalXYZVector pl2(pg2);
            LocalXYZVector pl3(ppg);
            pl.Dot(vg);
            pl.Cross(vg);
            pg3 = ppg + pg;
            pg3 = ppg + pl;
            vg4 = pg - pl;
#endif

            // operator -
            XYZPoint q1(1., 2., 3.);
            XYZPoint q2 = -1. * q1;
            XYZVector v2 = -XYZVector(q1);
            iret[0] |= compare(XYZVector(q2) == v2, true);
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\t\t\t\t\tOK\n";
   else
      std::cout << "\t\t\t\tFAILED\n";

   return iret_host;
}

typedef DisplacementVector2D<Cartesian2D<double>, GlobalCoordinateSystemTag> GlobalXYVector;
typedef DisplacementVector2D<Cartesian2D<double>, LocalCoordinateSystemTag> LocalXYVector;
typedef DisplacementVector2D<Polar2D<double>, GlobalCoordinateSystemTag> GlobalPolar2DVector;

int testVector2D()
{

   int iret_host = 0;

   std::cout << "testing Vector2D   \t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testVector2D>([=]() {
            // test the vector tags

            GlobalXYVector vg(1., 2.);
            GlobalXYVector vg2(vg);

            GlobalPolar2DVector vpg(vg);

            iret[0] |= compare(vpg.R(), vg2.R());

            //   std::cout << vg2 << std::endl;

            double r = vg.Dot(vpg);
            iret[0] |= compare(r, vg.Mag2());

            //   std::cout << vg.Dot(vpg) << std::endl;

            GlobalXYVector vg3 = vg + vpg;
            iret[0] |= compare(vg3.R(), 2 * vg.R());

            GlobalXYVector vg4 = vg - vpg;
            iret[0] |= compare(vg4.R(), 0.0, 10);

            double angle = 1.;
            vg.Rotate(angle);
            iret[0] |= compare(vg.Phi(), vpg.Phi() + angle);
            iret[0] |= compare(vg.R(), vpg.R());

            GlobalXYZVector v3d(1, 2, 0);
            GlobalXYZVector vr3d = RotationZ(angle) * v3d;
            iret[0] |= compare(vg.X(), vr3d.X());
            iret[0] |= compare(vg.Y(), vr3d.Y());

            GlobalXYVector vu = vg3.Unit();
            iret[0] |= compare(vu.R(), 1.);

#ifdef TEST_COMPILE_ERROR
            LocalXYVector vl;
            vl = vg;
            LocalXYVector vl2(vg2);
            LocalXYVector vl3(vpg);
            vg.Dot(vl);
            vg3 = vg + vl;
            vg4 = vg - vl;
#endif
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\t\t\t\tOK\n";
   else
      std::cout << "\t\t\tFAILED\n";

   return iret_host;
}

typedef PositionVector2D<Cartesian2D<double>, GlobalCoordinateSystemTag> GlobalXYPoint;
typedef PositionVector2D<Cartesian2D<double>, LocalCoordinateSystemTag> LocalXYPoint;
typedef PositionVector2D<Polar2D<double>, GlobalCoordinateSystemTag> GlobalPolar2DPoint;
typedef PositionVector2D<Polar2D<double>, LocalCoordinateSystemTag> LocalPolar2DPoint;

int testPoint2D()
{

   int iret_host = 0;

   std::cout << "testing Point2D    \t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::queue queue{sycl::default_selector_v};

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testPoint2D>([=]() {
            // test the vector tags

            GlobalXYPoint pg(1., 2.);
            GlobalXYPoint pg2(pg);

            GlobalPolar2DPoint ppg(pg);

            iret[0] |= compare(ppg.R(), pg2.R());
            // std::cout << pg2 << std::endl;

            GlobalXYVector vg(pg);

            double r = pg.Dot(vg);
            iret[0] |= compare(r, pg.Mag2());

            GlobalPolar2DVector vpg(pg);

            GlobalPolar2DPoint pg3 = ppg + vg;
            iret[0] |= compare(pg3.R(), 2 * pg.R());

            GlobalXYVector vg4 = pg - ppg;
            iret[0] |= compare(vg4.R(), 0.0, 10);

#ifdef TEST_COMPILE_ERROR
            LocalXYPoint pl;
            pl = pg;
            LocalXYVector pl2(pg2);
            LocalXYVector pl3(ppg);
            pl.Dot(vg);
            pl.Cross(vg);
            pg3 = ppg + pg;
            pg3 = ppg + pl;
            vg4 = pg - pl;
#endif

            // operator -
            XYPoint q1(1., 2.);
            XYPoint q2 = -1. * q1;
            XYVector v2 = -XYVector(q1);
            iret[0] |= compare(XYVector(q2) == v2, true);

            double angle = 1.;
            pg.Rotate(angle);
            iret[0] |= compare(pg.Phi(), ppg.Phi() + angle);
            iret[0] |= compare(pg.R(), ppg.R());

            GlobalXYZVector v3d(1, 2, 0);
            GlobalXYZVector vr3d = RotationZ(angle) * v3d;
            iret[0] |= compare(pg.X(), vr3d.X());
            iret[0] |= compare(pg.Y(), vr3d.Y());
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\t\t\t\tOK\n";
   else
      std::cout << "\t\t\tFAILED\n";

   return iret_host;
}

int testRotations3D()
{

   int iret_host = 0;
   std::cout << "testing 3D Rotations\t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testRotations3D>([=]() {
            // RotationZYX rotZ = RotationZYX(RotationZ(1));
            // RotationZYX rotY = RotationZYX(RotationY(2));
            // RotationZYX rotX = RotationZYX(RotationX(3));

            RotationZ rotZ = RotationZ(1);
            RotationY rotY = RotationY(2);
            RotationX rotX = RotationX(3);

            Rotation3D rot = rotZ * (rotY * rotX); // RotationZ(1.) * RotationY(2) * RotationX(3);
            GlobalXYZVector vg(1., 2., 3);
            GlobalXYZPoint pg(1., 2., 3);
            GlobalPolar3DVector vpg(vg);

            // GlobalXYZVector vg2 = rot.operator()<Cartesian3D,GlobalCoordinateSystemTag, GlobalCoordinateSystemTag>
            // (vg);
            GlobalXYZVector vg2 = rot(vg);
            iret[0] |= compare(vg2.R(), vg.R());

            GlobalXYZPoint pg2 = rot(pg);
            iret[0] |= compare(pg2.X(), vg2.X());
            iret[0] |= compare(pg2.Y(), vg2.Y());
            iret[0] |= compare(pg2.Z(), vg2.Z());

            Quaternion qrot(rot);

            pg2 = qrot(pg);
            iret[0] |= compare(pg2.X(), vg2.X(), 10);
            iret[0] |= compare(pg2.Y(), vg2.Y(), 10);
            iret[0] |= compare(pg2.Z(), vg2.Z(), 10);

            GlobalPolar3DVector vpg2 = qrot * vpg;
            iret[0] |= compare(vpg2.X(), vg2.X(), 10);
            iret[0] |= compare(vpg2.Y(), vg2.Y(), 10);
            iret[0] |= compare(vpg2.Z(), vg2.Z(), 10);

            AxisAngle arot(rot);
            pg2 = arot(pg);
            iret[0] |= compare(pg2.X(), vg2.X(), 10);
            iret[0] |= compare(pg2.Y(), vg2.Y(), 10);
            iret[0] |= compare(pg2.Z(), vg2.Z(), 10);

            vpg2 = arot(vpg);
            iret[0] |= compare(vpg2.X(), vg2.X(), 10);
            iret[0] |= compare(vpg2.Y(), vg2.Y(), 10);
            iret[0] |= compare(vpg2.Z(), vg2.Z(), 10);

            EulerAngles erot(rot);

            vpg2 = erot(vpg);
            iret[0] |= compare(vpg2.X(), vg2.X(), 10);
            iret[0] |= compare(vpg2.Y(), vg2.Y(), 10);
            iret[0] |= compare(vpg2.Z(), vg2.Z(), 10);

            GlobalXYZVector vrx = RotationX(3) * vg;
            GlobalXYZVector vry = RotationY(2) * vrx;
            vpg2 = RotationZ(1) * GlobalPolar3DVector(vry);
            iret[0] |= compare(vpg2.X(), vg2.X(), 10);
            iret[0] |= compare(vpg2.Y(), vg2.Y(), 10);
            iret[0] |= compare(vpg2.Z(), vg2.Z(), 10);

            // test Get/SetComponents
            XYZVector v1, v2, v3;
            rot.GetComponents(v1, v2, v3);
            const Rotation3D rot2(v1, v2, v3);
            // rot2.SetComponents(v1,v2,v3);
            double r1[9], r2[9];
            rot.GetComponents(r1, r1 + 9);
            rot2.GetComponents(r2, r2 + 9);
            for (int i = 0; i < 9; ++i) {
               iret[0] |= compare(r1[i], r2[i]);
            }
            // operator == fails for numerical precision
            // iret[0] |= compare( (rot2==rot),true,"Get/SetComponens");

            // test get/set with a matrix
            // #ifndef NO_SMATRIX
            //    SMatrix<double, 3> mat;
            //    rot2.GetRotationMatrix(mat);
            //    rot.SetRotationMatrix(mat);
            //    iret[0] |= compare((rot2 == rot), true);
            // #endif

            // test inversion
            Rotation3D rotInv = rot.Inverse();
            rot.Invert(); // invert in place
            bool comp = (rotInv == rot);
            iret[0] |= compare(comp, true);

            // rotation and scaling of points
            XYZPoint q1(1., 2, 3);
            double a = 3;
            XYZPoint qr1 = rot(a * q1);
            XYZPoint qr2 = a * rot(q1);
            iret[0] |= compare(qr1.X(), qr2.X(), 10);
            iret[0] |= compare(qr1.Y(), qr2.Y(), 10);
            iret[0] |= compare(qr1.Z(), qr2.Z(), 10);

            // test TVector3-like Rotate function around some axis by an angle. Test case taken from cwessel:
            // https://root-forum.cern.ch/t/tvector3-rotate-to-arbitrary-rotation-using-xyzvector/63244/7
            XYZVector ag(17, -4, -23);
            double angle = 100;
            XYZVector axisg(-23.4, 1.7, -0.3);
            XYZVector rotated = Rotate(ag, angle, axisg);
            // should be equivalent to:
            // TVector3 at(17, -4, -23);
            // TVector3 axist(-23.4, 1.7, -0.3);
            // at.Rotate(angle, axist);
            // at.Print();
            // (17.856456,8.106555,-21.199782)
            iret[0] |= compare(rotated.X(), 17.856456, 1e10);
            iret[0] |= compare(rotated.Y(), 8.106555, 1e10);
            iret[0] |= compare(rotated.Z(), -21.199782, 1e10);
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\tOK\n";
   else
      std::cout << "\t FAILED\n";

   return iret_host;
}

int testTransform3D()
{

   int iret_host = 0;
   std::cout << "testing 3D Transform\t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testRotations3D>([=]() {
            EulerAngles r(1., 2., 3.);

            GlobalPolar3DVector v(1., 2., 3.);
            GlobalXYZVector w(v);

            Transform3D t1(v);
            GlobalXYZPoint pg;
            t1.Transform(LocalXYZPoint(), pg);
            iret[0] |= compare(pg.X(), v.X(), 10);
            iret[0] |= compare(pg.Y(), v.Y(), 10);
            iret[0] |= compare(pg.Z(), v.Z(), 10);

            Transform3D t2(r, v);

            GlobalPolar3DVector vr = r.Inverse() * v;

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

            // test Translation3D

            Translation3D tr1(v);
            Translation3D tr2(v.X(), v.Y(), v.Z());
// skip this test on 32 bits architecture. It might fail due to extended precision
#if !defined(__i386__)
            iret[0] |= compare(tr1 == tr2, 1, 1);
#else
            // add a dummy test to have the same outputfile for roottest
            // otherwise it will complain that the output is different !
            iret[0] |= compare(0, 0, 1);
#endif

            Translation3D tr3 = tr1 * tr1.Inverse();
            GlobalPolar3DVector vp2 = tr3 * v;
            iret[0] |= compare(vp2.X(), v.X(), 10);
            iret[0] |= compare(vp2.Y(), v.Y(), 10);
            iret[0] |= compare(vp2.Z(), v.Z(), 10);

            Transform3D t2b = tr1 * Rotation3D(r);
            // this above fails on Windows - use a comparison with tolerance
            // 12 is size of Transform3D internal vector
            iret[0] |= compare(IsEqual(t2, t2b, 12), true, 1);
            // iret[0] |= compare(t2 ==t2b, 1,"eq1 transf",1 );
            Transform3D t2c(r, tr1);
            iret[0] |= compare(IsEqual(t2, t2c, 12), true, 1);
            // iret[0] |= compare(t2 ==t2c, 1,"eq2 transf",1 );

            Transform3D t3 = Rotation3D(r) * Translation3D(vr);

            Rotation3D rrr;
            XYZVector vvv;
            t2b.GetDecomposition(rrr, vvv);
            iret[0] |= compare(Rotation3D(r) == rrr, 1, 1);
            iret[0] |= compare(tr1.Vect() == vvv, 1, 1);
            //   if (iret) std::cout << vvv << std::endl;
            //   if (iret) std::cout << Translation3D(vr) << std::endl;

            Translation3D ttt;
            t2b.GetDecomposition(rrr, ttt);
            iret[0] |= compare(tr1 == ttt, 1, 1);
            //   if (iret) std::cout << ttt << std::endl;

            EulerAngles err2;
            GlobalPolar3DVector vvv2;
            t2b.GetDecomposition(err2, vvv2);
            iret[0] |= compare(r.Phi(), err2.Phi(), 4);
            iret[0] |= compare(r.Theta(), err2.Theta(), 1);
            iret[0] |= compare(r.Psi(), err2.Psi(), 1);

            // iret[0] |= compare( v == vvv2, 1,"eq transf g vec",1 );
            iret[0] |= compare(v.X(), vvv2.X(), 4);
            iret[0] |= compare(v.Y(), vvv2.Y(), 1);
            iret[0] |= compare(v.Z(), vvv2.Z(), 1);

            // create from other rotations
            RotationZYX rzyx(r);
            Transform3D t4(rzyx);
            iret[0] |= compare(t4.Rotation() == Rotation3D(rzyx), 1, 1);

            Transform3D trf2 = tr1 * r;
            iret[0] |= compare(trf2 == t2b, 1, 1);
            Transform3D trf3 = r * Translation3D(vr);
            // iret[0] |= compare( trf3 == t3, 1,"e rot * transl",1 );
            //  this above fails on i686-slc5-gcc43-opt - use a comparison with tolerance
            iret[0] |= compare(IsEqual(trf3, t3, 12), true, 1);

            Transform3D t5(rzyx, v);
            Transform3D trf5 = Translation3D(v) * rzyx;
            // iret[0] |= compare( trf5 == t5, 1,"trasl * rzyx",1 );
            iret[0] |= compare(IsEqual(trf5, t5, 12), true, 1);

            Transform3D t6(rzyx, rzyx * Translation3D(v).Vect());
            Transform3D trf6 = rzyx * Translation3D(v);
            iret[0] |= compare(trf6 == t6, 1, 1);
            // if (iret)
            //    std::cout << t6 << "\n---\n" << trf6 << std::endl;

            Transform3D trf7 = t4 * Translation3D(v);
            // iret[0] |= compare( trf7 == trf6, 1,"tranf * transl",1 );
            iret[0] |= compare(IsEqual(trf7, trf6, 12), true, 1);
            Transform3D trf8 = Translation3D(v) * t4;
            iret[0] |= compare(trf8 == trf5, 1, 1);

            Transform3D trf9 = Transform3D(v) * rzyx;
            iret[0] |= compare(trf9 == trf5, 1, 1);
            Transform3D trf10 = rzyx * Transform3D(v);
            iret[0] |= compare(trf10 == trf6, 1, 1);
            Transform3D trf11 = Rotation3D(rzyx) * Transform3D(v);
            iret[0] |= compare(trf11 == trf10, 1, 1);

            RotationZYX rrr2 = trf10.Rotation<RotationZYX>();
            // iret[0] |= compare( rzyx == rrr2, 1,"gen Rotation()",1 );
            iret[0] |= compare(rzyx.Phi(), rrr2.Phi(), 1);
            iret[0] |= compare(rzyx.Theta(), rrr2.Theta(), 10);
            iret[0] |= compare(rzyx.Psi(), rrr2.Psi(), 1);
            // if (iret)
            //    std::cout << rzyx << "\n---\n" << rrr2 << std::endl;

            // std::cout << t2 << std::endl;
            // std::cout << t3 << std::endl;

            XYZPoint p1(-1., 2., -3);

            XYZPoint p2 = t2(p1);
            Polar3DPoint p3 = t3(Polar3DPoint(p1));
            iret[0] |= compare(p3.X(), p2.X(), 10);
            iret[0] |= compare(p3.Y(), p2.Y(), 10);
            iret[0] |= compare(p3.Z(), p2.Z(), 10);

            GlobalXYZVector v1(1., 2., 3.);
            LocalXYZVector v2;
            t2.Transform(v1, v2);
            GlobalPolar3DVector v3;
            t3.Transform(GlobalPolar3DVector(v1), v3);

            iret[0] |= compare(v3.X(), v2.X(), 10);
            iret[0] |= compare(v3.Y(), v2.Y(), 10);
            iret[0] |= compare(v3.Z(), v2.Z(), 10);

            XYZPoint q1(1, 2, 3);
            XYZPoint q2(-1, -2, -3);
            XYZPoint q3 = q1 + XYZVector(q2);
            // std::cout << q3 << std::endl;
            XYZPoint qt3 = t3(q3);
            // std::cout << qt3 << std::endl;
            XYZPoint qt1 = t3(q1);
            XYZVector vt2 = t3(XYZVector(q2));
            XYZPoint qt4 = qt1 + vt2;
            iret[0] |= compare(qt3.X(), qt4.X(), 10);
            iret[0] |= compare(qt3.Y(), qt4.Y(), 10);
            iret[0] |= compare(qt3.Z(), qt4.Z(), 10);
            // std::cout << qt4 << std::endl;

            // this fails
            //  double a = 3;
            // XYZPoint q4 = a*q1;
            //   std::cout << t3( a * q1) << std::endl;
            //   std::cout << a * t3(q1) << std::endl;

            {
               // testing ApplyInverse on Point
               XYZPoint point(1., -2., 3.);
               Transform3D tr(EulerAngles(10, -10, 10), XYZVector(10, -10, 0));

               // test that applying transformation + Inverse is identity
               auto r0 = tr.ApplyInverse(tr(point));
               auto r0_2 = tr.Inverse()(tr(point));

               iret[0] |= compare(r0.X(), point.X(), 100);
               iret[0] |= compare(r0_2.X(), point.X(), 100);
               iret[0] |= compare(r0.Y(), point.Y(), 10);
               iret[0] |= compare(r0_2.Y(), point.Y(), 10);
               iret[0] |= compare(r0.Z(), point.Z(), 10);
               iret[0] |= compare(r0_2.Z(), point.Z(), 10);

               // compare ApplyInverse with Inverse()
               auto r1 = tr.ApplyInverse(point);
               auto r2 = tr.Inverse()(point);
               iret[0] |= compare(r1.X(), r2.X(), 10);
               iret[0] |= compare(r1.Y(), r2.Y(), 10);
               iret[0] |= compare(r1.Z(), r2.Z(), 10);
            }

            {
               // testing ApplyInverse on Vector
               XYZVector vector(1, -2., 3);
               Transform3D tr(EulerAngles(10, -10, 10), XYZVector(10, -10, 0));
               auto r1 = tr.ApplyInverse(vector);
               auto r2 = tr.Inverse()(vector);
               iret[0] |= compare(r1.X(), r2.X(), 10);
               iret[0] |= compare(r1.Y(), r2.Y(), 10);
               iret[0] |= compare(r1.Z(), r2.Z(), 10);
            }
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\tOK\n";
   else
      std::cout << "\t FAILED\n";

   return iret_host;
}

int testVectorUtil()
{
   int iret_host = 0;
   std::cout << "testing VectorUtil  \t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testRotations3D>([=]() {
            // test new perp functions
            XYZVector v(1., 2., 3.);

            XYZVector vx = ProjVector(v, XYZVector(3, 0, 0));
            iret[0] |= compare(vx.X(), v.X(), 1);
            iret[0] |= compare(vx.Y(), 0, 1);
            iret[0] |= compare(vx.Z(), 0, 1);

            XYZVector vpx = PerpVector(v, XYZVector(2, 0, 0));
            iret[0] |= compare(vpx.X(), 0, 1);
            iret[0] |= compare(vpx.Y(), v.Y(), 1);
            iret[0] |= compare(vpx.Z(), v.Z(), 1);

            double perpy = Perp(v, XYZVector(0, 2, 0));
            iret[0] |= compare(perpy, std::sqrt(v.Mag2() - v.y() * v.y()));

            XYZPoint u(1, 1, 1);
            XYZPoint un = u / u.R();

            XYZVector vl = ProjVector(v, u);
            XYZVector vl2 = XYZVector(un) * (v.Dot(un));

            iret[0] |= compare(vl.X(), vl2.X(), 1);
            iret[0] |= compare(vl.Y(), vl2.Y(), 1);
            iret[0] |= compare(vl.Z(), vl2.Z(), 1);

            XYZVector vp = PerpVector(v, u);
            XYZVector vp2 = v - XYZVector(un * (v.Dot(un)));
            iret[0] |= compare(vp.X(), vp2.X(), 10);
            iret[0] |= compare(vp.Y(), vp2.Y(), 10);
            iret[0] |= compare(vp.Z(), vp2.Z(), 10);

            double perp = Perp(v, u);
            iret[0] |= compare(perp, vp.R(), 1);
            double perp2 = Perp2(v, u);
            iret[0] |= compare(perp2, vp.Mag2(), 1);

            // test rotations
            double angle = 1;
            XYZVector vr1 = RotateX(v, angle);
            XYZVector vr2 = RotationX(angle) * v;
            iret[0] |= compare(vr1.Y(), vr2.Y(), 1);
            iret[0] |= compare(vr1.Z(), vr2.Z(), 1);

            vr1 = RotateY(v, angle);
            vr2 = RotationY(angle) * v;
            iret[0] |= compare(vr1.X(), vr2.X(), 1);
            iret[0] |= compare(vr1.Z(), vr2.Z(), 1);

            vr1 = RotateZ(v, angle);
            vr2 = RotationZ(angle) * v;
            iret[0] |= compare(vr1.X(), vr2.X(), 1);
            iret[0] |= compare(vr1.Y(), vr2.Y(), 1);
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\tOK\n";
   else
      std::cout << "\t FAILED\n";

   return iret_host;
}

int testLorentzVector()
{
   int iret_host = 0;
   std::cout << "testing LorentzVector  \t:\n";

   sycl::buffer<int, 1> iret_buf(&iret_host, sycl::range<1>(1));
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {
      queue.submit([&](sycl::handler &cgh) {
         auto iret = iret_buf.get_access<sycl::access::mode::read_write>(cgh);
         cgh.single_task<class testRotations3D>([=]() {
            LorentzVector<PtEtaPhiM4D<float>> v1(1, 2, 3, 4);
            LorentzVector<PtEtaPhiM4D<float>> v2(5, 6, 7, 8);
            iret[0] |= compare(v1.DeltaR(v2), 4.60575f);

            LorentzVector<PtEtaPhiM4D<float>> v = v1 + v2;
            iret[0] |= compare(v.M(), 62.03058f);
         });
      });
   }

   if (iret_host == 0)
      std::cout << "\tOK\n";
   else
      std::cout << "\t FAILED\n";

   return iret_host;
}

int testGenVector()
{
   // SYCL syclcontext();

   int iret = 0;
   iret |= testVector3D();
   iret |= testPoint3D();

   iret |= testVector2D();
   iret |= testPoint2D();

   // iret |= testRotations3D();

   // iret |= testTransform3D();

   // iret |= testVectorUtil();
   iret |= testLorentzVector();

   if (iret != 0)
      std::cout << "\nTest GenVector FAILED!!!!!!!!!\n";
   return iret;
}

int main()
{

   int ret = testGenVector();
   if (ret)
      std::cerr << "test FAILED !!! " << std::endl;
   else
      std::cout << "test OK " << std::endl;
   return ret;
}
