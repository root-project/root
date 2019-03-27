// ROOT
#include "Math/GenVector/PositionVector3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Plane3D.h"
#include "Math/GenVector/Transform3D.h"
#include "TStopwatch.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <Vc/Vc>
#pragma GCC diagnostic pop

// STL
#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <typeinfo>
#include <cmath>
#include <type_traits>

template<typename T>
T relativeError(const T &x, const T &y)
{
   if (x == y)
      return 0;

   T diff = std::abs(x - y);

   if (x * y == T(0) || diff < std::numeric_limits<T>::epsilon())
      return diff;

   return diff / (std::abs(x) + std::abs(y));
}

int compare(double x, double y, double tolerance = 1.0e-12)
{
   double error = relativeError(x, y);

   if (error > tolerance) {
      int pr = std::cerr.precision(16);
      std::cerr << "Error above tolerance:"         << std::endl
                << "  expected = " << x             << std::endl
                << "true value = " << y             << std::endl
                << "abs. error = " << std::abs(x-y) << std::endl
                << "rel. error = " << error         << std::endl
                << "tolerance  = " << tolerance     << std::endl;
      std::cerr.precision(pr);
      return 1;
   }

   return 0;
}

// randomn generator
static std::default_random_engine gen;
// Distributions for each member
static std::uniform_real_distribution<double> p_x(-800, 800), p_y(-600, 600), p_z(10000, 10500);
static std::uniform_real_distribution<double> d_x(-0.2, 0.2), d_y(-0.1, 0.1), d_z(0.95, 0.99);
static std::uniform_real_distribution<double> c_x(3100, 3200), c_y(10, 15), c_z(3200, 3300);
static std::uniform_real_distribution<double> r_rad(8500, 8600);
static std::uniform_real_distribution<double> p0(-0.002, 0.002), p1(-0.2, 0.2), p2(0.97, 0.99), p3(-1300, 1300);

template <typename POINT, typename VECTOR, typename PLANE, typename FTYPE>
class Data {
public:
   typedef std::vector<Data, Vc::Allocator<Data>> Vector;

public:
   POINT  position;
   VECTOR direction;
   POINT  CoC;
   PLANE  plane;
   FTYPE  radius{0};

public:
   template <typename INDATA>
   Data(const INDATA &ind)
      : position(ind.position.x(), ind.position.y(), ind.position.z()),
        direction(ind.direction.x(), ind.direction.y(), ind.direction.z()), CoC(ind.CoC.x(), ind.CoC.y(), ind.CoC.z()),
        plane(ind.plane.A(), ind.plane.B(), ind.plane.C(), ind.plane.D()), radius(ind.radius)
   {
   }
   Data()
      : position(p_x(gen), p_y(gen), p_z(gen)), direction(d_x(gen), d_y(gen), d_z(gen)),
        CoC(c_x(gen), c_y(gen), c_z(gen)), plane(p0(gen), p1(gen), p2(gen), p3(gen)), radius(r_rad(gen))
   {
   }
};

template <typename INDATA, typename OUTDATA>
void fill(const INDATA &in, OUTDATA &out)
{
   out.clear();
   out.reserve(in.size());
   for (const auto &i : in) {
      out.emplace_back(i);
   }
}

template <typename POINT, typename VECTOR, typename FTYPE,
          typename = typename std::enable_if<std::is_arithmetic<typename POINT::Scalar>::value &&
                                             std::is_arithmetic<typename VECTOR::Scalar>::value &&
                                             std::is_arithmetic<FTYPE>::value>::type>
inline bool reflectSpherical(POINT &position, VECTOR &direction, const POINT &CoC, const FTYPE radius)
{
   constexpr FTYPE zero(0), two(2.0), four(4.0), half(0.5);
   const FTYPE     a     = direction.Mag2();
   const VECTOR    delta = position - CoC;
   const FTYPE     b     = two * direction.Dot(delta);
   const FTYPE     c     = delta.Mag2() - radius * radius;
   const FTYPE     discr = b * b - four * a * c;
   const bool      OK    = discr > zero;
   if (OK) {
      const FTYPE dist = half * (std::sqrt(discr) - b) / a;
      // change position to the intersection point
      position += dist * direction;
      // reflect the vector
      // r = u - 2(u.n)n, r=reflection, u=incident, n=normal
      const VECTOR normal = position - CoC;
      direction -= (two * normal.Dot(direction) / normal.Mag2()) * normal;
   }
   return OK;
}

template <typename POINT, typename VECTOR, typename FTYPE,
          typename = typename std::enable_if<!std::is_arithmetic<typename POINT::Scalar>::value &&
                                             !std::is_arithmetic<typename VECTOR::Scalar>::value &&
                                             !std::is_arithmetic<FTYPE>::value>::type>
inline typename FTYPE::mask_type reflectSpherical(POINT &position, VECTOR &direction, const POINT &CoC,
                                                  const FTYPE radius)
{
   const FTYPE               two(2.0), four(4.0), half(0.5);
   const FTYPE               a     = direction.Mag2();
   const VECTOR              delta = position - CoC;
   const FTYPE               b     = two * direction.Dot(delta);
   const FTYPE               c     = delta.Mag2() - radius * radius;
   FTYPE                     discr = b * b - four * a * c;
   typename FTYPE::mask_type OK    = discr > FTYPE::Zero();
   if (any_of(OK)) {
      // Zero out the negative values in discr, to prevent sqrt(-ve)
      discr(!OK) = FTYPE::Zero();
      // compute the distance
      const FTYPE dist = half * (sqrt(discr) - b) / a;
      // change position to the intersection point
      position += dist * direction;
      // reflect the vector
      // r = u - 2(u.n)n, r=reflection, u=incident, n=normal
      const VECTOR normal = position - CoC;
      direction -= (two * normal.Dot(direction) / normal.Mag2()) * normal;
   }
   // return the mask indicating which results should be used
   return OK;
}

template <typename POINT, typename VECTOR, typename PLANE,
          typename = typename std::enable_if<std::is_arithmetic<typename POINT::Scalar>::value &&
                                             std::is_arithmetic<typename VECTOR::Scalar>::value>::type>
inline bool reflectPlane(POINT &position, VECTOR &direction, const PLANE &plane)
{
   constexpr typename POINT::Scalar two(2.0);
   const bool                       OK = true;
   // Plane normal
   const auto &normal = plane.Normal();
   // compute distance to the plane
   const auto scalar   = direction.Dot(normal);
   const auto distance = -(plane.Distance(position)) / scalar;
   // change position to reflection point and update direction
   position += distance * direction;
   direction -= two * scalar * normal;
   return OK;
}

template <typename POINT, typename VECTOR, typename PLANE, typename FTYPE = typename POINT::Scalar,
          typename = typename std::enable_if<!std::is_arithmetic<typename POINT::Scalar>::value &&
                                             !std::is_arithmetic<typename VECTOR::Scalar>::value>::type>
inline typename FTYPE::mask_type reflectPlane(POINT &position, VECTOR &direction, const PLANE &plane)
{
   const typename POINT::Scalar    two(2.0);
   const typename FTYPE::mask_type OK(true);
   // Plane normal
   const VECTOR normal = plane.Normal();
   // compute distance to the plane
   const FTYPE scalar   = direction.Dot(normal);
   const FTYPE distance = -(plane.Distance(position)) / scalar;
   // change position to reflection point and update direction
   position += distance * direction;
   direction -= two * scalar * normal;
   return OK;
}

template <typename T>
using PositionVector = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Plane = ROOT::Math::Impl::Plane3D<T>;

int main(int /*argc*/, char ** /*argv*/)
{
   int ret = 0;

   {

      const unsigned int nPhotons = 100;
      std::cout << "Creating " << nPhotons << " random photons ..." << std::endl;

      // Scalar Types
      Data<PositionVector<double>, Vector<double>, Plane<double>, double>::Vector scalar_data(nPhotons);

      // Vc Types
      Data<PositionVector<Vc::double_v>, Vector<Vc::double_v>, Plane<Vc::double_v>, Vc::double_v>::Vector vc_data;
      // Clone the exact random values from the Scalar vector
      // Note we are making the same number of entries in the container, but each entry is a vector entry
      // with Vc::double_t::Size entries.
      fill(scalar_data, vc_data);

      // Loop over the two containers and compare
      std::cout << "Ray Tracing :-" << std::endl;

      for (size_t i = 0; i < nPhotons; ++i) {
         auto &sc = scalar_data[i];
         auto &vc = vc_data[i];

         // ray tracing
         reflectSpherical(sc.position, sc.direction, sc.CoC, sc.radius);
         reflectPlane(sc.position, sc.direction, sc.plane);
         reflectSpherical(vc.position, vc.direction, vc.CoC, vc.radius);
         reflectPlane(vc.position, vc.direction, vc.plane);

         std::cout << "Position  " << sc.position << " " << vc.position << std::endl;
         std::cout << "Direction " << sc.direction << " " << vc.direction << std::endl;

         for (std::size_t j = 0; j < Vc::double_v::Size; ++j) {
            ret |= compare(sc.position.x(), vc.position.x()[j]);
            ret |= compare(sc.position.y(), vc.position.y()[j]);
            ret |= compare(sc.position.z(), vc.position.z()[j]);
            ret |= compare(sc.direction.x(), vc.direction.x()[j]);
            ret |= compare(sc.direction.y(), vc.direction.y()[j]);
            ret |= compare(sc.direction.z(), vc.direction.z()[j]);
         }
      }

      // Now test Transformation3D
      std::cout << "Transforms :-" << std::endl;
      for (size_t i = 0; i < nPhotons; ++i) {
         auto &sc = scalar_data[i];
         auto &vc = vc_data[i];

         // make 6 random scalar PositionVectors
         PositionVector<double> sp1(p_x(gen), p_y(gen), p_z(gen));
         PositionVector<double> sp2(p_x(gen), p_y(gen), p_z(gen));
         PositionVector<double> sp3(p_x(gen), p_y(gen), p_z(gen));
         PositionVector<double> sp4(p_x(gen), p_y(gen), p_z(gen));
         PositionVector<double> sp5(p_x(gen), p_y(gen), p_z(gen));
         PositionVector<double> sp6(p_x(gen), p_y(gen), p_z(gen));
         // clone to Vc versions
         PositionVector<Vc::double_v> vp1(sp1.x(), sp1.y(), sp1.z());
         PositionVector<Vc::double_v> vp2(sp2.x(), sp2.y(), sp2.z());
         PositionVector<Vc::double_v> vp3(sp3.x(), sp3.y(), sp3.z());
         PositionVector<Vc::double_v> vp4(sp4.x(), sp4.y(), sp4.z());
         PositionVector<Vc::double_v> vp5(sp5.x(), sp5.y(), sp5.z());
         PositionVector<Vc::double_v> vp6(sp6.x(), sp6.y(), sp6.z());

         // Make transformations from points
         // note warnings about axis not having the same angles expected here...
         // point is to check scalar and vector versions do the same thing
         const ROOT::Math::Impl::Transform3D<double> st(sp1, sp2, sp3, sp4, sp5, sp6);
         const ROOT::Math::Impl::Transform3D<Vc::double_v> vt(vp1, vp2, vp3, vp4, vp5, vp6);

         // transform the vectors
         const auto sv = st * sc.direction;
         const auto vv = vt * vc.direction;
         std::cout << "Transformed Direction " << sv << " " << vv << std::endl;

         // invert the transformations
         const auto st_i = st.Inverse();
         const auto vt_i = vt.Inverse();

         // Move the points back
         const auto sv_i = st_i * sv;
         const auto vv_i = vt_i * vv;
         std::cout << "Transformed Back Direction " << sc.direction << " " << sv_i << " " << vv_i << std::endl;

         for (std::size_t j = 0; j < Vc::double_v::Size; ++j) {
            ret |= compare(sv.x(), vv.x()[j]);
            ret |= compare(sv.y(), vv.y()[j]);
            ret |= compare(sv.z(), vv.z()[j]);
            ret |= compare(sc.direction.x(), vv_i.x()[j]);
            ret |= compare(sc.direction.y(), vv_i.y()[j]);
            ret |= compare(sc.direction.z(), vv_i.z()[j]);
         }

         ret |= compare(sc.direction.x(), sv_i.x());
         ret |= compare(sc.direction.y(), sv_i.y());
         ret |= compare(sc.direction.z(), sv_i.z());

         // Make a scalar Plane
         const double a(p0(gen)), b(p1(gen)), c(p2(gen)), d(p3(gen));
         Plane<double> sc_plane(a, b, c, d);
         // make a vector plane
         Plane<Vc::double_v> vc_plane(a, b, c, d);

         // transform the planes
         const auto new_sc_plane = st * sc_plane;
         const auto new_vc_plane = vt * vc_plane;
         std::cout << "Transformed plane " << new_sc_plane << " " << new_vc_plane << std::endl;

         // now transform the planes back
         const auto sc_plane_i = st_i * new_sc_plane;
         const auto vc_plane_i = vt_i * new_vc_plane;
         std::cout << "Transformed Back plane " << sc_plane_i << " " << vc_plane_i << std::endl;

         for (std::size_t j = 0; j < Vc::double_v::Size; ++j) {
            ret |= compare(vc_plane.A()[j], vc_plane_i.A()[j]);
            ret |= compare(vc_plane.B()[j], vc_plane_i.B()[j]);
            ret |= compare(vc_plane.C()[j], vc_plane_i.C()[j]);
            ret |= compare(vc_plane.D()[j], vc_plane_i.D()[j]);
            ret |= compare(sc_plane_i.A(), vc_plane_i.A()[j]);
            ret |= compare(sc_plane_i.B(), vc_plane_i.B()[j]);
            ret |= compare(sc_plane_i.C(), vc_plane_i.C()[j]);
            ret |= compare(sc_plane_i.D(), vc_plane_i.D()[j]);
         }
      }
   }

   // now run some timing tests
   {
      const unsigned int nPhotons = 96000; // Must be multiple of 16 to avoid padding issues below...

      const unsigned int nTests = 1000; // number of tests to run

      // scalar data
      Data<PositionVector<double>, Vector<double>, Plane<double>, double>::Vector scalar_data(nPhotons);
      // vector data with total equal number of photons (including vectorised size)
      Data<PositionVector<Vc::double_v>, Vector<Vc::double_v>, Plane<Vc::double_v>, Vc::double_v>::Vector vc_data(
         nPhotons / Vc::double_v::Size);

      TStopwatch t;

      double best_time_scalar{9e30}, best_time_vector{9e30};

      // time the scalar implementation
      for (unsigned int i = 0; i < nTests; ++i) {
         t.Start();
         for (auto &sc : scalar_data) {
            reflectSpherical(sc.position, sc.direction, sc.CoC, sc.radius);
            reflectPlane(sc.position, sc.direction, sc.plane);
         }
         t.Stop();
         const auto time = t.RealTime();
         if (time < best_time_scalar) {
            best_time_scalar = time;
         }
      }

      // time the Vc implementation
      for (unsigned int i = 0; i < nTests; ++i) {
         t.Start();
         for (auto &vc : vc_data) {
            reflectSpherical(vc.position, vc.direction, vc.CoC, vc.radius);
            reflectPlane(vc.position, vc.direction, vc.plane);
         }
         t.Stop();
         const auto time = t.RealTime();
         if (time < best_time_vector) {
            best_time_vector = time;
         }
      }

      std::cout << "Scalar best time        = " << best_time_scalar << std::endl;
      std::cout << "Vectorised Vc best time = " << best_time_vector << std::endl;
      std::cout << "Vectorised Vc SIMD size = " << Vc::double_v::Size << std::endl;
      std::cout << "Vectorised Vc speedup   = " << best_time_scalar / best_time_vector << std::endl;

      // assert that the vector time is roughly Vc::double_v::Size times smaller than the scalar time
      // allow 25% for 'safety'
      // if (std::fabs((best_time_vector * Vc::double_v::Size) - best_time_scalar) > 0.25 * best_time_scalar) {
      //   ++ret;
      // }
   }

   if (ret)
      std::cerr << "test FAILED !!! " << std::endl;
   else
      std::cout << "test OK " << std::endl;
   return ret;
}
