#include <type_traits>

// Test that all vectors and coordinate systems are trivially copyable and move constructible
template <class T>
struct TypeTests {
   static_assert(std::is_nothrow_move_constructible_v<T>);
   static_assert(std::is_trivially_move_constructible_v<T>);
   static_assert(std::is_nothrow_move_assignable_v<T>);
   static_assert(std::is_trivially_move_assignable_v<T>);
   static_assert(std::is_trivially_copyable_v<T>);
   static_assert(std::is_trivially_destructible_v<T>);
};

#include "Math/Point2D.h"
template struct TypeTests<ROOT::Math::XYPoint>;
template struct TypeTests<ROOT::Math::Polar2DPoint>;

#include "Math/Point3D.h"
template struct TypeTests<ROOT::Math::XYZPoint>;
template struct TypeTests<ROOT::Math::Polar3DPoint>;
template struct TypeTests<ROOT::Math::RhoZPhiPoint>;
template struct TypeTests<ROOT::Math::RhoEtaPhiPoint>;

#include "Math/Vector2D.h"
template struct TypeTests<ROOT::Math::XYVector>;
template struct TypeTests<ROOT::Math::Polar2DVector>;

#include "Math/Vector3D.h"
template struct TypeTests<ROOT::Math::XYZVector>;
template struct TypeTests<ROOT::Math::Polar3DVector>;
template struct TypeTests<ROOT::Math::RhoZPhiVector>;
template struct TypeTests<ROOT::Math::RhoEtaPhiVector>;

#include "Math/Vector4D.h"
template struct TypeTests<ROOT::Math::PxPyPzEVector>;
template struct TypeTests<ROOT::Math::PxPyPzMVector>;
template struct TypeTests<ROOT::Math::PtEtaPhiEVector>;
template struct TypeTests<ROOT::Math::PtEtaPhiMVector>;