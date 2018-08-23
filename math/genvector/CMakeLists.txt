############################################################################
# CMakeLists.txt file for building ROOT math/genvector package
############################################################################

set(headers32
  Math/Point2D.h
  Math/Point3D.h
  Math/Vector2D.h
  Math/Vector3D.h
  Math/Vector4D.h
)

set(headers
  Math/AxisAngle.h
  Math/Boost.h
  Math/BoostX.h
  Math/BoostY.h
  Math/BoostZ.h
  Math/EulerAngles.h
  Math/LorentzRotation.h
  Math/Plane3D.h
  Math/Point2D.h
  Math/Point3D.h
  Math/Quaternion.h
  Math/Rotation3D.h
  Math/RotationX.h
  Math/RotationY.h
  Math/RotationZ.h
  Math/RotationZYX.h
  Math/Transform3D.h
  Math/Translation3D.h
  Math/Vector2D.h
  Math/Vector3D.h
  Math/Vector4D.h
  Math/VectorUtil.h
)

set(sources
  3DConversions.cxx
  3DDistances.cxx
  AxisAngle.cxx
  AxisAngleXother.cxx
  BitReproducible.cxx
  Boost.cxx
  BoostX.cxx
  BoostY.cxx
  BoostZ.cxx
  EulerAngles.cxx
  GenVector_exception.cxx
  LorentzRotation.cxx
  Quaternion.cxx
  QuaternionXaxial.cxx
  Rotation3D.cxx
  Rotation3DxAxial.cxx
  RotationZYX.cxx
  VectorUtil.cxx
)

ROOT_GENERATE_DICTIONARY(G__GenVector ${headers}
  MODULE GenVector
  LINKDEF Math/LinkDef_GenVector.h
  OPTIONS "-writeEmptyRootPCM"
  DEPENDENCIES Core MathCore)

ROOT_GENERATE_DICTIONARY(G__GenVector32 ${headers32}
  MODULE GenVector
  MULTIDICT
  LINKDEF Math/LinkDef_GenVector32.h
  OPTIONS "-writeEmptyRootPCM"
  DEPENDENCIES Core MathCore)

ROOT_LINKER_LIBRARY(GenVector ${sources} G__GenVector.cxx G__GenVector32.cxx
  LIBRARIES Core MathCore)

ROOT_INSTALL_HEADERS()
