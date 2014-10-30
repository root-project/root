############################################################################
# CMakeLists.txt file for building ROOT math/genvector package
############################################################################

set(libname GenVector)
ROOT_USE_PACKAGE(math)

set(headers   Math/Vector2D.h Math/Point2D.h
              Math/Vector3D.h Math/Point3D.h
              Math/Vector4D.h Math/Rotation3D.h Math/RotationZYX.h
              Math/RotationX.h Math/RotationY.h Math/RotationZ.h
              Math/LorentzRotation.h
              Math/Boost.h Math/BoostX.h Math/BoostY.h Math/BoostZ.h
              Math/EulerAngles.h Math/AxisAngle.h Math/Quaternion.h
              Math/Transform3D.h Math/Translation3D.h Math/Plane3D.h
              Math/VectorUtil.h)
set(headers32 Math/Vector2D.h Math/Point2D.h
             Math/Vector3D.h Math/Point3D.h Math/Vector4D.h)

ROOT_GENERATE_DICTIONARY(G__${libname}   ${headers} MODULE ${libname} LINKDEF Math/LinkDef_GenVector.h OPTIONS "-writeEmptyRootPCM")
ROOT_GENERATE_DICTIONARY(G__${libname}32 ${headers32} MULTIDICT MODULE ${libname} LINKDEF Math/LinkDef_GenVector32.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx G__${libname}32.cxx LIBRARIES Core)
ROOT_INSTALL_HEADERS()

