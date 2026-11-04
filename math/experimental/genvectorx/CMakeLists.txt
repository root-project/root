# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/genvector package
############################################################################

message(STATUS "Experimental GenVectorX")

  ROOT_LINKER_LIBRARY(GenVectorSYCL
      src/3DConversions.cxx
      src/3DDistances.cxx
      src/AxisAngle.cxx
      src/AxisAngleXother.cxx
      src/BitReproducible.cxx
      src/Boost.cxx
      src/BoostX.cxx
      src/BoostY.cxx
      src/BoostZ.cxx
      src/EulerAngles.cxx
      src/LorentzRotation.cxx
      src/Quaternion.cxx
      src/QuaternionXaxial.cxx
      src/Rotation3D.cxx
      src/Rotation3DxAxial.cxx
      src/RotationZYX.cxx
      src/VectorUtil.cxx
    DEPENDENCIES
      Core
      MathCore
  )
 
  add_sycl_to_root_target(
    TARGET 
      GenVectorSYCL 
    SOURCES 
      src/3DConversions.cxx
      src/3DDistances.cxx
      src/AxisAngle.cxx
      src/AxisAngleXother.cxx
      src/BitReproducible.cxx
      src/Boost.cxx
      src/BoostX.cxx
      src/BoostY.cxx
      src/BoostZ.cxx
      src/EulerAngles.cxx
      src/LorentzRotation.cxx
      src/Quaternion.cxx
      src/QuaternionXaxial.cxx
      src/Rotation3D.cxx
      src/Rotation3DxAxial.cxx
      src/RotationZYX.cxx
      src/VectorUtil.cxx
     COMPILE_DEFINITIONS
       ROOT_MATH_SYCL
  )
  
set(ACPP_EXPORTS acpp-rt acpp-common)

foreach(target IN LISTS ACPP_EXPORTS)
   install(TARGETS ${target} EXPORT ROOTExports)
endforeach()

target_link_libraries(GenVectorSYCL PUBLIC m stdc++)

target_compile_definitions(GenVectorSYCL PUBLIC ROOT_MATH_SYCL)

target_sources(GenVectorSYCL
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    MathX/AxisAngle.h
    MathX/Boost.h
    MathX/BoostX.h
    MathX/BoostY.h
    MathX/BoostZ.h
    MathX/Cartesian2D.h
    MathX/Cartesian3D.h
    MathX/Cylindrical3D.h
    MathX/CylindricalEta3D.h
    MathX/DisplacementVector2D.h
    MathX/DisplacementVector3D.h
    MathX/EulerAngles.h
    MathX/LinkDef_GenVector.h
    MathX/LinkDef_GenVector2.h
    MathX/LinkDef_GenVector32.h
    MathX/LinkDef_GenVectorAll.h
    MathX/LorentzRotation.h
    MathX/LorentzVector.h
    MathX/Plane3D.h
    MathX/Point2D.h
    MathX/Point2Dfwd.h
    MathX/Point3D.h
    MathX/Point3Dfwd.h
    MathX/Polar2D.h
    MathX/Polar3D.h
    MathX/PositionVector2D.h
    MathX/PositionVector3D.h
    MathX/PtEtaPhiE4D.h
    MathX/PtEtaPhiM4D.h
    MathX/PxPyPzE4D.h
    MathX/PxPyPzM4D.h
    MathX/Quaternion.h
    MathX/Rotation3D.h
    MathX/RotationX.h
    MathX/RotationY.h
    MathX/RotationZ.h
    MathX/RotationZYX.h
    MathX/Transform3D.h
    MathX/Translation3D.h
    MathX/Vector2D.h
    MathX/Vector2Dfwd.h
    MathX/Vector3D.h
    MathX/Vector3Dfwd.h
    MathX/Vector4D.h
    MathX/Vector4Dfwd.h
    MathX/VectorUtil.h
    MathX/GenVectorX/3DConversions.h
    MathX/GenVectorX/3DDistances.h
    MathX/GenVectorX/AccHeaders.h
    MathX/GenVectorX/AxisAngle.h
    MathX/GenVectorX/AxisAnglefwd.h
    MathX/GenVectorX/BitReproducible.h
    MathX/GenVectorX/Boost.h
    MathX/GenVectorX/BoostX.h
    MathX/GenVectorX/BoostXfwd.h
    MathX/GenVectorX/BoostY.h
    MathX/GenVectorX/BoostYfwd.h
    MathX/GenVectorX/BoostZ.h
    MathX/GenVectorX/BoostZfwd.h
    MathX/GenVectorX/Boostfwd.h
    MathX/GenVectorX/Cartesian2D.h
    MathX/GenVectorX/Cartesian2Dfwd.h
    MathX/GenVectorX/Cartesian3D.h
    MathX/GenVectorX/Cartesian3Dfwd.h
    MathX/GenVectorX/CoordinateSystemTags.h
    MathX/GenVectorX/Cylindrical3D.h
    MathX/GenVectorX/Cylindrical3Dfwd.h
    MathX/GenVectorX/CylindricalEta3D.h
    MathX/GenVectorX/CylindricalEta3Dfwd.h
    MathX/GenVectorX/DisplacementVector2D.h
    MathX/GenVectorX/DisplacementVector2Dfwd.h
    MathX/GenVectorX/DisplacementVector3D.h
    MathX/GenVectorX/DisplacementVector3Dfwd.h
    MathX/GenVectorX/EulerAngles.h
    MathX/GenVectorX/EulerAnglesfwd.h
    MathX/GenVectorX/GenVectorIO.h
    MathX/GenVectorX/GenVector_exception.h
    MathX/GenVectorX/LorentzRotation.h
    MathX/GenVectorX/LorentzRotationfwd.h
    MathX/GenVectorX/LorentzVector.h
    MathX/GenVectorX/LorentzVectorfwd.h
    MathX/GenVectorX/MathHeaders.h
    MathX/GenVectorX/Plane3D.h
    MathX/GenVectorX/Polar2D.h
    MathX/GenVectorX/Polar2Dfwd.h
    MathX/GenVectorX/Polar3D.h
    MathX/GenVectorX/Polar3Dfwd.h
    MathX/GenVectorX/PositionVector2D.h
    MathX/GenVectorX/PositionVector2Dfwd.h
    MathX/GenVectorX/PositionVector3D.h
    MathX/GenVectorX/PositionVector3Dfwd.h
    MathX/GenVectorX/PtEtaPhiE4D.h
    MathX/GenVectorX/PtEtaPhiE4Dfwd.h
    MathX/GenVectorX/PtEtaPhiM4D.h
    MathX/GenVectorX/PtEtaPhiM4Dfwd.h
    MathX/GenVectorX/PxPyPzE4D.h
    MathX/GenVectorX/PxPyPzE4Dfwd.h
    MathX/GenVectorX/PxPyPzM4D.h
    MathX/GenVectorX/PxPyPzM4Dfwd.h
    MathX/GenVectorX/Quaternion.h
    MathX/GenVectorX/Quaternionfwd.h
    MathX/GenVectorX/Rotation3D.h
    MathX/GenVectorX/Rotation3Dfwd.h
    MathX/GenVectorX/RotationX.h
    MathX/GenVectorX/RotationXfwd.h
    MathX/GenVectorX/RotationY.h
    MathX/GenVectorX/RotationYfwd.h
    MathX/GenVectorX/RotationZ.h
    MathX/GenVectorX/RotationZYX.h
    MathX/GenVectorX/RotationZYXfwd.h
    MathX/GenVectorX/RotationZfwd.h
    MathX/GenVectorX/Transform3D.h
    MathX/GenVectorX/Translation3D.h
    MathX/GenVectorX/VectorUtil.h
    MathX/GenVectorX/eta.h
    MathX/GenVectorX/etaMax.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)

ROOT_INSTALL_HEADERS()
