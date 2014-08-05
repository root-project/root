// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

// rotation functions

// rotation 3D
#pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (double *, double*);
#pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (ROOT::Math::XYZVector &, ROOT::Math::XYZVector &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (ROOT::Math::XYZVector &, ROOT::Math::XYZVector &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Rotation3D::GetComponents (double *);
#pragma link C++ function  ROOT::Math::Rotation3D::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Rotation3D::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Rotation3D::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::operator* (const ROOT::Math::XYZTVector &);
// axis angle
// this explicit template ctor are not found by CINT
// #pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (const ROOT::Math::Rotation3D &);
// #pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (const ROOT::Math::EulerAngles &);
// #pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (const ROOT::Math::RotationZYX &);
// #pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (const ROOT::Math::RotationX &);
// #pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (const ROOT::Math::RotationY &);
// #pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (const ROOT::Math::RotationZ &);
// #pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (const ROOT::Math::Quaternion &);

#pragma link C++ function  ROOT::Math::AxisAngle::operator= (const ROOT::Math::Rotation3D &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator= (const ROOT::Math::EulerAngles &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator= (const ROOT::Math::RotationZYX &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator= (const ROOT::Math::RotationX &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator= (const ROOT::Math::RotationY &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator= (const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator= (const ROOT::Math::Quaternion &);
#pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (ROOT::Math::XYZVector &, double);
#pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (double *, double*);
#pragma link C++ function  ROOT::Math::AxisAngle::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::AxisAngle::GetComponents (double *);
#pragma link C++ function  ROOT::Math::AxisAngle::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator* (const ROOT::Math::XYZTVector &);
// Euler angles
// #pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (const ROOT::Math::Rotation3D &);
// #pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (const ROOT::Math::AxisAngle &);
// #pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (const ROOT::Math::RotationZYX &);
// #pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (const ROOT::Math::RotationX &);
// #pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (const ROOT::Math::RotationY &);
// #pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (const ROOT::Math::RotationZ &);
// #pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (const ROOT::Math::Quaternion &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator= (const ROOT::Math::Rotation3D &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator= (const ROOT::Math::AxisAngle &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator= (const ROOT::Math::Quaternion &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator= (const ROOT::Math::RotationZYX &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator= (const ROOT::Math::RotationX &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator= (const ROOT::Math::RotationY &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator= (const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::EulerAngles::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::EulerAngles::GetComponents (double *);
#pragma link C++ function  ROOT::Math::EulerAngles::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator* (const ROOT::Math::XYZTVector &);
// quaternion
// #pragma link C++ function  ROOT::Math::Quaternion::Quaternion (const ROOT::Math::Rotation3D &);
// #pragma link C++ function  ROOT::Math::Quaternion::Quaternion (const ROOT::Math::AxisAngle &);
// #pragma link C++ function  ROOT::Math::Quaternion::Quaternion (const ROOT::Math::EulerAngles &);
// #pragma link C++ function  ROOT::Math::Quaternion::Quaternion (const ROOT::Math::RotationZYX &);
// #pragma link C++ function  ROOT::Math::Quaternion::Quaternion (const ROOT::Math::RotationX &);
// #pragma link C++ function  ROOT::Math::Quaternion::Quaternion (const ROOT::Math::RotationY &);
// #pragma link C++ function  ROOT::Math::Quaternion::Quaternion (const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::Quaternion::operator= (const ROOT::Math::Rotation3D &);
#pragma link C++ function  ROOT::Math::Quaternion::operator= (const ROOT::Math::AxisAngle &);
#pragma link C++ function  ROOT::Math::Quaternion::operator= (const ROOT::Math::EulerAngles &);
#pragma link C++ function  ROOT::Math::Quaternion::operator= (const ROOT::Math::RotationZYX &);
#pragma link C++ function  ROOT::Math::Quaternion::operator= (const ROOT::Math::RotationX &);
#pragma link C++ function  ROOT::Math::Quaternion::operator= (const ROOT::Math::RotationY &);
#pragma link C++ function  ROOT::Math::Quaternion::operator= (const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::Quaternion::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Quaternion::GetComponents (double *);
#pragma link C++ function  ROOT::Math::Quaternion::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Quaternion::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Quaternion::operator* (const ROOT::Math::XYZTVector &);
// rotation ZYX
// #pragma link C++ function  ROOT::Math::RotationZYX::RotationZYX (const ROOT::Math::Rotation3D &);
// #pragma link C++ function  ROOT::Math::RotationZYX::RotationZYX (const ROOT::Math::AxisAngle &);
// #pragma link C++ function  ROOT::Math::RotationZYX::RotationZYX (const ROOT::Math::EulerAngles &);
// #pragma link C++ function  ROOT::Math::RotationZYX::RotationZYX (const ROOT::Math::Quaternion &);
// #pragma link C++ function  ROOT::Math::RotationZYX::RotationZYX (const ROOT::Math::RotationX &);
// #pragma link C++ function  ROOT::Math::RotationZYX::RotationZYX (const ROOT::Math::RotationY &);
// #pragma link C++ function  ROOT::Math::RotationZYX::RotationZYX (const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator= (const ROOT::Math::Rotation3D &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator= (const ROOT::Math::AxisAngle &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator= (const ROOT::Math::EulerAngles &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator= (const ROOT::Math::Quaternion &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator= (const ROOT::Math::RotationX &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator= (const ROOT::Math::RotationY &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator= (const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::RotationZYX::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::RotationZYX::GetComponents (double *);
#pragma link C++ function  ROOT::Math::RotationZYX::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationZYX::operator* (const ROOT::Math::XYZTVector &);
// rotation X
#pragma link C++ function  ROOT::Math::RotationX::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationX::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationX::operator* (const ROOT::Math::XYZTVector &);
// rotation Y
#pragma link C++ function  ROOT::Math::RotationY::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationY::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationY::operator* (const ROOT::Math::XYZTVector &);
// rotation Z
#pragma link C++ function  ROOT::Math::RotationZ::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationZ::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationZ::operator* (const ROOT::Math::XYZTVector &);

// transform3D
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::Rotation3D &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::AxisAngle &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::EulerAngles &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::RotationZYX &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::Quaternion &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::RotationX &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::RotationY &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (const ROOT::Math::RotationZ &,const ROOT::Math::Translation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (double *, double*);

#pragma link C++ function  ROOT::Math::Transform3D::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Transform3D::GetComponents (double *);
//#pragma link C++ function  ROOT::Math::Transform3D::Rotation< ROOT::Math::RotationZYX>();
#pragma link C++ function  ROOT::Math::Transform3D::GetRotation ( ROOT::Math::Rotation3D &);
#pragma link C++ function  ROOT::Math::Transform3D::GetRotation ( ROOT::Math::RotationZYX &);
#pragma link C++ function  ROOT::Math::Transform3D::GetRotation ( ROOT::Math::AxisAngle &);
#pragma link C++ function  ROOT::Math::Transform3D::GetRotation ( ROOT::Math::EulerAngles &);
#pragma link C++ function  ROOT::Math::Transform3D::GetRotation ( ROOT::Math::Quaternion &);
#pragma link C++ function  ROOT::Math::Transform3D::GetDecomposition ( ROOT::Math::RotationZYX &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Transform3D::GetDecomposition ( ROOT::Math::AxisAngle &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Transform3D::GetDecomposition ( ROOT::Math::EulerAngles &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Transform3D::GetDecomposition ( ROOT::Math::Quaternion &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Transform3D::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Transform3D::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Transform3D::operator* (const ROOT::Math::XYZTVector &);

// LorentzRotation
#pragma link C++ function  ROOT::Math::LorentzRotation::LorentzRotation (double *, double*);
#pragma link C++ function  ROOT::Math::LorentzRotation::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::LorentzRotation::GetComponents (double *);
#pragma link C++ function  ROOT::Math::LorentzRotation::operator* (const ROOT::Math::XYZTVector &);

// Boost
//#pragma link C++ function  ROOT::Math::Boost::Boost (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Boost::Boost (double *, double*);
#pragma link C++ function  ROOT::Math::Boost::SetComponents (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Boost::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Boost::GetComponents (double *);
#pragma link C++ function  ROOT::Math::Boost::operator* (const ROOT::Math::XYZTVector &);

// Boost X
#pragma link C++ function  ROOT::Math::BoostX::operator* (const ROOT::Math::XYZTVector &);
// Boost Y
#pragma link C++ function  ROOT::Math::BoostY::operator* (const ROOT::Math::XYZTVector &);
// Boost Z
#pragma link C++ function  ROOT::Math::BoostZ::operator* (const ROOT::Math::XYZTVector &);



//Rotation3D free functions
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationX &, const ROOT::Math::Rotation3D &);
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationY &, const ROOT::Math::Rotation3D &);
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationZ &, const ROOT::Math::Rotation3D &);

#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationX &, const ROOT::Math::RotationY &);
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationX &, const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationY &, const ROOT::Math::RotationX &);
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationY &, const ROOT::Math::RotationZ &);
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationZ &, const ROOT::Math::RotationX &);
#pragma link C++ function  ROOT::Math::operator* (const ROOT::Math::RotationZ &, const ROOT::Math::RotationY &);
