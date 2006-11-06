// @(#)root/mathcore:$Name:  $:$Id: LinkDef_Rotation.h,v 1.5 2006/04/11 13:06:15 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

// rotation functions

// rot 3D  
#pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (double *, double*);
#pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (ROOT::Math::XYZVector &, ROOT::Math::XYZVector &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (ROOT::Math::XYZVector &, ROOT::Math::XYZVector &,ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Rotation3D::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Rotation3D::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::operator() (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Rotation3D::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Rotation3D::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Rotation3D::operator* (const ROOT::Math::XYZTVector &);
// axis angle
#pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (ROOT::Math::XYZVector &, double);
#pragma link C++ function  ROOT::Math::AxisAngle::AxisAngle (double *, double*);
#pragma link C++ function  ROOT::Math::AxisAngle::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::AxisAngle::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::AxisAngle::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator() (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::AxisAngle::operator* (const ROOT::Math::XYZTVector &);
// Euler angles 
#pragma link C++ function  ROOT::Math::EulerAngles::EulerAngles (double *, double*);
#pragma link C++ function  ROOT::Math::EulerAngles::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::EulerAngles::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::EulerAngles::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator() (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::EulerAngles::operator* (const ROOT::Math::XYZTVector &);
// quaternion 
#pragma link C++ function  ROOT::Math::Quaternion::Quaternion (double *, double*);
#pragma link C++ function  ROOT::Math::Quaternion::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Quaternion::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Quaternion::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::Quaternion::operator() (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Quaternion::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Quaternion::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Quaternion::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Quaternion::operator* (const ROOT::Math::XYZTVector &);
// rotation X
#pragma link C++ function  ROOT::Math::RotationX::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::RotationX::operator() (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationX::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationX::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationX::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationX::operator* (const ROOT::Math::XYZTVector &);
// rotation Y
#pragma link C++ function  ROOT::Math::RotationY::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::RotationY::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationY::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationY::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationY::operator* (const ROOT::Math::XYZTVector &);
// rotation Z
#pragma link C++ function  ROOT::Math::RotationZ::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::RotationZ::operator() (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationZ::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationZ::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::RotationZ::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::RotationZ::operator* (const ROOT::Math::XYZTVector &);

// transform3D
#pragma link C++ function  ROOT::Math::Transform3D::Transform3D (double *, double*);
#pragma link C++ function  ROOT::Math::Transform3D::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Transform3D::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Transform3D::operator() (const ROOT::Math::XYZTVector &);
#pragma link C++ function  ROOT::Math::Transform3D::operator() (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Transform3D::operator() (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Transform3D::operator* (const ROOT::Math::XYZPoint &);
#pragma link C++ function  ROOT::Math::Transform3D::operator* (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Transform3D::operator* (const ROOT::Math::XYZTVector &);

// LorentzRotation
#pragma link C++ function  ROOT::Math::LorentzRotation::LorentzRotation (double *, double*);
#pragma link C++ function  ROOT::Math::LorentzRotation::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::LorentzRotation::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::LorentzRotation::operator* (const ROOT::Math::XYZTVector &);

// Boost
//#pragma link C++ function  ROOT::Math::Boost::Boost (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Boost::Boost (double *, double*);
#pragma link C++ function  ROOT::Math::Boost::SetComponents (const ROOT::Math::XYZVector &);
#pragma link C++ function  ROOT::Math::Boost::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Boost::GetComponents (double *, double *);
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
