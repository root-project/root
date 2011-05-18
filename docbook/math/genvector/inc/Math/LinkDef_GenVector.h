// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  



#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Math;


#pragma link C++ class ROOT::Math::Cartesian2D<double>+;
#pragma link C++ class ROOT::Math::Polar2D<double>+;


#pragma link C++ class ROOT::Math::Cartesian3D<double>+;
#pragma link C++ class ROOT::Math::Polar3D<double>+;
#pragma link C++ class ROOT::Math::Cylindrical3D<double>+;
#pragma link C++ class ROOT::Math::CylindricalEta3D<double>+;

#pragma link C++ class ROOT::Math::DefaultCoordinateSystemTag+; 
#pragma link C++ class ROOT::Math::LocalCoordinateSystemTag+; 
#pragma link C++ class ROOT::Math::GlobalCoordinateSystemTag+; 

#pragma link C++ class ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double> >+;
#pragma link C++ class ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<double> >+;

#pragma link C++ class ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<double> >+;
#pragma link C++ class ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<double> >+;


#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >+;

#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >+;

#if 0
// Work around CINT and autoloader deficiency with template default parameter
// Those requests are solely for rlibmap, they do no need to be seen by rootcint.
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;

#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#endif

#pragma link C++ class ROOT::Math::PxPyPzE4D<double>+;
#pragma link C++ class ROOT::Math::PtEtaPhiE4D<double>+;
#pragma link C++ class ROOT::Math::PxPyPzM4D<double>+;
#pragma link C++ class ROOT::Math::PtEtaPhiM4D<double>+;
//#pragma link C++ class ROOT::Math::EEtaPhiMSystem<double>+;
//#pragma link C++ class ROOT::Math::PtEtaPhiMSystem<double>+;

#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >+;
#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >+;
#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >+;
#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >+;

// rotations
//#ifdef LATER

#pragma link C++ class ROOT::Math::Rotation3D+;
#pragma link C++ class ROOT::Math::AxisAngle+;
#pragma link C++ class ROOT::Math::EulerAngles+;
#pragma link C++ class ROOT::Math::Quaternion+;
#pragma link C++ class ROOT::Math::RotationZYX+;
#pragma link C++ class ROOT::Math::RotationX+;
#pragma link C++ class ROOT::Math::RotationY+;
#pragma link C++ class ROOT::Math::RotationZ+;
#pragma link C++ class ROOT::Math::LorentzRotation+;
#pragma link C++ class ROOT::Math::Boost+;
#pragma link C++ class ROOT::Math::BoostX+;
#pragma link C++ class ROOT::Math::BoostY+;
#pragma link C++ class ROOT::Math::BoostZ+;


#pragma link C++ class ROOT::Math::Plane3D+;
#pragma link C++ class ROOT::Math::Transform3D+;
#pragma link C++ class ROOT::Math::Translation3D+;

//#endif

#pragma link C++ typedef ROOT::Math::XYVector;
#pragma link C++ typedef ROOT::Math::Polar2DVector;

#pragma link C++ typedef ROOT::Math::XYPoint;
#pragma link C++ typedef ROOT::Math::Polar2DPoint;

#pragma link C++ typedef ROOT::Math::XYZVector;
#pragma link C++ typedef ROOT::Math::RhoEtaPhiVector;
#pragma link C++ typedef ROOT::Math::Polar3DVector;

#pragma link C++ typedef ROOT::Math::XYZPoint;
#pragma link C++ typedef ROOT::Math::RhoEtaPhiPoint;
#pragma link C++ typedef ROOT::Math::Polar3DPoint;

#pragma link C++ typedef ROOT::Math::XYZTVector;
#pragma link C++ typedef ROOT::Math::PtEtaPhiEVector;
#pragma link C++ typedef ROOT::Math::PxPyPzMVector;
#pragma link C++ typedef ROOT::Math::PtEtaPhiMVector;




// dictionary for points and vectors functions

#include "LinkDef_Vector3D.h"
#include "LinkDef_Point3D.h"
#include "LinkDef_Vector4D.h"
#include "LinkDef_Rotation.h"

#ifndef _WIN32
// for std::vector of genvectors
#include "LinkDef_GenVector2.h"
#endif


// utility functions

#pragma link C++ namespace ROOT::Math::VectorUtil;



#endif
