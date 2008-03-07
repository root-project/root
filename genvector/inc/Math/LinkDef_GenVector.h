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



// dictionary for points and vectors
#include "LinkDef_Vector3D.h"
#include "LinkDef_Point3D.h"
#include "LinkDef_Vector4D.h"
#include "LinkDef_Rotation.h"



// rotation functions
//#pragma link C++ function  ROOT::Math::Rotation3D::operator() (const ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D> &);

// #include "TMatrix.h"
// #pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (const TMatrixD &m);
// #pragma link C++ function  ROOT::Math::Rotation3D::operator= (const TMatrixD &m);
// #pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (const TMatrixD &m);
// #pragma link C++ function  ROOT::Math::Rotation3D::GetComponents (TMatrixD &m);



//#pragma extra_include "TVectorD.h";
// #pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >(const TVectorD &, size_t);
//#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::AssignFrom(const TVectorD &, size_t);
//#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::AssignFrom(const TVectorD &, size_t);
//#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::AssignFrom(const TVectorD &, size_t);


// utility functions
//#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi < ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > , ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > >( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &)

#pragma link C++ namespace ROOT::Math::VectorUtil;

// problem on Windows: CINT cannot deal with  too long class name
// generated by an std::vector<GenVector> 
#ifndef _WIN32
#pragma extra_include "vector";
#include <vector>


// conflict on solaris between template class T from std::vector and T(). 
#ifndef __sun      
#pragma link C++ class vector<ROOT::Math::XYZTVector >+;
#pragma link C++ class vector<ROOT::Math::PtEtaPhiEVector >+;
#endif

#pragma link C++ class vector<ROOT::Math::XYZVector >+;
#pragma link C++ class vector<ROOT::Math::XYZPoint >+;

#pragma link C++ class vector<ROOT::Math::RhoEtaPhiVector >+;
#pragma link C++ class vector<ROOT::Math::RhoEtaPhiPoint >+;


#endif       // endif Win32




#endif
