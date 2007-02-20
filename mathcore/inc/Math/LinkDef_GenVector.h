// @(#)root/mathcore:$Name:  $:$Id: LinkDef_GenVector.h,v 1.9 2006/06/30 15:23:33 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  



#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT::Math;


#pragma link C++ class ROOT::Math::Cartesian3D<double>+;
#pragma link C++ class ROOT::Math::Polar3D<double>+;
#pragma link C++ class ROOT::Math::Cylindrical3D<double>+;
#pragma link C++ class ROOT::Math::CylindricalEta3D<double>+;

#pragma link C++ class ROOT::Math::DefaultCoordinateSystemTag+; 
#pragma link C++ class ROOT::Math::LocalCoordinateSystemTag+; 
#pragma link C++ class ROOT::Math::GlobalCoordinateSystemTag+; 

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
#pragma link C++ class ROOT::Math::RotationX+;
#pragma link C++ class ROOT::Math::RotationY+;
#pragma link C++ class ROOT::Math::RotationZ+;
#pragma link C++ class ROOT::Math::LorentzRotation+;
#pragma link C++ class ROOT::Math::Boost+;
#pragma link C++ class ROOT::Math::BoostX+;
#pragma link C++ class ROOT::Math::BoostY+;
#pragma link C++ class ROOT::Math::BoostZ+;


#pragma link C++ class ROOT::Math::Transform3D+;
#pragma link C++ class ROOT::Math::Plane3D+;

//#endif


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
#pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (double *, double*);
#pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Rotation3D::GetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Rotation3D::operator() (const ROOT::Math::XYZTVector &);

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




#pragma extra_include "vector";
#include <vector>


// conflict on solaris between template class T from std::vector and T(). 
#ifndef __sun      
#pragma link C++ class vector<ROOT::Math::XYZTVector >+;
#endif
// problem on Windows: CINT cannot deal with  too long class name
#ifdef TEST_LATER

#ifndef _WIN32   
#pragma link C++ class vector<ROOT::Math::XYZVector >+;
#pragma link C++ class vector<ROOT::Math::XYZPoint >+;
#endif

// exclude this (they make loibrary  too big and are not used in tests in CVS)

#pragma link C++ class vector<ROOT::Math::Polar3DVector >+;
#pragma link C++ class vector<ROOT::Math::Polar3DPoint >+;

//too long names in Windows - skip this dictionary
#ifndef _WIN32
#ifndef __sun
#pragma link C++ class vector<ROOT::Math::PtEtaPhiEVector >+;
#endif
#pragma link C++ class vector<ROOT::Math::RhoEtaPhiVector >+;
#pragma link C++ class vector<ROOT::Math::RhoEtaPhiPoint >+;

#endif


// // dictionary fo rextra types 
// #pragma extra_include "Rtypes.h";
// #pragma link C++ class ROOT::Math::Cartesian3D<Double32_t>+;
// #pragma link C++ class ROOT::Math::PxPyPzE4D<Double32_t>+;
// #pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> >+;
// #pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >+;


#endif
