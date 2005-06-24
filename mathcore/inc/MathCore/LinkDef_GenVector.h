// @(#)root/mathcore:$Name:  $:$Id: LinkDef_GenVector.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 



#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace ROOT::Math;


#pragma link C++ class ROOT::Math::Cartesian3D<double>+;
#pragma link C++ class ROOT::Math::Polar3D<double>+;
#pragma link C++ class ROOT::Math::CylindricalEta3D<double>+;

#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >+;

#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >+;

#pragma link C++ class ROOT::Math::Cartesian4D<double>+;
#pragma link C++ class ROOT::Math::CylindricalEta4D<double>+;
//#pragma link C++ class ROOT::Math::EEtaPhiMSystem<double>+;
//#pragma link C++ class ROOT::Math::PtEtaPhiMSystem<double>+;

#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >+;
#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >+;
//#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::EEtaPhiMSystem<double> >+;
//#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiMSystem<double> >+;

// rotations
#pragma link C++ class ROOT::Math::Rotation3D+;



#pragma link C++ typedef ROOT::Math::XYZVector;
#pragma link C++ typedef ROOT::Math::RhoEtaPhiVector;
#pragma link C++ typedef ROOT::Math::Polar3DVector;

#pragma link C++ typedef ROOT::Math::XYZPoint;
#pragma link C++ typedef ROOT::Math::RhoEtaPhiPoint;
#pragma link C++ typedef ROOT::Math::Polar3DPoint;

#pragma link C++ typedef ROOT::Math::XYZTVector;
#pragma link C++ typedef ROOT::Math::PtEtaPhiEVector;



// dictionary for points and vectors
#include "LinkDef_Vector3D.h"
#include "LinkDef_Point3D.h"
#include "LinkDef_Vector4D.h"



// rotation func
//#pragma link C++ function  ROOT::Math::Rotation3D::operator() (const ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D> &);
#pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (double *, double*);
#pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (double *, double *);
#pragma link C++ function  ROOT::Math::Rotation3D::GetComponents (double *, double *);

// #include "TMatrix.h"
// #pragma link C++ function  ROOT::Math::Rotation3D::Rotation3D (const TMatrixD &m);
// #pragma link C++ function  ROOT::Math::Rotation3D::operator= (const TMatrixD &m);
// #pragma link C++ function  ROOT::Math::Rotation3D::SetComponents (const TMatrixD &m);
// #pragma link C++ function  ROOT::Math::Rotation3D::GetComponents (TMatrixD &m);



//#pragma extra_include "TVectorD.h";
// #pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >(const TVectorD &, size_t);
//#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::AssignFrom(const TVectorD &, size_t);
//#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::AssignFrom(const TVectorD &, size_t);
//#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::AssignFrom(const TVectorD &, size_t);


// utility functions
//#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi < ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > , ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > >( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &)

#pragma link C++ namespace ROOT::Math::VectorUtil;




#pragma extra_include "vector";
#include <vector>

//#pragma link C++ class vector<ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > >+;
#pragma link C++ class vector<ROOT::Math::XYZTVector >+;
#pragma link C++ class vector<ROOT::Math::XYZVector >+;
#pragma link C++ class vector<ROOT::Math::XYZPoint >+;


#endif
