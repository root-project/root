// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

// Linkdef for Doublr32_t types


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


#pragma link C++ class ROOT::Math::Cartesian2D<Double32_t>+;
#pragma link C++ class ROOT::Math::Polar2D<Double32_t>+;

#pragma link C++ class ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<Double32_t> >+;
#pragma link C++ class ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<Double32_t> >+;


#pragma link C++ class ROOT::Math::Cartesian3D<Double32_t>+;
#pragma link C++ class ROOT::Math::CylindricalEta3D<Double32_t>+;
#pragma link C++ class ROOT::Math::Polar3D<Double32_t>+;
#pragma link C++ class ROOT::Math::Cylindrical3D<Double32_t>+;


#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Double32_t> >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Double32_t> >+;

#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Double32_t> >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Double32_t> >+;

// using a tag (only cartesian and cylindrical eta)

#if 0
// Work around CINT and autoloader deficiency with template default parameter
// Those requests as solely for rlibmap, they do no need to be seen by rootcint
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >+;

#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >+;
#endif

#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t>, ROOT::Math::LocalCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::LocalCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t>, ROOT::Math::GlobalCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::GlobalCoordinateSystemTag >+;

#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::LocalCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::LocalCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::GlobalCoordinateSystemTag >+;
#pragma link C++ class ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::GlobalCoordinateSystemTag >+;


#pragma link C++ class ROOT::Math::PxPyPzE4D<Double32_t>+;
#pragma link C++ class ROOT::Math::PtEtaPhiE4D<Double32_t>+;
#pragma link C++ class ROOT::Math::PxPyPzM4D<Double32_t>+;
#pragma link C++ class ROOT::Math::PtEtaPhiM4D<Double32_t>+;

#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >+;
#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> >+;
#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >+;
#pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<Double32_t> >+;



// #pragma link C++ typedef ROOT::Math::XYZVectorD32;
// #pragma link C++ typedef ROOT::Math::RhoEtaPhiVectorD32;
// #pragma link C++ typedef ROOT::Math::Polar3DVectorD32;

// #pragma link C++ typedef ROOT::Math::XYZPointD32;
// #pragma link C++ typedef ROOT::Math::RhoEtaPhiPointD32;
// #pragma link C++ typedef ROOT::Math::Polar3DPointD32;

// #pragma link C++ typedef ROOT::Math::XYZTVectorD32;
// #pragma link C++ typedef ROOT::Math::PtEtaPhiEVectorD32;
// #pragma link C++ typedef ROOT::Math::PxPyPzMVectorD32;
// #pragma link C++ typedef ROOT::Math::PtEtaPhiMVectorD32;



#endif
