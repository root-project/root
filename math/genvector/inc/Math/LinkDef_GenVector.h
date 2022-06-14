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


#pragma link C++ class    ROOT::Math::Cartesian2D<double>+;
#pragma read sourceClass="ROOT::Math::Cartesian2D<Double32_t>" \
             targetClass="ROOT::Math::Cartesian2D<double>";
#pragma read sourceClass="ROOT::Math::Cartesian2D<float>"      \
             targetClass="ROOT::Math::Cartesian2D<double>";
#pragma read sourceClass="ROOT::Math::Cartesian2D<Float16_t>"  \
             targetClass="ROOT::Math::Cartesian2D<double>";

#pragma link C++ class    ROOT::Math::Polar2D<double>+;
#pragma read sourceClass="ROOT::Math::Polar2D<Double32_t>" \
             targetClass="ROOT::Math::Polar2D<double>";
#pragma read sourceClass="ROOT::Math::Polar2D<float>"      \
             targetClass="ROOT::Math::Polar2D<double>";
#pragma read sourceClass="ROOT::Math::Polar2D<Float16_t>"  \
             targetClass="ROOT::Math::Polar2D<double>";



#pragma link C++ class    ROOT::Math::Cartesian3D<double>+;
#pragma read sourceClass="ROOT::Math::Cartesian3D<Double32_t>" \
             targetClass="ROOT::Math::Cartesian3D<double>";
#pragma read sourceClass="ROOT::Math::Cartesian3D<float>"      \
             targetClass="ROOT::Math::Cartesian3D<double>";
#pragma read sourceClass="ROOT::Math::Cartesian3D<Float16_t>"  \
             targetClass="ROOT::Math::Cartesian3D<double>";

#pragma link C++ class    ROOT::Math::Polar3D<double>+;
#pragma read sourceClass="ROOT::Math::Polar3D<Double32_t>" \
             targetClass="ROOT::Math::Polar3D<double>";
#pragma read sourceClass="ROOT::Math::Polar3D<float>"      \
             targetClass="ROOT::Math::Polar3D<double>";
#pragma read sourceClass="ROOT::Math::Polar3D<Float16_t>"  \
             targetClass="ROOT::Math::Polar3D<double>";

#pragma link C++ class    ROOT::Math::Cylindrical3D<double>+;
#pragma read sourceClass="ROOT::Math::Cylindrical3D<Double32_t>" \
             targetClass="ROOT::Math::Cylindrical3D<double>";
#pragma read sourceClass="ROOT::Math::Cylindrical3D<float>"      \
             targetClass="ROOT::Math::Cylindrical3D<double>";
#pragma read sourceClass="ROOT::Math::Cylindrical3D<Float16_t>"  \
             targetClass="ROOT::Math::Cylindrical3D<double>";

#pragma link C++ class    ROOT::Math::CylindricalEta3D<double>+;
#pragma read sourceClass="ROOT::Math::CylindricalEta3D<Double32_t>" \
             targetClass="ROOT::Math::CylindricalEta3D<double>";
#pragma read sourceClass="ROOT::Math::CylindricalEta3D<float>"      \
             targetClass="ROOT::Math::CylindricalEta3D<double>";
#pragma read sourceClass="ROOT::Math::CylindricalEta3D<Float16_t>"  \
             targetClass="ROOT::Math::CylindricalEta3D<double>";


#pragma link C++ class ROOT::Math::DefaultCoordinateSystemTag+;
#pragma link C++ class ROOT::Math::LocalCoordinateSystemTag+;
#pragma link C++ class ROOT::Math::GlobalCoordinateSystemTag+;

#pragma link C++ class    ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<float> >"      \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double> >";

#pragma link C++ class    ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<double> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<float> >"      \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<double> >";


#pragma link C++ class    ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<double> >+;
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<float> >"      \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<double> >";

#pragma link C++ class    ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<double> >+;
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<float> >"      \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<double> >";



#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float> >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double> >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >";


#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float> >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float> >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double> >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float> >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >";


#ifdef __CLING__
// Work around CINT and autoloader deficiency with template default parameter
// Those requests are solely for rlibmap, they do no need to be seen by rootcint.
#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";


#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >";

#endif

#pragma link C++ class    ROOT::Math::PxPyPzE4D<double>+;
#pragma read sourceClass="ROOT::Math::PxPyPzE4D<Double32_t>" \
             targetClass="ROOT::Math::PxPyPzE4D<double>";
#pragma read sourceClass="ROOT::Math::PxPyPzE4D<float>"      \
             targetClass="ROOT::Math::PxPyPzE4D<double>";
#pragma read sourceClass="ROOT::Math::PxPyPzE4D<Float16_t>"  \
             targetClass="ROOT::Math::PxPyPzE4D<double>";

#pragma link C++ class    ROOT::Math::PtEtaPhiE4D<double>+;
#pragma read sourceClass="ROOT::Math::PtEtaPhiE4D<Double32_t>" \
             targetClass="ROOT::Math::PtEtaPhiE4D<double>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiE4D<float>"      \
             targetClass="ROOT::Math::PtEtaPhiE4D<double>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiE4D<Float16_t>"  \
             targetClass="ROOT::Math::PtEtaPhiE4D<double>";

#pragma link C++ class    ROOT::Math::PxPyPzM4D<double>+;
#pragma read sourceClass="ROOT::Math::PxPyPzM4D<Double32_t>" \
             targetClass="ROOT::Math::PxPyPzM4D<double>";
#pragma read sourceClass="ROOT::Math::PxPyPzM4D<float>"      \
             targetClass="ROOT::Math::PxPyPzM4D<double>";
#pragma read sourceClass="ROOT::Math::PxPyPzM4D<Float16_t>"  \
             targetClass="ROOT::Math::PxPyPzM4D<double>";

#pragma link C++ class    ROOT::Math::PtEtaPhiM4D<double>+;
#pragma read sourceClass="ROOT::Math::PtEtaPhiM4D<Double32_t>" \
             targetClass="ROOT::Math::PtEtaPhiM4D<double>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiM4D<float>"      \
             targetClass="ROOT::Math::PtEtaPhiM4D<double>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiM4D<Float16_t>"  \
             targetClass="ROOT::Math::PtEtaPhiM4D<double>";

//#pragma link C++ class    ROOT::Math::EEtaPhiMSystem<double>+;
#pragma read sourceClass="ROOT::Math::EEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Math::EEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Math::EEtaPhiMSystem<float>"      \
             targetClass="ROOT::Math::EEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Math::EEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Math::EEtaPhiMSystem<double>";

//#pragma link C++ class    ROOT::Math::PtEtaPhiMSystem<double>+;
#pragma read sourceClass="ROOT::Math::PtEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Math::PtEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiMSystem<float>"      \
             targetClass="ROOT::Math::PtEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Math::PtEtaPhiMSystem<double>";


#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >"      \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >";

#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> >"      \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >";

#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<float> >"      \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >";

#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> >"      \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >";


//// Floating types 

#pragma link C++ class    ROOT::Math::Cartesian2D<float>+;
#pragma read sourceClass="ROOT::Math::Cartesian2D<double>"     \
             targetClass="ROOT::Math::Cartesian2D<float>";
#pragma read sourceClass="ROOT::Math::Cartesian2D<Double32_t>" \
             targetClass="ROOT::Math::Cartesian2D<float>";
#pragma read sourceClass="ROOT::Math::Cartesian2D<Float16_t>"  \
             targetClass="ROOT::Math::Cartesian2D<float>";

#pragma link C++ class    ROOT::Math::Polar2D<float>+;
#pragma read sourceClass="ROOT::Math::Polar2D<double>"     \
             targetClass="ROOT::Math::Polar2D<float>";
#pragma read sourceClass="ROOT::Math::Polar2D<Double32_t>" \
             targetClass="ROOT::Math::Polar2D<float>";
#pragma read sourceClass="ROOT::Math::Polar2D<Float16_t>"  \
             targetClass="ROOT::Math::Polar2D<float>";


#pragma link C++ class    ROOT::Math::Cartesian3D<float>+;
#pragma read sourceClass="ROOT::Math::Cartesian3D<double>"     \
             targetClass="ROOT::Math::Cartesian3D<float>";
#pragma read sourceClass="ROOT::Math::Cartesian3D<Double32_t>" \
             targetClass="ROOT::Math::Cartesian3D<float>";
#pragma read sourceClass="ROOT::Math::Cartesian3D<Float16_t>"  \
             targetClass="ROOT::Math::Cartesian3D<float>";

#pragma link C++ class    ROOT::Math::Polar3D<float>+;
#pragma read sourceClass="ROOT::Math::Polar3D<double>"     \
             targetClass="ROOT::Math::Polar3D<float>";
#pragma read sourceClass="ROOT::Math::Polar3D<Double32_t>" \
             targetClass="ROOT::Math::Polar3D<float>";
#pragma read sourceClass="ROOT::Math::Polar3D<Float16_t>"  \
             targetClass="ROOT::Math::Polar3D<float>";

#pragma link C++ class    ROOT::Math::Cylindrical3D<float>+;
#pragma read sourceClass="ROOT::Math::Cylindrical3D<double>"     \
             targetClass="ROOT::Math::Cylindrical3D<float>";
#pragma read sourceClass="ROOT::Math::Cylindrical3D<Double32_t>" \
             targetClass="ROOT::Math::Cylindrical3D<float>";
#pragma read sourceClass="ROOT::Math::Cylindrical3D<Float16_t>"  \
             targetClass="ROOT::Math::Cylindrical3D<float>";

#pragma link C++ class    ROOT::Math::CylindricalEta3D<float>+;
#pragma read sourceClass="ROOT::Math::CylindricalEta3D<double>"     \
             targetClass="ROOT::Math::CylindricalEta3D<float>";
#pragma read sourceClass="ROOT::Math::CylindricalEta3D<Double32_t>" \
             targetClass="ROOT::Math::CylindricalEta3D<float>";
#pragma read sourceClass="ROOT::Math::CylindricalEta3D<Float16_t>"  \
             targetClass="ROOT::Math::CylindricalEta3D<float>";


#pragma link C++ class ROOT::Math::DefaultCoordinateSystemTag+;
#pragma link C++ class ROOT::Math::LocalCoordinateSystemTag+;
#pragma link C++ class ROOT::Math::GlobalCoordinateSystemTag+;

#pragma link C++ class    ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<float> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double> >"     \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<float> >";

#pragma link C++ class    ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<float> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<double> >"     \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector2D<ROOT::Math::Polar2D<float> >";


#pragma link C++ class    ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<float> >+;
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<double> >"     \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<float> >";

#pragma link C++ class    ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<float> >+;
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<double> >"     \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector2D<ROOT::Math::Polar2D<float> >";



#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double> >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float> >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> >";


#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float> >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double> >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float> >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float> >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float> >";


#ifdef __CLING__
// Work around CINT and autoloader deficiency with template default parameter
// Those requests are solely for rlibmap, they do no need to be seen by rootcint.
#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";


#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::Cylindrical3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double>,ROOT::Math::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Double32_t>,ROOT::Math::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<Float16_t>,ROOT::Math::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float>,ROOT::Math::DefaultCoordinateSystemTag >";

#endif

#pragma link C++ class    ROOT::Math::PxPyPzE4D<float>+;
#pragma read sourceClass="ROOT::Math::PxPyPzE4D<double>"     \
             targetClass="ROOT::Math::PxPyPzE4D<float>";
#pragma read sourceClass="ROOT::Math::PxPyPzE4D<Double32_t>" \
             targetClass="ROOT::Math::PxPyPzE4D<float>";
#pragma read sourceClass="ROOT::Math::PxPyPzE4D<Float16_t>"  \
             targetClass="ROOT::Math::PxPyPzE4D<float>";

#pragma link C++ class    ROOT::Math::PtEtaPhiE4D<float>+;
#pragma read sourceClass="ROOT::Math::PtEtaPhiE4D<double>"     \
             targetClass="ROOT::Math::PtEtaPhiE4D<float>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiE4D<Double32_t>" \
             targetClass="ROOT::Math::PtEtaPhiE4D<float>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiE4D<Float16_t>"  \
             targetClass="ROOT::Math::PtEtaPhiE4D<float>";

#pragma link C++ class    ROOT::Math::PxPyPzM4D<float>+;
#pragma read sourceClass="ROOT::Math::PxPyPzM4D<double>"     \
             targetClass="ROOT::Math::PxPyPzM4D<float>";
#pragma read sourceClass="ROOT::Math::PxPyPzM4D<Double32_t>" \
             targetClass="ROOT::Math::PxPyPzM4D<float>";
#pragma read sourceClass="ROOT::Math::PxPyPzM4D<Float16_t>"  \
             targetClass="ROOT::Math::PxPyPzM4D<float>";

#pragma link C++ class    ROOT::Math::PtEtaPhiM4D<float>+;
#pragma read sourceClass="ROOT::Math::PtEtaPhiM4D<double>"     \
             targetClass="ROOT::Math::PtEtaPhiM4D<float>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiM4D<Double32_t>" \
             targetClass="ROOT::Math::PtEtaPhiM4D<float>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiM4D<Float16_t>"  \
             targetClass="ROOT::Math::PtEtaPhiM4D<float>";

//#pragma link C++ class    ROOT::Math::EEtaPhiMSystem<float>+;
#pragma read sourceClass="ROOT::Math::EEtaPhiMSystem<double>"     \
             targetClass="ROOT::Math::EEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Math::EEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Math::EEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Math::EEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Math::EEtaPhiMSystem<float>";

//#pragma link C++ class    ROOT::Math::PtEtaPhiMSystem<float>+;
#pragma read sourceClass="ROOT::Math::PtEtaPhiMSystem<double>"     \
             targetClass="ROOT::Math::PtEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Math::PtEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Math::PtEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Math::PtEtaPhiMSystem<float>";


#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >"     \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >";

#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >"     \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> >";

#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<float> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> >"     \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<float> >";

#pragma link C++ class    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> >+;
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >"     \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >" \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> >";
#pragma read sourceClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Float16_t> >"  \
             targetClass="ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> >";




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

// typedef's


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

#pragma link C++ typedef ROOT::Math::RhoZPhiVector;
#pragma link C++ typedef ROOT::Math::PxPyPzEVector;

// tyoedef for floating types

#pragma link C++ typedef ROOT::Math::XYVectorF;
#pragma link C++ typedef ROOT::Math::Polar2DVectorF;

#pragma link C++ typedef ROOT::Math::XYPointF;
#pragma link C++ typedef ROOT::Math::Polar2DPointF;

#pragma link C++ typedef ROOT::Math::XYZVectorF;
#pragma link C++ typedef ROOT::Math::RhoEtaPhiVectorF;
#pragma link C++ typedef ROOT::Math::Polar3DVectorF;

#pragma link C++ typedef ROOT::Math::XYZPointF;
#pragma link C++ typedef ROOT::Math::RhoEtaPhiPointF;
#pragma link C++ typedef ROOT::Math::Polar3DPointF;

#pragma link C++ typedef ROOT::Math::XYZTVectorF;

// dictionary for points and vectors functions
// not needed with Cling
//#include "LinkDef_Vector3D.h"
//#include "LinkDef_Point3D.h"
//#include "LinkDef_Vector4D.h"
//#include "LinkDef_Rotation.h"

// for std::vector of genvectors
#include "LinkDef_GenVector2.h"


// utility functions

#pragma link C++ namespace ROOT::Math::VectorUtil;



#endif
