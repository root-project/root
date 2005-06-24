// @(#)root/mathcore:$Name:  $:$Id: LinkDef_Vector4D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

// @(#)root/mathcore:$Name:  $:$Id: LinkDef4.h,v 1.1 2005/06/23 12:15:32 brun Exp $


/// linkdef for LorentzVectors


// forget constructors for the moment
//#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// assignment operator
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
// #pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::EEtaPhiMSystem<double> > &);
// #pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiMSystem<double> > &);


// Dot product 

#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// operator +=
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);

// operator -=
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// operator +
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// operator -
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > &);

// operator * 

//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::LorentzVector<ROOT::Math::Cartesian4D<double> > );
//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::LorentzVector<ROOT::Math::CylindricalEta4D<double> > );

// utility functions

// delta Phi
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// deltaR 

#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// cosTheta

#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// angle

#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// invariantMass 

#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::Cartesian4D< double> > &);

