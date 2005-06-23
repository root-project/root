// @(#)root/mathcore:$Name:  $:$Id: LinkDef4.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

/// linkdef for LorentzVectors


// forget constructors for the moment
//#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// assignment operator
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
// #pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::EEtaPhiMSystem<double> > &);
// #pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::PtEtaPhiMSystem<double> > &);


// Dot product 

#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::Dot(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::Dot(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::Dot(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::Dot(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// operator +=
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator+=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator+=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);

// operator -=
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator-=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator-=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-=(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// operator +
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator+(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator+(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator+(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);


// operator -
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator-(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> >::operator-(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-(const  ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > &);
#pragma link C++ function  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> >::operator-(const  ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > &);

// operator * 

//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::BasicLorentzVector<ROOT::Math::Cartesian4D<double> > );
//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::BasicLorentzVector<ROOT::Math::CylindricalEta4D<double> > );

// utility functions

// delta Phi
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// deltaR 

#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// cosTheta

#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// angle

#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &);

// invariantMass 

#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::BasicLorentzVector< ROOT::Math::CylindricalEta4D< double> > &, const  ROOT::Math::BasicLorentzVector< ROOT::Math::Cartesian4D< double> > &);

