// @(#)root/mathcore:$Name:  $:$Id: LinkDef_Vector4D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  


/// linkdef for LorentzVectors


// forget constructors for the moment
//#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);


// assignment operator
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
// #pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::EEtaPhiMSystem<double> > &);
// #pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator=(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiMSystem<double> > &);


// Dot product 

#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::Dot(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);


// operator +=
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator+=(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);

// operator -=
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator-=(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);


// operator +
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator+(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);


// operator -
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > &);
#pragma link C++ function  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >::operator-(const  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > &);

// operator * 

//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > );
//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > );

// utility functions

// delta Phi
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &);

// deltaR 

#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &);

// cosTheta

#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &);

// angle

#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &);

// invariantMass 

#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::InvariantMass ( const  ROOT::Math::LorentzVector< ROOT::Math::PtEtaPhiE4D< double> > &, const  ROOT::Math::LorentzVector< ROOT::Math::PxPyPzE4D< double> > &);

