// @(#)root/mathcore:$Name:  $:$Id: LinkDef_Vector3D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

// @(#)root/mathcore:$Name:  $:$Id: LinkDef2.h,v 1.1 2005/06/23 12:15:32 brun Exp $


// for template DisplacementVector3D functions

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

// operator + 
#pragma link C++ function ROOT::Math::operator+( ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function ROOT::Math::operator+( ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function ROOT::Math::operator+( ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

// operator -
#pragma link C++ function ROOT::Math::operator-( ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function ROOT::Math::operator-( ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function ROOT::Math::operator-( ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

// operator * 

#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > );
//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > );
//#pragma link C++ function ROOT::Math::operator*( const double & , ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > );


// utility functions

// delta Phi
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &);

// deltaR 

#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &);

// cosTheta

#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &);

// angle

#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::DisplacementVector3D< ROOT::Math::CylindricalEta3D< double> > &);

