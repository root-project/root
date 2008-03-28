// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  


// for template PositionVector3D functions

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator=(const  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator=(const  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator=(const  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator=(const  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator=(const  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator=(const  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > &);

//dot product 

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &) const;
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::Dot(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


// Cross product 

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &) const;
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::Cross(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


// operator +=

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator-=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


// operator -=

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >::operator+=(const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


// operator P = P + V returning point 
#pragma link C++ function ROOT::Math::operator+( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function ROOT::Math::operator+( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function ROOT::Math::operator+( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function ROOT::Math::operator+( ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function ROOT::Math::operator+(ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function ROOT::Math::operator+(ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::operator+(ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::operator+(ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::operator+(ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


// operator P = V + P returning point
// these will not wok since CINT will instantiate those like V = V + P  
#pragma link C++ function ROOT::Math::operator+( const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > & , ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> >  );
#pragma link C++ function ROOT::Math::operator+( const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &, ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> > );
#pragma link C++ function ROOT::Math::operator+( const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &, ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> > );




// operator V = P-P (cannot work in CINT if I can have V = V-V
#pragma link C++ function ROOT::Math::operator-( const ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > &, const ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function ROOT::Math::operator-( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function ROOT::Math::operator-( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> > &);



// // operator P = P-V

#pragma link C++ function ROOT::Math::operator-( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);
#pragma link C++ function ROOT::Math::operator-( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function ROOT::Math::operator-( ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > , const ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);

#pragma link C++ function ROOT::Math::operator-( ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function ROOT::Math::operator-(ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function ROOT::Math::operator-(ROOT::Math::PositionVector3D<ROOT::Math::Polar3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);

#pragma link C++ function  ROOT::Math::operator-(ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > &);
#pragma link C++ function  ROOT::Math::operator-(ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > &);
#pragma link C++ function  ROOT::Math::operator-(ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> >, const  ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > &);


// utility functions

// delta Phi
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaPhi ( const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &);

// deltaR 

#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::DeltaR ( const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &);

// cosTheta

#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::CosTheta ( const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &);

// angle

#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::Polar3D< double> > &);
#pragma link C++ function  ROOT::Math::VectorUtil::Angle ( const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &, const  ROOT::Math::PositionVector3D< ROOT::Math::CylindricalEta3D< double> > &);

