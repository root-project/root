// @(#)root/mathcore:$Id$
// Authors:  M. Fischler  2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Boost, a 4x4 symmetric matrix representation of
// an axial Lorentz transformation
//
// Created by: Mark Fischler Mon Nov 1  2005
//
#include "Math/GenVector/Boost.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/GenVector_exception.h"

#include <cmath>
#include <algorithm>

//#ifdef TEX
/**   

	A variable names bgamma appears in several places in this file. A few
	words of elaboration are needed to make its meaning clear.  On page 69
	of Misner, Thorne and Wheeler, (Exercise 2.7) the elements of the matrix
	for a general Lorentz boost are given as

	\f[	\Lambda^{j'}_k = \Lambda^{k'}_j
			     = (\gamma - 1) n^j n^k + \delta^{jk}  \f]

	where the n^i are unit vectors in the direction of the three spatial
	axes.  Using the definitions, \f$ n^i = \beta_i/\beta \f$ , then, for example,

	\f[	\Lambda_{xy} = (\gamma - 1) n_x n_y
			     = (\gamma - 1) \beta_x \beta_y/\beta^2  \f]

	By definition, \f[	\gamma^2 = 1/(1 - \beta^2)  \f]

	so that	\f[	\gamma^2 \beta^2 = \gamma^2 - 1  \f]

	or	\f[	\beta^2 = (\gamma^2 - 1)/\gamma^2  \f]

	If we insert this into the expression for \f$ \Lambda_{xy} \f$, we get

	\f[	\Lambda_{xy} = (\gamma - 1) \gamma^2/(\gamma^2 - 1) \beta_x \beta_y \f]

	or, finally

	\f[	\Lambda_{xy} = \gamma^2/(\gamma+1) \beta_x \beta_y  \f]

	The expression \f$ \gamma^2/(\gamma+1) \f$ is what we call <em>bgamma</em> in the code below.

	\class ROOT::Math::Boost
*/
//#endif

namespace ROOT {

namespace Math {

void Boost::SetIdentity() {
   // set identity boost
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0; fM[kXT] = 0.0;
   fM[kYY] = 1.0; fM[kYZ] = 0.0; fM[kYT] = 0.0;
   fM[kZZ] = 1.0; fM[kZT] = 0.0;
   fM[kTT] = 1.0;
}


void Boost::SetComponents (Scalar bx, Scalar by, Scalar bz) {
   // set the boost beta as 3 components
   Scalar bp2 = bx*bx + by*by + bz*bz;
   if (bp2 >= 1) {
      GenVector::Throw ( 
                              "Beta Vector supplied to set Boost represents speed >= c");
      // SetIdentity(); 
      return;
   }    
   Scalar gamma = 1.0 / std::sqrt(1.0 - bp2);
   Scalar bgamma = gamma * gamma / (1.0 + gamma);
   fM[kXX] = 1.0 + bgamma * bx * bx;
   fM[kYY] = 1.0 + bgamma * by * by;
   fM[kZZ] = 1.0 + bgamma * bz * bz;
   fM[kXY] = bgamma * bx * by;
   fM[kXZ] = bgamma * bx * bz;
   fM[kYZ] = bgamma * by * bz;
   fM[kXT] = gamma * bx;
   fM[kYT] = gamma * by;
   fM[kZT] = gamma * bz;
   fM[kTT] = gamma;
}

void Boost::GetComponents (Scalar& bx, Scalar& by, Scalar& bz) const {
   // get beta of the boots as 3 components
   Scalar gaminv = 1.0/fM[kTT];
   bx = fM[kXT]*gaminv;
   by = fM[kYT]*gaminv;
   bz = fM[kZT]*gaminv;
}

DisplacementVector3D< Cartesian3D<Boost::Scalar> >
Boost::BetaVector() const {
   // get boost beta vector
   Scalar gaminv = 1.0/fM[kTT];
   return DisplacementVector3D< Cartesian3D<Scalar> >
      ( fM[kXT]*gaminv, fM[kYT]*gaminv, fM[kZT]*gaminv );
}

void Boost::GetLorentzRotation (Scalar r[]) const {
   // get Lorentz rotation corresponding to this boost as an array of 16 values 
   r[kLXX] = fM[kXX];  r[kLXY] = fM[kXY];  r[kLXZ] = fM[kXZ];  r[kLXT] = fM[kXT];  
   r[kLYX] = fM[kXY];  r[kLYY] = fM[kYY];  r[kLYZ] = fM[kYZ];  r[kLYT] = fM[kYT];  
   r[kLZX] = fM[kXZ];  r[kLZY] = fM[kYZ];  r[kLZZ] = fM[kZZ];  r[kLZT] = fM[kZT];  
   r[kLTX] = fM[kXT];  r[kLTY] = fM[kYT];  r[kLTZ] = fM[kZT];  r[kLTT] = fM[kTT];  
}

void Boost::Rectify() {
   // Assuming the representation of this is close to a true Lorentz Rotation,
   // but may have drifted due to round-off error from many operations,
   // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
   // again.
   
   if (fM[kTT] <= 0) {	
      GenVector::Throw ( 
                              "Attempt to rectify a boost with non-positive gamma");
      return;
   }    
   DisplacementVector3D< Cartesian3D<Scalar> > beta ( fM[kXT], fM[kYT], fM[kZT] );
   beta /= fM[kTT];
   if ( beta.mag2() >= 1 ) {			    
      beta /= ( beta.R() * ( 1.0 + 1.0e-16 ) );  
   }
   SetComponents ( beta );
}

LorentzVector< PxPyPzE4D<double> >
Boost::operator() (const LorentzVector< PxPyPzE4D<double> > & v) const {
   // apply bosost to a PxPyPzE LorentzVector
   Scalar x = v.Px();
   Scalar y = v.Py();
   Scalar z = v.Pz();
   Scalar t = v.E();
   return LorentzVector< PxPyPzE4D<double> > 
      ( fM[kXX]*x + fM[kXY]*y + fM[kXZ]*z + fM[kXT]*t 
        , fM[kXY]*x + fM[kYY]*y + fM[kYZ]*z + fM[kYT]*t
        , fM[kXZ]*x + fM[kYZ]*y + fM[kZZ]*z + fM[kZT]*t
        , fM[kXT]*x + fM[kYT]*y + fM[kZT]*z + fM[kTT]*t );
}

void Boost::Invert() {
   // invert in place boost (modifying the object)
   fM[kXT] = -fM[kXT];
   fM[kYT] = -fM[kYT];
   fM[kZT] = -fM[kZT];
}

Boost Boost::Inverse() const {
   // return inverse of boost 
   Boost tmp(*this);
   tmp.Invert();
   return tmp; 
}


// ========== I/O =====================

std::ostream & operator<< (std::ostream & os, const Boost & b) {
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form needs formatiing improvements
   double m[16];
   b.GetLorentzRotation(m);
   os << "\n" << m[0]  << "  " << m[1]  << "  " << m[2]  << "  " << m[3]; 
   os << "\n" << "\t"  << "  " << m[5]  << "  " << m[6]  << "  " << m[7]; 
   os << "\n" << "\t"  << "  " << "\t"  << "  " << m[10] << "  " << m[11]; 
   os << "\n" << "\t"  << "  " << "\t"  << "  " << "\t"  << "  " << m[15] << "\n";
   return os;
}

} //namespace Math
} //namespace ROOT
