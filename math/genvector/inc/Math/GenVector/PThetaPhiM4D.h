// @(#)root/mathcore:$Id$
// Authors: W.R. Armstrong (2018)
//
#ifndef ROOT_Math_GenVector_PThetaPhiM4D
#define ROOT_Math_GenVector_PThetaPhiM4D  1

#include "Math/GenVector/eta.h"

#include "Math/GenVector/GenVector_exception.h"

#include "Math/GenVector/Polar3D.h"



#include <cmath>

namespace ROOT {

namespace Math {

//__________________________________________________________________________________________
/**
    Class describing a four-vector constructed from a three-vector and invariant mass. 
    Vectors stored as (P = \sqrt(Px^2+Py^2+Pz^2)), Theta, Phi, M) making use of 
    the ROOT::Math::Polar3D class.
    The metric used is (-,-,-,+)
    Spacelike particles (M2 < 0) are described with negative mass values,
    but in this case m2 must alwasy be less than P2 to preserve a positive value of E2

    @ingroup GenVector
*/

template <class ScalarType = double>
class PThetaPhiM4D {

public :

   typedef ScalarType Scalar;
   
   using Polar3D = typename ROOT::Math::Polar3D<ScalarType>;

   // --------- Constructors ---------------

   /**
      Default constructor  with r=theta=phi=m=0
   */
   PThetaPhiM4D() : fPolar3D(0.0,0.0,0.0), fM(0.0) { }


   /**
      Constructor  from x, y , z , m values
   */
   PThetaPhiM4D(Scalar p, Scalar th, Scalar phi, Scalar m) :
      fPolar3D(p, th, phi), fM(m) {

      if (fM < 0) RestrictNegMass();
   }

   PThetaPhiM4D(const Polar3D& p, Scalar m) :
      fPolar3D(p), fM(m)
   {
      if (fM < 0) RestrictNegMass();
   }

   /**
      construct from any 4D  coordinate system class
      implementing R(), Theta(), Phi() and M()
   */
   template <class CoordSystem>
   explicit PThetaPhiM4D(const CoordSystem & v) :
      fPolar3D( v.R(), v.Theta(), v.Phi() ), fM( v.M() )
   { }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // so we decided to re-implement them ( there is no no need to have them with g++4)
   // 
   ///**
   //   copy constructor
   // */
   //PThetaPhiM4D(const PThetaPhiM4D & v) :
   //   fX(v.fX), fY(v.fY), fZ(v.fZ), fM(v.fM) { }
   // Ignoring this problem because those are old compilers! -Whit
   PThetaPhiM4D(const PThetaPhiM4D& v)  = default;

   /**
      assignment operator
    */
   PThetaPhiM4D & operator = (const PThetaPhiM4D&) = default;


   /**
      construct from any 4D  coordinate system class
      implementing R(), Theta(), Phi() and M()
   */
   template <class AnyCoordSystem>
   PThetaPhiM4D& operator = (const AnyCoordSystem& v) {
      fPolar3D = Polar3D(v.R(), v.Theta(), v.Phi());
      fM = v.M();
      return *this;
   }

   /**
      Set internal data based on an array of 4 Scalar numbers
   */
   void SetCoordinates( const Scalar src[] ) {
     fPolar3D.SetCoordinates(src);
     fM=src[3];
      if (fM < 0) RestrictNegMass();
   }

   /**
      get internal data into an array of 4 Scalar numbers
   */
   void GetCoordinates( Scalar dest[] ) const
   { fPolar3D.GetCoordinates(dest); dest[3] = fM; }

   /**
      Set internal data based on 4 Scalar numbers
   */
   void SetCoordinates(Scalar  p, Scalar  th, Scalar  phi, Scalar m) {
      fPolar3D = {p,th,phi}; fM=m;
      if (fM < 0) RestrictNegMass();
   }

   /**
      get internal data into 4 Scalar numbers
   */
   void GetCoordinates(Scalar& p, Scalar& th, Scalar& phi, Scalar& m) const
   { p=fPolar3D.R(); th==fPolar3D.Theta(); phi=fPolar3D.Phi(); m=fM;}

   // --------- Coordinates and Coordinate-like Scalar properties -------------

   // cartesian (Minkowski)coordinate accessors

   Scalar Px() const { return fPolar3D.X();}
   Scalar Py() const { return fPolar3D.Y();}
   Scalar Pz() const { return fPolar3D.Z();}
   Scalar M() const  { return fM; }

   Scalar X() const { return fPolar3D.X();}
   Scalar Y() const { return fPolar3D.Y();}
   Scalar Z() const { return fPolar3D.Z();}

   // other coordinate representation
   /**
      Energy
    */
   Scalar E() const { return sqrt(E2()); }

   Scalar T() const { return E();}

   /**
      squared magnitude of spatial components
   */
   Scalar P2() const { return fPolar3D.Mag2(); }

   /**
      magnitude of spatial components (magnitude of 3-momentum)
   */
   Scalar P() const { return fPolar3D.R(); }
   Scalar R() const { return fPolar3D.R(); }

   /**
      vector magnitude squared (or mass squared)
      In case of negative mass (spacelike particles return negative values)
   */
   Scalar M2() const   {
      return ( fM  >= 0 ) ?  fM*fM :  -fM*fM;
   }
   Scalar Mag2() const { return M2(); }

   Scalar Mag() const    { return M(); }

   /**
      energy squared
   */
   Scalar E2() const {
      Scalar e2 =  P2() + M2();
      // protect against numerical errors when M2() is negative
      return e2 > 0 ? e2 : 0;
   }

   /**
       transverse spatial component squared
   */
   Scalar Pt2()   const { return fPolar3D.Perp2();}
   Scalar Perp2() const { return fPolar3D.Perp2();}

   /**
      Transverse spatial component (P_perp or rho)
   */
   Scalar Pt()   const { return fPolar3D.Rho();}
   Scalar Perp() const { return fPolar3D.Rho();}
   Scalar Rho()  const { return fPolar3D.Rho();}

   /**
       transverse mass squared
   */
   Scalar Mt2() const { return Pt2() + M2(); }

   /**
      transverse mass
   */
   Scalar Mt() const {
      const Scalar mm = Mt2();
      if (mm >= 0) {
         return sqrt(mm);
      } else {
         GenVector::Throw ("PThetaPhiM4D::Mt() - Tachyonic:\n"
                           "    Pz^2 > E^2 so the transverse mass would be imaginary");
         return -sqrt(-mm);
      }
   }

   /**
       transverse energy squared
   */
   Scalar Et2() const {  // is (E^2 * pt ^2) / p^2
      // but it is faster to form p^2 from pt^2
      Scalar pt2 = Pt2();
      return pt2 == 0 ? 0 : E2() * pt2/( pt2 + Z()*Z() );
   }

   /**
      transverse energy
   */
   Scalar Et() const {
      const Scalar etet = Et2();
      return sqrt(etet);
   }

   /**
      azimuthal angle
   */
   Scalar Phi() const { return fPolar3D.Phi(); }//(fX == 0.0 && fY == 0.0) ? 0.0 : atan2(fY, fX); }

   /**
      polar angle
   */
   Scalar Theta() const { return fPolar3D.Theta(); }//(fX == 0.0 && fY == 0.0 && fZ == 0.0) ? 0 : atan2(Pt(), fZ); }

   /**
       pseudorapidity
   */
   Scalar Eta() const { return fPolar3D.Eta(); }

   //Polar3D Vector() const { return fPolar3D; }

   // --------- Set Coordinates of this system  ---------------


   /**
      set X value
   */
   void SetP( Scalar  p) {
      fPolar3D.SetR(p);
   }
   /**
      set Y value
   */
   void SetPy( Scalar  th) {
      fPolar3D.SetTheta(th);
   }
   /**
      set Z value
   */
   void SetPz( Scalar  phi) {
      fPolar3D.SetPhi(phi);
   }
   /**
      set T value
   */
   void SetM( Scalar  m) {
      fM = m;
      if (fM < 0) RestrictNegMass();
   }

   /**
       set all values
   */
   void SetPxPyPzE4D(Scalar p, Scalar th, Scalar phi, Scalar e);

   // ------ Manipulations -------------

   /**
      negate the 4-vector -  Note that the energy cannot be negate (would need an additional data member)
      therefore negate will work only on the spatial components.
      One would need to use negate only with vectors having the energy as data members
   */
   void Negate( ) {
     fPolar3D.Negate();
     GenVector::Throw ("PThetaPhiM4D::Negate - cannot negate the energy - can negate only the spatial components");
   }

   /**
      scale coordinate values by a scalar quantity a
   */
   void Scale( const Scalar & a) {
      fPolar3D.Scale(a);
      fM *= a;
   }


   /**
      Exact equality
   */
   bool operator == (const PThetaPhiM4D & rhs) const {
      return fPolar3D == rhs.fPolar3D && fM == rhs.fM;
   }
   bool operator != (const PThetaPhiM4D & rhs) const {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   Scalar x() const { return X(); }
   Scalar y() const { return Y(); }
   Scalar z() const { return Z(); }
   Scalar t() const { return E(); }



#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   void SetPt(Scalar pt);

   void SetEta(Scalar eta);

   void SetPhi(Scalar phi);

   void SetE(Scalar t);

#endif

private:

   // restrict the value of negative mass to avoid unphysical negative E2 values
   // M2 must be less than P2 for the tachionic particles - otherwise use positive values
   inline void RestrictNegMass() {
      if ( fM >=0 ) return;
      if ( P2() - fM*fM  < 0 ) {
         GenVector::Throw("PThetaPhiM4D::unphysical value of mass, set to closest physical value");
         fM = - P();
      }
      return;
   }


   /**
      (contigous) data containing the coordinate values x,y,z,t
   */

   Polar3D fPolar3D;
   ScalarType fM;

};

} // end namespace Math
} // end namespace ROOT


// move implementations here to avoid circle dependencies

#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/PtEtaPhiM4D.h"

namespace ROOT {

namespace Math {

template <class ScalarType>
inline void PThetaPhiM4D<ScalarType>::SetPxPyPzE4D(Scalar px, Scalar py, Scalar pz, Scalar e) {
   *this = PxPyPzE4D<Scalar> (px, py, pz, e);
}


#if defined(__MAKECINT__) || defined(G__DICTIONARY)

  // ====== Set member functions for coordinates in other systems =======

  // ====== Set member functions for coordinates in other systems =======

template <class ScalarType>
inline void PThetaPhiM4D<ScalarType>::SetPt(ScalarType pt) {
   GenVector_exception e("PThetaPhiM4D::SetPt() is not supposed to be called");
   throw e;
   PtEtaPhiM4D<ScalarType> v(*this); v.SetPt(pt); *this = PThetaPhiM4D<ScalarType>(v);
}
template <class ScalarType>
inline void PThetaPhiM4D<ScalarType>::SetEta(ScalarType eta) {
   GenVector_exception e("PThetaPhiM4D::SetEta() is not supposed to be called");
   throw e;
   PtEtaPhiM4D<ScalarType> v(*this); v.SetEta(eta); *this = PThetaPhiM4D<ScalarType>(v);
}
template <class ScalarType>
inline void PThetaPhiM4D<ScalarType>::SetPhi(ScalarType phi) {
   GenVector_exception e("PThetaPhiM4D::SetPhi() is not supposed to be called");
   throw e;
   PtEtaPhiM4D<ScalarType> v(*this); v.SetPhi(phi); *this = PThetaPhiM4D<ScalarType>(v);
}
template <class ScalarType>
inline void PThetaPhiM4D<ScalarType>::SetE(ScalarType energy) {
   GenVector_exception e("PThetaPhiM4D::SetE() is not supposed to be called");
   throw e;
   PThetaPhiM4D<ScalarType> v(*this); v.SetE(energy);
   *this = PThetaPhiM4D<ScalarType>(v);
}


#endif  // endif __MAKE__CINT || G__DICTIONARY

} // end namespace Math

} // end namespace ROOT



#endif // ROOT_Math_GenVector_PThetaPhiM4D
