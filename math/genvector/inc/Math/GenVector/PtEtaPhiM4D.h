// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class PtEtaPhiM4D
//
// Created by: fischler at Wed Jul 21 2005
//   Similar to PtEtaPhiMSystem by moneta
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_PtEtaPhiM4D
#define ROOT_Math_GenVector_PtEtaPhiM4D  1

#include "Math/Math.h"

#include "Math/GenVector/etaMax.h"

#include "Math/GenVector/GenVector_exception.h"


//#define TRACE_CE
#ifdef TRACE_CE
#include <iostream>
#endif

#include <cmath>

namespace ROOT {

namespace Math {

//__________________________________________________________________________________________
/**
    Class describing a 4D cylindrical coordinate system
    using Pt , Phi, Eta and M (mass)
    The metric used is (-,-,-,+).
    Spacelike particles (M2 < 0) are described with negative mass values,
    but in this case m2 must alwasy be less than P2 to preserve a positive value of E2
    Phi is restricted to be in the range [-PI,PI)

    @ingroup GenVector
*/

template <class ScalarType>
class PtEtaPhiM4D {

public :

   typedef ScalarType Scalar;

   // --------- Constructors ---------------

   /**
      Default constructor gives zero 4-vector (with zero mass)
   */
   PtEtaPhiM4D() : fPt(0), fEta(0), fPhi(0), fM(0) { }

   /**
      Constructor  from pt, eta, phi, mass values
   */
   PtEtaPhiM4D(Scalar pt, Scalar eta, Scalar phi, Scalar mass) :
      fPt(pt), fEta(eta), fPhi(phi), fM(mass) {
      RestrictPhi();
      if (fM < 0) RestrictNegMass();
   }

   /**
      Generic constructor from any 4D coordinate system implementing
      Pt(), Eta(), Phi() and M()
   */
   template <class CoordSystem >
   explicit PtEtaPhiM4D(const CoordSystem & c) :
      fPt(c.Pt()), fEta(c.Eta()), fPhi(c.Phi()), fM(c.M())  { RestrictPhi(); }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // so we decided to re-implement them ( there is no no need to have them with g++4)

   /**
      copy constructor
    */
   PtEtaPhiM4D(const PtEtaPhiM4D & v) :
      fPt(v.fPt), fEta(v.fEta), fPhi(v.fPhi), fM(v.fM) { }

   /**
      assignment operator
    */
   PtEtaPhiM4D & operator = (const PtEtaPhiM4D & v) {
      fPt  = v.fPt;
      fEta = v.fEta;
      fPhi = v.fPhi;
      fM   = v.fM;
      return *this;
   }


   /**
      Set internal data based on an array of 4 Scalar numbers
   */
   void SetCoordinates( const Scalar src[] ) {
      fPt=src[0]; fEta=src[1]; fPhi=src[2]; fM=src[3];
      RestrictPhi();
      if (fM <0) RestrictNegMass();
   }

   /**
      get internal data into an array of 4 Scalar numbers
   */
   void GetCoordinates( Scalar dest[] ) const
   { dest[0] = fPt; dest[1] = fEta; dest[2] = fPhi; dest[3] = fM; }

   /**
      Set internal data based on 4 Scalar numbers
   */
   void SetCoordinates(Scalar pt, Scalar eta, Scalar phi, Scalar mass) {
      fPt=pt; fEta = eta; fPhi = phi; fM = mass;
      RestrictPhi();
      if (fM <0) RestrictNegMass();
   }

   /**
      get internal data into 4 Scalar numbers
   */
   void
   GetCoordinates(Scalar& pt, Scalar & eta, Scalar & phi, Scalar& mass) const
   { pt=fPt; eta=fEta; phi = fPhi; mass = fM; }

   // --------- Coordinates and Coordinate-like Scalar properties -------------

   // 4-D Cylindrical eta coordinate accessors

   Scalar Pt()  const { return fPt;  }
   Scalar Eta() const { return fEta; }
   Scalar Phi() const { return fPhi; }
   /**
       M() is the invariant mass;
       in this coordinate system it can be negagative if set that way.
   */
   Scalar M()   const { return fM;   }
   Scalar Mag() const { return M(); }

   Scalar Perp()const { return Pt(); }
   Scalar Rho() const { return Pt(); }

   // other coordinate representation

   Scalar Px() const { return fPt * std::cos(fPhi); }
   Scalar X () const { return Px();         }
   Scalar Py() const { return fPt * std::sin(fPhi); }
   Scalar Y () const { return Py();         }
   Scalar Pz() const {
      return fPt > 0 ? fPt * std::sinh(fEta) : fEta == 0 ? 0 : fEta > 0 ? fEta - etaMax<Scalar>() : fEta + etaMax<Scalar>();
   }
   Scalar Z () const { return Pz(); }

   /**
       magnitude of momentum
   */
   Scalar P() const {
      return fPt > 0 ? fPt * std::cosh(fEta)
                     : fEta > etaMax<Scalar>() ? fEta - etaMax<Scalar>()
                                               : fEta < -etaMax<Scalar>() ? -fEta - etaMax<Scalar>() : 0;
   }
   Scalar R() const { return P(); }

   /**
       squared magnitude of spatial components (momentum squared)
   */
   Scalar P2() const
   {
      const Scalar p = P();
      return p * p;
   }

   /**
       energy squared
   */
   Scalar E2() const {
      Scalar e2 =  P2() + M2();
      // avoid rounding error which can make E2 negative when M2 is negative
      return e2 > 0 ? e2 : 0;
   }

   /**
       Energy (timelike component of momentum-energy 4-vector)
   */
   Scalar E() const { return std::sqrt(E2()); }

   Scalar T()   const { return E();  }

   /**
      vector magnitude squared (or mass squared)
      In case of negative mass (spacelike particles return negative values)
   */
   Scalar M2() const   {
      return ( fM  >= 0 ) ?  fM*fM :  -fM*fM;
   }
   Scalar Mag2() const { return M2();  }

   /**
       transverse spatial component squared
   */
   Scalar Pt2()   const { return fPt*fPt;}
   Scalar Perp2() const { return Pt2();  }

   /**
       transverse mass squared
   */
   Scalar Mt2() const { return M2()  + fPt*fPt; }

   /**
      transverse mass - will be negative if Mt2() is negative
   */
   Scalar Mt() const {
      const Scalar mm = Mt2();
      if (mm >= 0) {
         return std::sqrt(mm);
      } else {
         GenVector::Throw  ("PtEtaPhiM4D::Mt() - Tachyonic:\n"
                            "    Pz^2 > E^2 so the transverse mass would be imaginary");
         return -std::sqrt(-mm);
      }
   }

   /**
       transverse energy squared
   */
   Scalar Et2() const {
      // a bit faster than et * et
      return 2. * E2() / (std::cosh(2 * fEta) + 1);
   }

   /**
      transverse energy
   */
   Scalar Et() const { return E() / std::cosh(fEta); }

private:
   inline static Scalar pi() { return M_PI; }
   inline void RestrictPhi() {
      if (fPhi <= -pi() || fPhi > pi()) fPhi = fPhi - std::floor(fPhi / (2 * pi()) + .5) * 2 * pi();
   }
   // restrict the value of negative mass to avoid unphysical negative E2 values
   // M2 must be less than P2 for the tachionic particles - otherwise use positive values
   inline void RestrictNegMass() {
      if (fM < 0) {
         if (P2() - fM * fM < 0) {
            GenVector::Throw("PtEtaPhiM4D::unphysical value of mass, set to closest physical value");
            fM = -P();
         }
      }
   }

public:

   /**
      polar angle
   */
   Scalar Theta() const { return (fPt > 0 ? Scalar(2) * std::atan(exp(-fEta)) : fEta >= 0 ? 0 : pi()); }

   // --------- Set Coordinates of this system  ---------------

   /**
      set Pt value
   */
   void SetPt( Scalar  pt) {
      fPt = pt;
   }
   /**
      set eta value
   */
   void SetEta( Scalar  eta) {
      fEta = eta;
   }
   /**
      set phi value
   */
   void SetPhi( Scalar  phi) {
      fPhi = phi;
      RestrictPhi();
   }
   /**
      set M value
   */
   void SetM( Scalar  mass) {
      fM = mass;
      if (fM <0) RestrictNegMass();
   }

   /**
       set values using cartesian coordinate system
   */
   void SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e);


   // ------ Manipulations -------------

   /**
      negate the 4-vector -- Note that the energy cannot be negate (would need an additional data member)
      therefore negate will work only on the spatial components
      One would need to use negate only with vectors having the energy as data members
   */
   void Negate( ) {
      fPhi = ( (fPhi > 0) ? fPhi - pi() : fPhi + pi()  );
      fEta = - fEta;
      GenVector::Throw ("PtEtaPhiM4D::Negate - cannot negate the energy - can negate only the spatial components");
   }

   /**
      Scale coordinate values by a scalar quantity a
   */
   void Scale( Scalar a) {
      if (a < 0) {
         Negate(); a = -a;
      }
      fPt *= a;
      fM  *= a;
   }

   /**
      Assignment from a generic coordinate system implementing
      Pt(), Eta(), Phi() and M()
   */
   template <class CoordSystem >
   PtEtaPhiM4D & operator = (const CoordSystem & c) {
      fPt  = c.Pt();
      fEta = c.Eta();
      fPhi = c.Phi();
      fM   = c.M();
      return *this;
   }

   /**
      Exact equality
   */
   bool operator == (const PtEtaPhiM4D & rhs) const {
      return fPt == rhs.fPt && fEta == rhs.fEta
         && fPhi == rhs.fPhi && fM == rhs.fM;
   }
   bool operator != (const PtEtaPhiM4D & rhs) const {return !(operator==(rhs));}

   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   Scalar x() const { return X(); }
   Scalar y() const { return Y(); }
   Scalar z() const { return Z(); }
   Scalar t() const { return E(); }


#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   void SetPx(Scalar px);

   void SetPy(Scalar py);

   void SetPz(Scalar pz);

   void SetE(Scalar t);

#endif

private:

   ScalarType fPt;
   ScalarType fEta;
   ScalarType fPhi;
   ScalarType fM;

};


} // end namespace Math
} // end namespace ROOT


// move implementations here to avoid circle dependencies
#include "Math/GenVector/PxPyPzE4D.h"



namespace ROOT {

namespace Math {


template <class ScalarType>
inline void PtEtaPhiM4D<ScalarType>::SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e) {
   *this = PxPyPzE4D<Scalar> (px, py, pz, e);
}


#if defined(__MAKECINT__) || defined(G__DICTIONARY)

  // ====== Set member functions for coordinates in other systems =======

template <class ScalarType>
void PtEtaPhiM4D<ScalarType>::SetPx(Scalar px) {
   GenVector_exception e("PtEtaPhiM4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPx(px); *this = PtEtaPhiM4D<Scalar>(v);
}
template <class ScalarType>
void PtEtaPhiM4D<ScalarType>::SetPy(Scalar py) {
   GenVector_exception e("PtEtaPhiM4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPy(py); *this = PtEtaPhiM4D<Scalar>(v);
}
template <class ScalarType>
void PtEtaPhiM4D<ScalarType>::SetPz(Scalar pz) {
   GenVector_exception e("PtEtaPhiM4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPz(pz); *this = PtEtaPhiM4D<Scalar>(v);
}
template <class ScalarType>
void PtEtaPhiM4D<ScalarType>::SetE(Scalar energy) {
   GenVector_exception e("PtEtaPhiM4D::SetE() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetE(energy);   *this = PtEtaPhiM4D<Scalar>(v);
}

#endif  // endif __MAKE__CINT || G__DICTIONARY

} // end namespace Math

} // end namespace ROOT



#endif // ROOT_Math_GenVector_PtEtaPhiM4D
