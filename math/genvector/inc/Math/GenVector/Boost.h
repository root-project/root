// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 ROOT FNAL MathLib Team                          *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for Boost
//
// Created by: Mark Fischler  Mon Nov 1  2005
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_Boost
#define ROOT_Math_GenVector_Boost 1

#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"

#include "Math/GenVector/BoostX.h"
#include "Math/GenVector/BoostY.h"
#include "Math/GenVector/BoostZ.h"

//#ifdef TEX
/**

   A variable names bgamma appears in several places in this file. A few
   words of elaboration are needed to make its meaning clear.  On page 69
   of Misner, Thorne and Wheeler, (Exercise 2.7) the elements of the matrix
   for a general Lorentz boost are given as

   \f[   \Lambda^{j'}_k = \Lambda^{k'}_j
              = (\gamma - 1) n^j n^k + \delta^{jk}  \f]

   where the n^i are unit vectors in the direction of the three spatial
   axes.  Using the definitions, \f$ n^i = \beta_i/\beta \f$ , then, for example,

   \f[   \Lambda_{xy} = (\gamma - 1) n_x n_y
              = (\gamma - 1) \beta_x \beta_y/\beta^2  \f]

   By definition, \f[   \gamma^2 = 1/(1 - \beta^2)  \f]

   so that   \f[   \gamma^2 \beta^2 = \gamma^2 - 1  \f]

   or   \f[   \beta^2 = (\gamma^2 - 1)/\gamma^2  \f]

   If we insert this into the expression for \f$ \Lambda_{xy} \f$, we get

   \f[   \Lambda_{xy} = (\gamma - 1) \gamma^2/(\gamma^2 - 1) \beta_x \beta_y \f]

   or, finally

   \f[   \Lambda_{xy} = \gamma^2/(\gamma+1) \beta_x \beta_y  \f]

   The expression \f$ \gamma^2/(\gamma+1) \f$ is what we call <em>bgamma</em> in the code below.

   \class ROOT::Math::Boost
*/
//#endif

namespace ROOT {
namespace Math {
namespace Impl {

//__________________________________________________________________________________________
  /**
     Lorentz boost class with the (4D) transformation represented internally
     by a 4x4 orthosymplectic matrix.
     See also BoostX, BoostY and BoostZ for classes representing
     specialized Lorentz boosts.
     Also, the 3-D rotation classes can be considered to be special Lorentz
     transformations which do not mix space and time components.

     @ingroup GenVector
  */

template < typename T >
class Boost {

public:

  typedef T Scalar;

  enum ELorentzRotationMatrixIndex {
      kLXX =  0, kLXY =  1, kLXZ =  2, kLXT =  3
    , kLYX =  4, kLYY =  5, kLYZ =  6, kLYT =  7
    , kLZX =  8, kLZY =  9, kLZZ = 10, kLZT = 11
    , kLTX = 12, kLTY = 13, kLTZ = 14, kLTT = 15
  };

  enum EBoostMatrixIndex {
      kXX =  0, kXY =  1, kXZ =  2, kXT =  3
     , kYY =  4, kYZ =  5, kYT =  6
     , kZZ =  7, kZT =  8
     , kTT =  9
     , kNElems = 10 
  };

  // ========== Constructors and Assignment =====================

  /**
      Default constructor (identity transformation)
  */
  Boost() { SetIdentity(); }

  /**
     Construct given a three Scalars beta_x, beta_y, and beta_z
   */
  Boost(Scalar beta_x, Scalar beta_y, Scalar beta_z)
  { SetComponents(beta_x, beta_y, beta_z); }

  /**
     Construct given a beta vector (which must have methods x(), y(), z())
   */
  template <class Avector>
  explicit
  Boost(const Avector & beta) { SetComponents(beta); }

  /**
     Construct given a pair of pointers or iterators defining the
     beginning and end of an array of three Scalars to use as beta_x, _y, and _z
   */
  template<class IT>
  Boost(IT begin, IT end) { SetComponents(begin,end); }

   /**
      copy constructor
   */
  Boost( Boost<T> const & b) {
      *this = b;
   }

  /**
     Construct from an axial boost
  */

  explicit Boost( BoostX<T> const & bx ) {SetComponents(bx.BetaVector());}
  explicit Boost( BoostY<T> const & by ) {SetComponents(by.BetaVector());}
  explicit Boost( BoostZ<T> const & bz ) {SetComponents(bz.BetaVector());}

  // The compiler-generated copy ctor, copy assignment, and dtor are OK.

   /**
      Assignment operator
    */
   Boost<T> &
   operator=(Boost<T> const & rhs ) {
    for (unsigned int i=0; i < kNElems; ++i) {
       fM[i] = rhs.fM[i];
    }
    return *this;
   }

  /**
     Assign from an axial pure boost
  */
  Boost<T> &
  operator=( BoostX<T> const & bx ) { return operator=(Boost<T>(bx)); }
  Boost<T> &
  operator=( BoostY<T> const & by ) { return operator=(Boost<T>(by)); }
  Boost<T> &
  operator=( BoostZ<T> const & bz ) { return operator=(Boost<T>(bz)); }

  /**
     Re-adjust components to eliminate small deviations from a perfect
     orthosyplectic matrix.
   */
  void Rectify() {
    // Assuming the representation of this is close to a true Lorentz Rotation,
    // but may have drifted due to round-off error from many operations,
    // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
    // again.
    
    if ( fM[kTT] <= Scalar(0) ) {
      GenVector::Throw("Attempt to rectify a boost with non-positive gamma");
    }
    else
    {
      DisplacementVector3D< Cartesian3D<Scalar> > beta ( fM[kXT], fM[kYT], fM[kZT] );
      beta /= fM[kTT];
      if ( beta.mag2() >= 1 ) {
        beta /= ( beta.R() * Scalar( 1.0 + 1.0e-16 ) );
      }
      SetComponents ( beta );
    }
  }

  // ======== Components ==============

  /**
     Set components from beta_x, beta_y, and beta_z
  */
  void
  SetComponents (Scalar bx, Scalar by, Scalar bz) {
    using namespace std;
    // set the boost beta as 3 components
    const Scalar bp2 = bx*bx + by*by + bz*bz;
    if ( bp2 >= Scalar(1) ) {
      GenVector::Throw("Beta Vector supplied to set Boost represents speed >= c");
      // SetIdentity();
    }
    else
    {
      const Scalar gamma = Scalar(1) / sqrt( Scalar(1) - bp2 );
      const Scalar bgamma = gamma * gamma / ( Scalar(1) + gamma );
      fM[kXX] = Scalar(1) + bgamma * bx * bx;
      fM[kYY] = Scalar(1) + bgamma * by * by;
      fM[kZZ] = Scalar(1) + bgamma * bz * bz;
      fM[kXY] = bgamma * bx * by;
      fM[kXZ] = bgamma * bx * bz;
      fM[kYZ] = bgamma * by * bz;
      fM[kXT] = gamma * bx;
      fM[kYT] = gamma * by;
      fM[kZT] = gamma * bz;
      fM[kTT] = gamma;
    }
  }

  /**
     Get components into beta_x, beta_y, and beta_z
  */
  void
  GetComponents (Scalar& bx, Scalar& by, Scalar& bz) const {
    // get beta of the boots as 3 components
    const Scalar gaminv = Scalar(1)/fM[kTT];
    bx = fM[kXT]*gaminv;
    by = fM[kYT]*gaminv;
    bz = fM[kZT]*gaminv;
  }

  /**
     Set components from a beta vector
  */
  template <class Avector>
  void
  SetComponents (const Avector & beta)
   { SetComponents(beta.x(), beta.y(), beta.z()); }

  /**
     Set given a pair of pointers or iterators defining the beginning and end of
     an array of three Scalars to use as beta_x,beta _y, and beta_z
   */
  template<class IT>
#ifndef NDEBUG
  void SetComponents(IT begin, IT end) {
#else
  void SetComponents(IT begin, IT ) {
#endif
    IT a = begin; IT b = ++begin; IT c = ++begin;
    assert (++begin==end);
    SetComponents (*a, *b, *c);
  }

  /**
     Get given a pair of pointers or iterators defining the beginning and end of
     an array of three Scalars into which to place beta_x, beta_y, and beta_z
   */
  template<class IT>
#ifndef NDEBUG
  void GetComponents(IT begin, IT end) const {
#else
  void GetComponents(IT begin, IT ) const {
#endif
    IT a = begin; IT b = ++begin; IT c = ++begin;
    assert (++begin==end);
    GetComponents (*a, *b, *c);
  }

  /**
     Get given a pointer or an iterator defining the beginning of
     an array into which to place beta_x, beta_y, and beta_z
   */
  template<class IT>
  void GetComponents(IT begin ) const {
     T bx,by,bz = 0;
     GetComponents (bx,by,bz);
     *begin++ = bx;
     *begin++ = by;
     *begin = bz;
  }

  /**
     The beta vector for this boost
   */
  typedef DisplacementVector3D<Cartesian3D<T>, DefaultCoordinateSystemTag > XYZVector;
  XYZVector BetaVector() const {
    // get boost beta vector
    const Scalar gaminv = Scalar(1)/fM[kTT];
    return DisplacementVector3D< Cartesian3D<Scalar> >
      ( fM[kXT]*gaminv, fM[kYT]*gaminv, fM[kZT]*gaminv );
  }

  /**
     Get elements of internal 4x4 symmetric representation, into a data
     array suitable for direct use as the components of a LorentzRotation
     Note -- 16 Scalars will be written into the array; if the array is not
     that large, then this will lead to undefined behavior.
  */
  void
  GetLorentzRotation (Scalar r[]) const {
    // get Lorentz rotation corresponding to this boost as an array of 16 values
    r[kLXX] = fM[kXX];  r[kLXY] = fM[kXY];  r[kLXZ] = fM[kXZ];  r[kLXT] = fM[kXT];
    r[kLYX] = fM[kXY];  r[kLYY] = fM[kYY];  r[kLYZ] = fM[kYZ];  r[kLYT] = fM[kYT];
    r[kLZX] = fM[kXZ];  r[kLZY] = fM[kYZ];  r[kLZZ] = fM[kZZ];  r[kLZT] = fM[kZT];
    r[kLTX] = fM[kXT];  r[kLTY] = fM[kYT];  r[kLTZ] = fM[kZT];  r[kLTT] = fM[kTT];
  }

  // =========== operations ==============

  /**
     Lorentz transformation operation on a Minkowski ('Cartesian')
     LorentzVector
  */
  LorentzVector< ROOT::Math::PxPyPzE4D<T> >
  operator() (const LorentzVector< ROOT::Math::PxPyPzE4D<T> > & v) const {
    // apply boost to a PxPyPzE LorentzVector
    const Scalar x = v.Px();
    const Scalar y = v.Py();
    const Scalar z = v.Pz();
    const Scalar t = v.E();
    return LorentzVector< PxPyPzE4D<T> >
      (   fM[kXX]*x + fM[kXY]*y + fM[kXZ]*z + fM[kXT]*t
        , fM[kXY]*x + fM[kYY]*y + fM[kYZ]*z + fM[kYT]*t
        , fM[kXZ]*x + fM[kYZ]*y + fM[kZZ]*z + fM[kZT]*t
        , fM[kXT]*x + fM[kYT]*y + fM[kZT]*z + fM[kTT]*t );
  }

  /**
     Lorentz transformation operation on a LorentzVector in any
     coordinate system
   */
  template <class CoordSystem>
  LorentzVector<CoordSystem>
  operator() (const LorentzVector<CoordSystem> & v) const {
    const LorentzVector< PxPyPzE4D<T> > xyzt(v);
    const LorentzVector< PxPyPzE4D<T> > r_xyzt = operator()(xyzt);
    return LorentzVector<CoordSystem> ( r_xyzt );
  }

  /**
     Lorentz transformation operation on an arbitrary 4-vector v.
     Preconditions:  v must implement methods x(), y(), z(), and t()
     and the arbitrary vector type must have a constructor taking (x,y,z,t)
   */
  template <class Foreign4Vector>
  Foreign4Vector
  operator() (const Foreign4Vector & v) const {
    const LorentzVector< PxPyPzE4D<T> > xyzt(v);
    const LorentzVector< PxPyPzE4D<T> > r_xyzt = operator()(xyzt);
    return Foreign4Vector ( r_xyzt.X(), r_xyzt.Y(), r_xyzt.Z(), r_xyzt.T() );
  }

  /**
     Overload operator * for boost on a vector
   */
  template <class A4Vector>
  inline
  A4Vector operator* (const A4Vector & v) const
  {
    return operator()(v);
  }

  /**
      Invert a Boost in place
   */
  void Invert() {
    // invert in place boost (modifying the object)
    fM[kXT] = -fM[kXT];
    fM[kYT] = -fM[kYT];
    fM[kZT] = -fM[kZT];
  }
 
  /**
      Return inverse of  a boost
   */
  Boost<T> Inverse() const {
   // return inverse of boost
    Boost<T> tmp(*this);
    tmp.Invert();
    return tmp;
  }

  /**
     Equality/inequality operators
   */
  bool operator == (const Boost<T> & rhs) const {
    bool OK = true;
    for (unsigned int i=0; i < kNElems; ++i) {
      if ( fM[i] != rhs.fM[i] ) { OK = false; break; }
    }
    return OK;
  }
  
  bool operator != (const Boost & rhs) const {
    return ! operator==(rhs);
  }

protected:

  void SetIdentity() {
    // set identity boost
    fM[kXX] = Scalar(1); fM[kXY] = Scalar(0); fM[kXZ] = Scalar(0); fM[kXT] = Scalar(0);
    fM[kYY] = Scalar(1); fM[kYZ] = Scalar(0); fM[kYT] = Scalar(0);
    fM[kZZ] = Scalar(1); fM[kZT] = Scalar(0);
    fM[kTT] = Scalar(1);
  }

private:

  Scalar fM[kNElems];

};  // Boost

// ============ Class Boost ends here ============

/**
   Stream Output and Input
 */
  // TODO - I/O should be put in the manipulator form

template< typename T >
std::ostream & operator<< (std::ostream & os, const Boost<T> & b ) {
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form needs formatiing improvements
   T m[16];
   b.GetLorentzRotation(m);
   os << "\n" << m[0]  << "  " << m[1]  << "  " << m[2]  << "  " << m[3];
   os << "\n" << "\t"  << "  " << m[5]  << "  " << m[6]  << "  " << m[7];
   os << "\n" << "\t"  << "  " << "\t"  << "  " << m[10] << "  " << m[11];
   os << "\n" << "\t"  << "  " << "\t"  << "  " << "\t"  << "  " << m[15] << "\n";
   return os;
}

} // namespace Impl

typedef Impl::Boost<double> Boost;
typedef Impl::Boost<float>  BoostF;  
  
} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_GenVector_Boost  */
