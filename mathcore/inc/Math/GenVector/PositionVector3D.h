// @(#)root/mathcore:$Name:  $:$Id: PositionVector3D.h,v 1.1 2005/09/18 17:33:47 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class PositionVector3D
//
// Created by: Lorenzo Moneta  at Mon May 30 15:25:04 2005
//
// Last update: $Id: PositionVector3D.h,v 1.1 2005/09/18 17:33:47 brun Exp $
//
#ifndef ROOT_Math_GenVector_PositionVector3D 
#define ROOT_Math_GenVector_PositionVector3D  1

#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/GenVectorIO.h"

#include <cassert>

namespace ROOT {

  namespace Math {


    /**
              Class describing a generic position vector (point) in 3 dimensions.
              This class is templated on the type of Coordinate system.
              One example is the XYZPoint which is a vector based on
              double precision x,y,z data members by using the
              Cartesian3D<double> Coordinate system.
    */

    template <class CoordSystem>
    class PositionVector3D {

    public:

      typedef typename CoordSystem::Scalar Scalar;
      typedef CoordSystem CoordinateType;

      // ------ ctors ------

      /**
         Default constructor. Construct an empty object with zero values
      */

      PositionVector3D() : fCoordinates() { }

      /**
         Construct from three values of type <em>Scalar</em>.
         In the case of a XYZPoint the values are x,y,z
         In the case of  a polar vector they are r,theta,phi
      */
      PositionVector3D(const Scalar & a, const Scalar & b, const Scalar & c) :
        fCoordinates ( a , b,  c)  { }

     /**
          Construct from a position vector expressed in different
          coordinates, or using a different Scalar type
      */
      template <class T>
      explicit PositionVector3D( const PositionVector3D<T> & v) :
        fCoordinates ( v.Coordinates() ) { }

     /**
          Construct from an arbitrary displacement vector
      */
      template <class T>
      explicit PositionVector3D( const DisplacementVector3D<T> & p) :
        fCoordinates ( p.Coordinates() ) { }

      /**
          Construct from a foreign 3D vector type, for example, Hep3Vector
          Precondition: v must implement methods x(), y() and z()
      */
      template <class ForeignVector>
      explicit PositionVector3D( const ForeignVector & v) :
        fCoordinates ( Cartesian3D<Scalar>( v.x(), v.y(), v.z() ) ) { }

#ifdef LATER
      /**
         construct from a generic linear algebra  vector of at least size 3
         implementing operator []. This could be also a C array
         \par v  LAVector
         \par index0   index where coordinates starts (typically zero)
         It works for all Coordinates types,
         ( x= v[index0] for Cartesian and r=v[index0] for Polar )
      */
      template <class LAVector>
      PositionVector3D(const LAVector & v, size_t index0 ) {
        fCoordinates = CoordSystem  ( v[index0], v[index0+1], v[index0+2] );
      }
#endif

      // compiler-generated copy ctor and dtor are fine.

      // ------ assignment ------

      /**
          Assignment operator from a position vector of arbitrary type
      */
      template <class OtherCoords>
      PositionVector3D & operator=
                        ( const PositionVector3D<OtherCoords> & v) {
        fCoordinates = v.Coordinates();
        return *this;
      }

      /**
          Assignment operator from a displacement vector of arbitrary type
      */
      template <class OtherCoords>
      PositionVector3D & operator=
                        ( const DisplacementVector3D<OtherCoords> & v) {
        fCoordinates = v.Coordinates();
        return *this;
      }

      /**
          Assignment from a foreign 3D vector type, for example, Hep3Vector
          Precondition: v must implement methods x(), y() and z()
      */
      template <class ForeignVector>
      PositionVector3D & operator= ( const ForeignVector & v) {
        SetXYZ( v.x(),  v.y(), v.z() );
        return *this;
      }

#ifdef LATER
      /**
         assign from a generic linear algebra  vector of at least size 3
         implementing operator [].
         \par v  LAVector
         \par index0   index where coordinates starts (typically zero)
         It works for all Coordinates types,
         ( x= v[index0] for Cartesian and r=v[index0] for Polar )
      */
      template <class LAVector>
      PositionVector3D & assignFrom(const LAVector & v, size_t index0 = 0) {
        fCoordinates = CoordSystem  ( v[index0], v[index0+1], v[index0+2] );
        return *this;
      }
#endif

      /**
          Retrieve a copy of the coordinates object
      */
      const CoordSystem & Coordinates() const {
        return fCoordinates;
      }

      /**
         Set internal data based on a C-style array of 3 Scalar numbers
       */
      void SetCoordinates( const Scalar src[] )
                            { fCoordinates.SetCoordinates(src);  }

      /**
         Set internal data based on 3 Scalar numbers
       */
      void SetCoordinates( Scalar a, Scalar b, Scalar c )
                            { fCoordinates.SetCoordinates(a, b, c);  }

      /**
         Set internal data based on 3 Scalars at *begin to *end
       */
      template <class IT>
      void SetCoordinates( IT begin, IT end ) {
        assert( begin != end && begin+1 != end && begin+2 != end);
        fCoordinates.SetCoordinates(*begin, *(begin+1), *(begin+2));
      }

      /**
        get internal data into 3 Scalar numbers
       */
      void GetCoordinates( Scalar& a, Scalar& b, Scalar& c ) const
                            { fCoordinates.GetCoordinates(a, b, c);  }

      /**
         get internal data into a C-style array of 3 Scalar numbers
       */
      void GetCoordinates( Scalar dest[] ) const
                            { fCoordinates.GetCoordinates(dest);  }

      /**
         get internal data into 3 Scalars at *begin to *end
       */
      template <class IT>
      void GetCoordinates( IT begin, IT end ) const
      { IT a = begin; IT b = ++begin; IT c = ++begin;
        assert (++begin==end);
        GetCoordinates (*a,*b,*c);
      }

      /**
         set the values of the vector from the cartesian components (x,y,z)
         (if the vector is held in polar or cylindrical eta coordinates,
         then (x, y, z) are converted to that form)
       */
      void SetXYZ (Scalar x, Scalar y, Scalar z) {
            fCoordinates =  Cartesian3D<Scalar> (x,y,z);
      }

      // ------------------- Equality -----------------

      /**
        Exact equality
       */
      bool operator==(const PositionVector3D & rhs) const {
        return fCoordinates==rhs.fCoordinates;
      }
      bool operator!= (const PositionVector3D & rhs) const {
        return !(operator==(rhs));
      }

      // ------ Individual element access, in various coordinate systems ------

      /**
          Cartesian X, converting if necessary from internal coordinate system.
      */
      Scalar X() const { return fCoordinates.X(); }

      /**
          Cartesian Y, converting if necessary from internal coordinate system.
      */
      Scalar Y() const { return fCoordinates.Y(); }

      /**
          Cartesian Z, converting if necessary from internal coordinate system.
      */
      Scalar Z() const { return fCoordinates.Z(); }

      /**
          Polar R, converting if necessary from internal coordinate system.
      */
      Scalar R() const { return fCoordinates.R(); }

      /**
          Polar theta, converting if necessary from internal coordinate system.
      */
      Scalar Theta() const { return fCoordinates.Theta(); }

      /**
          Polar phi, converting if necessary from internal coordinate system.
      */
      Scalar Phi() const { return fCoordinates.Phi(); }

      /**
          Polar eta, converting if necessary from internal coordinate system.
      */
      Scalar Eta() const { return fCoordinates.Eta(); }

      /**
          Cylindrical transverse component rho
      */
      Scalar Rho() const { return fCoordinates.Rho(); }

      // ----- Other fundamental properties -----

      /**
          Magnitute squared ( r^2 in spherical coordinate)
      */
      Scalar Mag2() const { return fCoordinates.Mag2();}

      /**
         Transverse component squared (rho^2 in cylindrical coordinates.
      */
      Scalar Perp2() const { return fCoordinates.Perp2();}

      // It is physically meaningless to speak of the unit vector corresponding
      // to a point.

      // ------ Setting individual elements present in coordinate system ------

      /**
         Change X - Cartesian3D coordinates only
      */
      void SetX (Scalar x) { fCoordinates.SetX(x); }

      /**
         Change Y - Cartesian3D coordinates only
      */
      void SetY (Scalar y) { fCoordinates.SetY(y); }

      /**
         Change Z - Cartesian3D coordinates only
      */
      void SetZ (Scalar z) { fCoordinates.SetZ(z); }

      /**
         Change R - Polar3D coordinates only
      */
      void SetR (Scalar r) { fCoordinates.SetR(r); }

      /**
         Change Theta - Polar3D coordinates only
      */
      void SetTheta (Scalar theta) { fCoordinates.SetTheta(theta); }

      /**
         Change Phi - Polar3D or CylindricalEta3D coordinates
      */
      void SetPhi (Scalar phi) { fCoordinates.SetPhi(phi); }

      /**
         Change Rho - CylindricalEta3D coordinates only
      */
      void SetRho (Scalar rho) { fCoordinates.SetRho(rho); }

      /**
         Change Eta - CylindricalEta3D coordinates only
      */
      void SetEta (Scalar eta) { fCoordinates.SetEta(eta); }

      // ------ Operations combining two vectors ------

     /**
          Return the scalar (Dot) product of this with a displacement vector.
      */
      template< class OtherVector >
      Scalar Dot( const  OtherVector & v) const {
        return X()*v.x() + Y()*v.y() + Z()*v.z();
      }


      /**
         Return vector (Cross) product of this point with a displacement, as a
         point vector in this coordinate system of the first.
      */
      template< class OtherVector >
      PositionVector3D Cross( const OtherVector & v) const  {
        PositionVector3D <CoordinateType> result;
        result.SetXYZ (  Y()*v.z() - v.y()*Z(),
                         Z()*v.x() - v.z()*X(),
                         X()*v.y() - v.x()*Y() );
        return result;
      }

      // The Dot and Cross products of a pair of point vectors are physically
      // meaningless concepts and thus are defined as private methods

      // It is physically meaningless to speak of the Unit vector corresponding
      // to a point.


      /**
          Self Addition with a displacement vector.
      */
#ifndef __CINT__
      template <class CoordSystem2>
      PositionVector3D & operator+= (const  DisplacementVector3D<CoordSystem2> & v)
#else
      template <class V>
      PositionVector3D & operator+= (const  V & v)
#endif
      {
        SetXYZ( X() + v.X(), Y() + v.Y(), Z() + v.Z() );
        return *this;
      }

      /**
          Self Difference with a displacement vector.
      */
#ifndef __CINT__
      template <class CoordSystem2>
      PositionVector3D & operator-= (const  DisplacementVector3D<CoordSystem2> & v)
#else
      template <class V>
      PositionVector3D & operator-= (const  V & v)
#endif
      {
        SetXYZ(  X() - v.X(), Y() - v.Y(), Z() - v.Z() );
        return *this;
      }

      // Limited backward name compatibility with CLHEP

      Scalar x()     const { return X();     }
      Scalar y()     const { return Y();     }
      Scalar z()     const { return Z();     }
      Scalar r()     const { return R();     }
      Scalar theta() const { return Theta(); }
      Scalar phi()   const { return Phi();   }
      Scalar eta()   const { return Eta();   }
      Scalar rho()   const { return Rho();   }
      Scalar mag2()  const { return Mag2();  }
      Scalar perp2() const { return Perp2(); }

    private:

      CoordSystem fCoordinates;

      // Prohibited methods
      /**
         Dot product of two position vectors is inappropriate
      */
      template <class T2>
      PositionVector3D Dot( const PositionVector3D<T2> & v) const;

      /**
         Cross product of two position vectors is inappropriate
      */
      template <class T2>
      PositionVector3D Cross( const PositionVector3D<T2> & v) const;

    };

// ---------- PositionVector3D class template ends here ----------------
// ---------------------------------------------------------------------

    /**
        Difference between two PositionVector3D vectors.
        The result is a DisplacementVector3D.
        The (coordinate system) type of the returned vector is defined to
        be identical to that of the first position vector.
    */

    template <class CoordSystem1, class CoordSystem2>
    inline
    DisplacementVector3D<CoordSystem1>
    operator-( const PositionVector3D<CoordSystem1> & v1,
               const PositionVector3D<CoordSystem2> & v2) {
      return DisplacementVector3D<CoordSystem1>( Cartesian3D<typename CoordSystem1::Scalar>(
                                                                               v1.X()-v2.X(), v1.Y()-v2.Y(),v1.Z()-v2.Z() )
                                             );
    }

    /**
        Addition of a PositionVector3D and a DisplacementVector3D.
        The return type is a PositionVector3D,
        of the same (coordinate system) type as the input PositionVector3D.
    */
    template <class CoordSystem1, class CoordSystem2>
    inline
    PositionVector3D<CoordSystem2>
    operator+( PositionVector3D<CoordSystem2> p1,
               const DisplacementVector3D<CoordSystem1>  & v2)        {
      return p1 += v2;
    }

    /**
        Addition of a DisplacementVector3D and a PositionVector3D.
        The return type is a PositionVector3D,
        of the same (coordinate system) type as the input PositionVector3D.
    */
    template <class CoordSystem1, class CoordSystem2>
    inline
    PositionVector3D<CoordSystem2>
    operator+( DisplacementVector3D<CoordSystem1> const & v1,
               PositionVector3D<CoordSystem2> p2)        {
      return p2 += v1;
    }

    /**
        Subtraction of a DisplacementVector3D from a PositionVector3D.
        The return type is a PositionVector3D,
        of the same (coordinate system) type as the input PositionVector3D.
    */
    template <class CoordSystem1, class CoordSystem2>
    inline
    PositionVector3D<CoordSystem2>
    operator-( PositionVector3D<CoordSystem2> p1,
               DisplacementVector3D<CoordSystem1> const & v2)        {
      return p1 -= v2;
    }

    // Scaling of a position vector with a real number is not physically meaningful

    // ------------- I/O to/from streams -------------

    template< class char_t, class traits_t, class T >
      inline
      std::basic_ostream<char_t,traits_t> &
      operator << ( std::basic_ostream<char_t,traits_t> & os
                  , PositionVector3D<T> const & v
                  )
    {
      if( !os )  return os;

      typename T::Scalar a, b, c;
      v.GetCoordinates(a, b, c);

      if( detail::get_manip( os, detail::bitforbit ) )  {
        detail::set_manip( os, detail::bitforbit, '\00' );
        typedef GenVector_detail::BitReproducible BR;
        BR::Output(os, a);
        BR::Output(os, b);
        BR::Output(os, c);
      }
      else  {
        os << detail::get_manip( os, detail::open  ) << a
           << detail::get_manip( os, detail::sep   ) << b
           << detail::get_manip( os, detail::sep   ) << c
           << detail::get_manip( os, detail::close );
      }

      return os;

    }  // op<< <>()


    template< class char_t, class traits_t, class T >
      inline
      std::basic_istream<char_t,traits_t> &
      operator >> ( std::basic_istream<char_t,traits_t> & is
                  , PositionVector3D<T> & v
                  )
    {
      if( !is )  return is;

      typename T::Scalar a, b, c;

      if( detail::get_manip( is, detail::bitforbit ) )  {
        detail::set_manip( is, detail::bitforbit, '\00' );
        typedef GenVector_detail::BitReproducible BR;
        BR::Input(is, a);
        BR::Input(is, b);
        BR::Input(is, c);
      }
      else  {
        detail::require_delim( is, detail::open  );  is >> a;
        detail::require_delim( is, detail::sep   );  is >> b;
        detail::require_delim( is, detail::sep   );  is >> c;
        detail::require_delim( is, detail::close );
      }

      if( is )
        v.SetCoordinates(a, b, c);
      return is;

    }  // op>> <>()



  } // namespace Math

} // namespace ROOT


#endif /* ROOT_Math_GenVector_PositionVector3D  */
