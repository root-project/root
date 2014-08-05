// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class PositionVector2D
//
// Created by: Lorenzo Moneta  at Mon Apr 16 2007
//
//
#ifndef ROOT_Math_GenVector_PositionVector2D
#define ROOT_Math_GenVector_PositionVector2D  1

#ifndef ROOT_Math_GenVector_DisplacementVector2Dfwd
#include "Math/GenVector/DisplacementVector2D.h"
#endif

#ifndef ROOT_Math_GenVector_GenVectorIO
#include "Math/GenVector/GenVectorIO.h"
#endif

#ifndef ROOT_Math_GenVector_BitReproducible
#include "Math/GenVector/BitReproducible.h"
#endif

#ifndef ROOT_Math_GenVector_CoordinateSystemTags
#include "Math/GenVector/CoordinateSystemTags.h"
#endif


namespace ROOT {

   namespace Math {


//__________________________________________________________________________________________
      /**
         Class describing a generic position vector (point) in 2 dimensions.
         This class is templated on the type of Coordinate system.
         One example is the XYPoint which is a vector based on
         double precision x,y data members by using the
         ROOT::Math::Cartesian2D<double> Coordinate system.
         The class is having also an extra template parameter, the coordinate system tag,
         to be able to identify (tag) vector described in different reference coordinate system,
         like global or local coordinate systems.

         @ingroup GenVector
      */

    template <class CoordSystem, class Tag = DefaultCoordinateSystemTag >
    class PositionVector2D {

    public:

       typedef typename CoordSystem::Scalar Scalar;
       typedef CoordSystem CoordinateType;
       typedef Tag  CoordinateSystemTag;

       // ------ ctors ------

       /**
          Default constructor. Construct an empty object with zero values
      */

       PositionVector2D() : fCoordinates() { }

       /**
          Construct from three values of type <em>Scalar</em>.
          In the case of a XYPoint the values are x,y
          In the case of  a polar vector they are r,phi
       */
       PositionVector2D(const Scalar & a, const Scalar & b) :
          fCoordinates ( a , b)  { }

       /**
          Construct from a position vector expressed in different
          coordinates, or using a different Scalar type
       */
       template <class T>
       explicit PositionVector2D( const PositionVector2D<T,Tag> & v) :
          fCoordinates ( v.Coordinates() ) { }

       /**
          Construct from an arbitrary displacement vector
       */
       template <class T>
       explicit PositionVector2D( const DisplacementVector2D<T,Tag> & p) :
          fCoordinates ( p.Coordinates() ) { }

       /**
          Construct from a foreign 2D vector type, for example, Hep2Vector
          Precondition: v must implement methods x() and  y()
       */
       template <class ForeignVector>
       explicit PositionVector2D( const ForeignVector & v) :
          fCoordinates ( Cartesian2D<Scalar>( v.x(), v.y() ) ) { }

       // compiler-generated copy ctor and dtor are fine.

       // ------ assignment ------

       /**
          Assignment operator from a position vector of arbitrary type
       */
       template <class OtherCoords>
       PositionVector2D & operator=
       ( const PositionVector2D<OtherCoords,Tag> & v) {
          fCoordinates = v.Coordinates();
          return *this;
       }

       /**
          Assignment operator from a displacement vector of arbitrary type
       */
       template <class OtherCoords>
       PositionVector2D & operator=
       ( const DisplacementVector2D<OtherCoords,Tag> & v) {
          fCoordinates = v.Coordinates();
          return *this;
       }

       /**
          Assignment from a foreign 2D vector type, for example, Hep2Vector
          Precondition: v must implement methods x() and y()
       */
       template <class ForeignVector>
       PositionVector2D & operator= ( const ForeignVector & v) {
          SetXY( v.x(),  v.y() );
          return *this;
       }

       /**
          Retrieve a copy of the coordinates object
       */
       const CoordSystem & Coordinates() const {
          return fCoordinates;
       }

       /**
          Set internal data based on 2 Scalar numbers.
          These are for example (x,y) for a cartesian vector or (r,phi) for a polar vector
       */
       PositionVector2D<CoordSystem, Tag>& SetCoordinates( Scalar a, Scalar b) {
          fCoordinates.SetCoordinates(a, b);
          return *this;
       }


       /**
          get internal data into 2 Scalar numbers.
          These are for example (x,y) for a cartesian vector or (r,phi) for a polar vector
       */
       void GetCoordinates( Scalar& a, Scalar& b) const
       { fCoordinates.GetCoordinates(a, b);  }


       /**
          set the values of the vector from the cartesian components (x,y)
          (if the vector is held in polar coordinates,
          then (x, y) are converted to that form)
       */
       PositionVector2D<CoordSystem, Tag>& SetXY (Scalar a, Scalar b) {
          fCoordinates.SetXY (a,b);
          return *this;
       }

       // ------------------- Equality -----------------

       /**
          Exact equality
       */
       bool operator==(const PositionVector2D & rhs) const {
          return fCoordinates==rhs.fCoordinates;
       }
       bool operator!= (const PositionVector2D & rhs) const {
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
          Polar R, converting if necessary from internal coordinate system.
       */
       Scalar R() const { return fCoordinates.R(); }

       /**
          Polar phi, converting if necessary from internal coordinate system.
       */
       Scalar Phi() const { return fCoordinates.Phi(); }

       /**
          Magnitute squared ( r^2 in spherical coordinate)
       */
       Scalar Mag2() const { return fCoordinates.Mag2();}


       // It is physically meaningless to speak of the unit vector corresponding
       // to a point.

       // ------ Setting individual elements present in coordinate system ------

       /**
          Change X - Cartesian2D coordinates only
       */
       PositionVector2D<CoordSystem, Tag>& SetX (Scalar a) {
          fCoordinates.SetX(a);
          return *this;
       }

       /**
          Change Y - Cartesian2D coordinates only
       */
       PositionVector2D<CoordSystem, Tag>& SetY (Scalar a) {
          fCoordinates.SetY(a);
          return *this;
       }


       /**
          Change R - Polar2D coordinates only
       */
       PositionVector2D<CoordSystem, Tag>& SetR (Scalar a) {
          fCoordinates.SetR(a);
          return *this;
       }

       /**
          Change Phi - Polar2D coordinates
       */
       PositionVector2D<CoordSystem, Tag>& SetPhi (Scalar ang) {
          fCoordinates.SetPhi(ang);
          return *this;
       }


       // ------ Operations combining two vectors ------
       // need to specialize to exclude those with a different tags

       /**
        Return the scalar (Dot) product of this with a displacement vector in
        any coordinate system, but with the same tag
       */
       template< class OtherCoords >
       Scalar Dot( const  DisplacementVector2D<OtherCoords,Tag> & v) const {
          return X()*v.x() + Y()*v.y();
       }


       // The Dot product of a pair of point vectors are physically
       // meaningless concepts and thus are defined as private methods


       /**
          Self Addition with a displacement vector.
       */
       template <class OtherCoords>
       PositionVector2D & operator+= (const  DisplacementVector2D<OtherCoords,Tag> & v)
       {
          SetXY( X() + v.X(), Y() + v.Y() );
          return *this;
       }

       /**
          Self Difference with a displacement vector.
       */
       template <class OtherCoords>
       PositionVector2D & operator-= (const  DisplacementVector2D<OtherCoords,Tag> & v)
       {
          SetXY(  X() - v.X(), Y() - v.Y() );
          return *this;
       }

       /**
          multiply this vector by a scalar quantity
       */
       PositionVector2D & operator *= (Scalar a) {
          fCoordinates.Scale(a);
          return *this;
       }

       /**
          divide this vector by a scalar quantity
       */
       PositionVector2D & operator /= (Scalar a) {
          fCoordinates.Scale(1/a);
          return *this;
       }

       // The following methods (v*a and v/a) could instead be free functions.
       // They were moved into the class to solve a problem on AIX.
       /**
          Multiply a vector by a real number
       */
       PositionVector2D operator * ( Scalar a ) const {
          PositionVector2D tmp(*this);
          tmp *= a;
          return tmp;
       }

       /**
          Division of a vector with a real number
       */
       PositionVector2D operator / (Scalar a) const {
          PositionVector2D tmp(*this);
          tmp /= a;
          return tmp;
       }

       /**
          Rotate by an angle
       */
       void Rotate( Scalar angle) {
          return fCoordinates.Rotate(angle);
       }

       // Limited backward name compatibility with CLHEP

       Scalar x()     const { return fCoordinates.X();     }
       Scalar y()     const { return fCoordinates.Y();     }
       Scalar r()     const { return fCoordinates.R();     }
       Scalar phi()   const { return fCoordinates.Phi();   }
       Scalar mag2()  const { return fCoordinates.Mag2();  }

    private:

       CoordSystem fCoordinates;

       // Prohibited methods

       // this should not compile (if from a vector or points with different tag

       template <class OtherCoords, class OtherTag>
       explicit PositionVector2D( const PositionVector2D<OtherCoords, OtherTag> & );

       template <class OtherCoords, class OtherTag>
       explicit PositionVector2D( const DisplacementVector2D<OtherCoords, OtherTag> & );

       template <class OtherCoords, class OtherTag>
       PositionVector2D & operator=( const PositionVector2D<OtherCoords, OtherTag> & );

       template <class OtherCoords, class OtherTag>
       PositionVector2D & operator=( const DisplacementVector2D<OtherCoords, OtherTag> & );

       template <class OtherCoords, class OtherTag>
       PositionVector2D & operator+=(const  DisplacementVector2D<OtherCoords, OtherTag> & );

       template <class OtherCoords, class OtherTag>
       PositionVector2D & operator-=(const  DisplacementVector2D<OtherCoords, OtherTag> & );

//       /**
//          Dot product of two position vectors is inappropriate
//       */
//       template <class T2, class U>
//       PositionVector2D Dot( const PositionVector2D<T2,U> & v) const;



    };

// ---------- PositionVector2D class template ends here ----------------
// ---------------------------------------------------------------------

      /**
         Multiplication of a position vector by real number  a*v
      */
      template <class CoordSystem, class U>
      inline
      PositionVector2D<CoordSystem>
      operator * ( typename PositionVector2D<CoordSystem,U>::Scalar a,
                   PositionVector2D<CoordSystem,U> v) {
         return v *= a;
         // Note - passing v by value and using operator *= may save one
         // copy relative to passing v by const ref and creating a temporary.
      }

      /**
         Difference between two PositionVector2D vectors.
         The result is a DisplacementVector2D.
         The (coordinate system) type of the returned vector is defined to
         be identical to that of the first position vector.
      */

      template <class CoordSystem1, class CoordSystem2, class U>
      inline
      DisplacementVector2D<CoordSystem1,U>
      operator-( const PositionVector2D<CoordSystem1,U> & v1,
                 const PositionVector2D<CoordSystem2,U> & v2) {
         return DisplacementVector2D<CoordSystem1,U>( Cartesian2D<typename CoordSystem1::Scalar>(
                                                         v1.X()-v2.X(), v1.Y()-v2.Y() )
            );
      }

      /**
         Addition of a PositionVector2D and a DisplacementVector2D.
         The return type is a PositionVector2D,
         of the same (coordinate system) type as the input PositionVector2D.
      */
      template <class CoordSystem1, class CoordSystem2, class U>
      inline
      PositionVector2D<CoordSystem2,U>
      operator+( PositionVector2D<CoordSystem2,U> p1,
                 const DisplacementVector2D<CoordSystem1,U>  & v2)        {
         return p1 += v2;
      }

      /**
         Addition of a DisplacementVector2D and a PositionVector2D.
         The return type is a PositionVector2D,
         of the same (coordinate system) type as the input PositionVector2D.
      */
      template <class CoordSystem1, class CoordSystem2, class U>
      inline
      PositionVector2D<CoordSystem2,U>
      operator+( DisplacementVector2D<CoordSystem1,U> const & v1,
                 PositionVector2D<CoordSystem2,U> p2)        {
         return p2 += v1;
      }

      /**
         Subtraction of a DisplacementVector2D from a PositionVector2D.
         The return type is a PositionVector2D,
         of the same (coordinate system) type as the input PositionVector2D.
      */
      template <class CoordSystem1, class CoordSystem2, class U>
      inline
      PositionVector2D<CoordSystem2,U>
      operator-( PositionVector2D<CoordSystem2,U> p1,
                 DisplacementVector2D<CoordSystem1,U> const & v2)        {
         return p1 -= v2;
      }

      // Scaling of a position vector with a real number is not physically meaningful

      // ------------- I/O to/from streams -------------

      template< class char_t, class traits_t, class T, class U >
      inline
      std::basic_ostream<char_t,traits_t> &
      operator << ( std::basic_ostream<char_t,traits_t> & os
                    , PositionVector2D<T,U> const & v
         )
      {
         if( !os )  return os;

         typename T::Scalar a, b;
         v.GetCoordinates(a, b);

         if( detail::get_manip( os, detail::bitforbit ) )  {
            detail::set_manip( os, detail::bitforbit, '\00' );
            typedef GenVector_detail::BitReproducible BR;
            BR::Output(os, a);
            BR::Output(os, b);
         }
         else  {
            os << detail::get_manip( os, detail::open  ) << a
               << detail::get_manip( os, detail::sep   ) << b
               << detail::get_manip( os, detail::close );
         }

         return os;

      }  // op<< <>()


      template< class char_t, class traits_t, class T, class U >
      inline
      std::basic_istream<char_t,traits_t> &
      operator >> ( std::basic_istream<char_t,traits_t> & is
                    , PositionVector2D<T,U> & v
         )
      {
         if( !is )  return is;

         typename T::Scalar a, b;

         if( detail::get_manip( is, detail::bitforbit ) )  {
            detail::set_manip( is, detail::bitforbit, '\00' );
            typedef GenVector_detail::BitReproducible BR;
            BR::Input(is, a);
            BR::Input(is, b);
         }
         else  {
            detail::require_delim( is, detail::open  );  is >> a;
            detail::require_delim( is, detail::sep   );  is >> b;
            detail::require_delim( is, detail::close );
         }

         if( is )
            v.SetCoordinates(a, b);
         return is;

      }  // op>> <>()




   } // namespace Math

} // namespace ROOT


#endif /* ROOT_Math_GenVector_PositionVector2D  */
