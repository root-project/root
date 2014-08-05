// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team and                     *
  *                      FNAL LCG ROOT MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header source file for class DisplacementVector2D
//
// Created by: Lorenzo Moneta  at Mon Apr 16 2007
//

#ifndef ROOT_Math_GenVector_DisplacementVector2D
#define ROOT_Math_GenVector_DisplacementVector2D  1

#ifndef ROOT_Math_GenVector_Cartesian2D
#include "Math/GenVector/Cartesian2D.h"
#endif

#ifndef ROOT_Math_GenVector_PositionVector2Dfwd
#include "Math/GenVector/PositionVector2Dfwd.h"
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

//#include "Math/GenVector/Expression2D.h"




namespace ROOT {

  namespace Math {



//__________________________________________________________________________________________
     /**
        Class describing a generic displacement vector in 2 dimensions.
        This class is templated on the type of Coordinate system.
        One example is the XYVector which is a vector based on
        double precision x,y  data members by using the
        ROOT::Math::Cartesian2D<double> Coordinate system.
        The class is having also an extra template parameter, the coordinate system tag,
        to be able to identify (tag) vector described in different reference coordinate system,
        like global or local coordinate systems.

        @ingroup GenVector
     */

     template <class CoordSystem, class Tag = DefaultCoordinateSystemTag >
     class DisplacementVector2D {

     public:

        typedef typename CoordSystem::Scalar Scalar;
        typedef CoordSystem CoordinateType;
        typedef Tag  CoordinateSystemTag;

        // ------ ctors ------

        /**
           Default constructor. Construct an empty object with zero values
        */
        DisplacementVector2D ( ) :   fCoordinates()  { }


        /**
           Construct from three values of type <em>Scalar</em>.
           In the case of a XYVector the values are x,y
           In the case of  a polar vector they are r, phi
        */
        DisplacementVector2D(Scalar a, Scalar b) :
           fCoordinates ( a , b )  { }

        /**
           Construct from a displacement vector expressed in different
           coordinates, or using a different Scalar type, but with same coordinate system tag
        */
        template <class OtherCoords>
        explicit DisplacementVector2D( const DisplacementVector2D<OtherCoords, Tag> & v) :
           fCoordinates ( v.Coordinates() ) { }


        /**
           Construct from a position vector expressed in different coordinates
           but with the same coordinate system tag
        */
        template <class OtherCoords>
        explicit DisplacementVector2D( const PositionVector2D<OtherCoords,Tag> & p) :
           fCoordinates ( p.Coordinates() ) { }


        /**
           Construct from a foreign 2D vector type, for example, Hep2Vector
           Precondition: v must implement methods x() and  y()
        */
        template <class ForeignVector>
        explicit DisplacementVector2D( const ForeignVector & v) :
           fCoordinates ( Cartesian2D<Scalar>( v.x(), v.y() ) ) { }



        // compiler-generated copy ctor and dtor are fine.

        // ------ assignment ------

        /**
           Assignment operator from a displacement vector of arbitrary type
        */
        template <class OtherCoords>
        DisplacementVector2D & operator=
        ( const DisplacementVector2D<OtherCoords, Tag> & v) {
           fCoordinates = v.Coordinates();
           return *this;
        }

        /**
           Assignment operator from a position vector
           (not necessarily efficient unless one or the other is Cartesian)
        */
        template <class OtherCoords>
        DisplacementVector2D & operator=
        ( const PositionVector2D<OtherCoords,Tag> & rhs) {
           SetXY(rhs.x(), rhs.y() );
           return *this;
        }


        /**
           Assignment from a foreign 2D vector type, for example, Hep2Vector
           Precondition: v must implement methods x() and  y()
        */
        template <class ForeignVector>
        DisplacementVector2D & operator= ( const ForeignVector & v) {
           SetXY( v.x(),  v.y()  );
           return *this;
        }


        // ------ Set, Get, and access coordinate data ------

        /**
           Retrieve a copy of the coordinates object
        */
        CoordSystem Coordinates() const {
           return fCoordinates;
        }

        /**
           Set internal data based on 2 Scalar numbers.
           These are for example (x,y) for a cartesian vector or (r,phi) for a polar vector
       */
        DisplacementVector2D<CoordSystem, Tag>& SetCoordinates( Scalar a, Scalar b) {
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
        DisplacementVector2D<CoordSystem, Tag>& SetXY (Scalar a, Scalar b) {
           fCoordinates.SetXY(a,b);
           return *this;
        }

        // ------------------- Equality -----------------

        /**
           Exact equality
        */
        bool operator==(const DisplacementVector2D & rhs) const {
           return fCoordinates==rhs.fCoordinates;
        }
        bool operator!= (const DisplacementVector2D & rhs) const {
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


        // ----- Other fundamental properties -----

        /**
           Magnitute squared ( r^2 in spherical coordinate)
        */
        Scalar Mag2() const { return fCoordinates.Mag2();}


        /**
           return unit vector parallel to this
        */
        DisplacementVector2D  Unit() const {
           Scalar tot = R();
           return tot == 0 ? *this : DisplacementVector2D(*this) / tot;
        }

        // ------ Setting individual elements present in coordinate system ------

        /**
           Change X - Cartesian2D coordinates only
        */
        DisplacementVector2D<CoordSystem, Tag>& SetX (Scalar a) {
           fCoordinates.SetX(a);
           return *this;
        }

        /**
           Change Y - Cartesian2D coordinates only
        */
        DisplacementVector2D<CoordSystem, Tag>& SetY (Scalar a) {
           fCoordinates.SetY(a);
           return *this;
        }


        /**
           Change R - Polar2D coordinates only
        */
        DisplacementVector2D<CoordSystem, Tag>& SetR (Scalar a) {
           fCoordinates.SetR(a);
           return *this;
        }


        /**
           Change Phi - Polar2D  coordinates
        */
        DisplacementVector2D<CoordSystem, Tag>& SetPhi (Scalar ang) {
           fCoordinates.SetPhi(ang);
           return *this;
        }



        // ------ Operations combining two vectors ------
        // -- need to have the specialized version in order to avoid

        /**
           Return the scalar (dot) product of two displacement vectors.
           It is possible to perform the product for any type of vector coordinates,
           but they must have the same coordinate system tag
        */
        template< class OtherCoords >
        Scalar Dot( const  DisplacementVector2D<OtherCoords,Tag>  & v) const {
           return X()*v.X() + Y()*v.Y();
        }
        /**
           Return the scalar (dot) product of two vectors.
           It is possible to perform the product for any classes
           implementing x() and  y()  member functions
        */
        template< class OtherVector >
        Scalar Dot( const  OtherVector & v) const {
           return X()*v.x() + Y()*v.y();
        }



        /**
           Self Addition with a displacement vector.
        */
        template <class OtherCoords>
        DisplacementVector2D & operator+=
        (const  DisplacementVector2D<OtherCoords,Tag> & v) {
           SetXY(  X() + v.X(), Y() + v.Y() );
           return *this;
        }

        /**
           Self Difference with a displacement vector.
        */
        template <class OtherCoords>
        DisplacementVector2D & operator-=
        (const  DisplacementVector2D<OtherCoords,Tag> & v) {
           SetXY(  x() - v.x(), y() - v.y() );
           return *this;
        }


        /**
           multiply this vector by a scalar quantity
        */
        DisplacementVector2D & operator*= (Scalar a) {
           fCoordinates.Scale(a);
           return *this;
        }

        /**
           divide this vector by a scalar quantity
        */
        DisplacementVector2D & operator/= (Scalar a) {
           fCoordinates.Scale(1/a);
           return *this;
        }

        // -- The following methods (v*a and v/a) could instead be free functions.
        // -- They were moved into the class to solve a problem on AIX.

        /**
           Multiply a vector by a real number
        */
        DisplacementVector2D operator * ( Scalar a ) const {
           DisplacementVector2D tmp(*this);
           tmp *= a;
           return tmp;
        }

        /**
           Negative of the vector
        */
        DisplacementVector2D operator - ( ) const {
           return operator*( Scalar(-1) );
        }

        /**
           Positive of the vector, return itself
        */
        DisplacementVector2D operator + ( ) const {return *this;}

        /**
           Division of a vector with a real number
        */
        DisplacementVector2D operator/ (Scalar a) const {
           DisplacementVector2D tmp(*this);
           tmp /= a;
           return tmp;
        }

        /**
           Rotate by an angle
         */
        void Rotate( Scalar angle) {
           return fCoordinates.Rotate(angle);
        }


        // Methods providing  Limited backward name compatibility with CLHEP

        Scalar x()     const { return fCoordinates.X();     }
        Scalar y()     const { return fCoordinates.Y();     }
        Scalar r()     const { return fCoordinates.R();     }
        Scalar phi()   const { return fCoordinates.Phi();   }
        Scalar mag2()  const { return fCoordinates.Mag2();  }
        DisplacementVector2D unit() const {return Unit();}


     private:

        CoordSystem fCoordinates;    // internal coordinate system


        // the following methods should not compile

        // this should not compile (if from a vector or points with different tag
        template <class OtherCoords, class OtherTag>
        explicit DisplacementVector2D( const DisplacementVector2D<OtherCoords, OtherTag> & ) {}

        template <class OtherCoords, class OtherTag>
        explicit DisplacementVector2D( const PositionVector2D<OtherCoords, OtherTag> & ) {}

        template <class OtherCoords, class OtherTag>
        DisplacementVector2D & operator=( const DisplacementVector2D<OtherCoords, OtherTag> & );


        template <class OtherCoords, class OtherTag>
        DisplacementVector2D & operator=( const PositionVector2D<OtherCoords, OtherTag> & );

        template <class OtherCoords, class OtherTag>
        DisplacementVector2D & operator+=(const  DisplacementVector2D<OtherCoords, OtherTag> & );

        template <class OtherCoords, class OtherTag>
        DisplacementVector2D & operator-=(const  DisplacementVector2D<OtherCoords, OtherTag> & );

        template<class OtherCoords, class OtherTag >
        Scalar Dot( const  DisplacementVector2D<OtherCoords, OtherTag> &  ) const;

        template<class OtherCoords, class OtherTag >
        DisplacementVector2D Cross( const  DisplacementVector2D<OtherCoords, OtherTag> &  ) const;


     };

// ---------- DisplacementVector2D class template ends here ------------
// ---------------------------------------------------------------------


     /**
        Addition of DisplacementVector2D vectors.
        The (coordinate system) type of the returned vector is defined to
        be identical to that of the first vector, which is passed by value
     */
     template <class CoordSystem1, class CoordSystem2, class U>
     inline
     DisplacementVector2D<CoordSystem1,U>
     operator+(       DisplacementVector2D<CoordSystem1,U> v1,
                      const DisplacementVector2D<CoordSystem2,U>  & v2) {
        return v1 += v2;
     }

     /**
        Difference between two DisplacementVector2D vectors.
        The (coordinate system) type of the returned vector is defined to
        be identical to that of the first vector.
     */
     template <class CoordSystem1, class CoordSystem2, class U>
     inline
     DisplacementVector2D<CoordSystem1,U>
     operator-( DisplacementVector2D<CoordSystem1,U> v1,
                DisplacementVector2D<CoordSystem2,U> const & v2) {
        return v1 -= v2;
     }





     /**
        Multiplication of a displacement vector by real number  a*v
     */
     template <class CoordSystem, class U>
     inline
     DisplacementVector2D<CoordSystem,U>
     operator * ( typename DisplacementVector2D<CoordSystem,U>::Scalar a,
                  DisplacementVector2D<CoordSystem,U> v) {
        return v *= a;
        // Note - passing v by value and using operator *= may save one
        // copy relative to passing v by const ref and creating a temporary.
     }


     // v1*v2 notation for Cross product of two vectors is omitted,
     // since it is always confusing as to whether dot product is meant.



     // ------------- I/O to/from streams -------------

     template< class char_t, class traits_t, class T, class U >
     inline
     std::basic_ostream<char_t,traits_t> &
     operator << ( std::basic_ostream<char_t,traits_t> & os
                   , DisplacementVector2D<T,U> const & v
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
                   , DisplacementVector2D<T,U> & v
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



  }  // namespace Math

}  // namespace ROOT


#endif /* ROOT_Math_GenVector_DisplacementVector2D  */

