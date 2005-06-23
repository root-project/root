// @(#)root/mathcore:$Name:  $:$Id: PositionVector3D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

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
// Last update: mf Tue Jun 21 2005
// 
#ifndef ROOT_MATH_POSITIONVECTOR3D
#define ROOT_MATH_POSITIONVECTOR3D 1

#ifndef ROOT_MATH_CARTESIAN3D
#include "GenVector/Cartesian3D.h"
#endif
#ifndef ROOT_MATH_POLAR3D
#include "GenVector/Polar3D.h"
#endif
#ifndef ROOT_MATH_CYLINDRICALETA3D
#include "GenVector/CylindricalEta3D.h"
#endif

#include "GenVector/DisplacementVector3Dfwd.h"
#include "GenVector/GenVectorIO.h"

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
	fCoordinates ( a , b,  c)  {}

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

      // Assignment operator from a displacement vector is inappropriate.  
 
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
      PositionVector3D & AssignFrom(const LAVector & v, size_t index0 = 0) { 
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

#ifndef __CINT__      
      /**
	 Set internal data based on 3 Scalar numbers
       */ 
      void SetCoordinates( Scalar a, Scalar b, Scalar c ) 
  			    { fCoordinates.SetCoordinates(a, b, c);  }

      /**
	 Set internal data based on 3 Scalars at *begin to *end
       */ 
      template <class IT> 
      void SetCoordinates( IT begin, IT end ) 
  			    { fCoordinates.SetCoordinates(begin); }

      /**
	 Set internal data based on an array of 3 Scalar numbers
       */ 
      void SetCoordinates( const Scalar * src ) 
  			    { fCoordinates.SetCoordinates(src);  }

      /**
	 Set internal data into 3 Scalar numbers
       */ 
      void GetCoordinates( Scalar& a, Scalar& b, Scalar& c ) const
  			    { fCoordinates.GetCoordinates(a, b, c);  }

      /**
	 get internal data into 3 Scalars at *begin to *end
       */ 
      template <class IT> 
      void GetCoordinates( IT begin, IT end ) const 
  			    { fCoordinates.GetCoordinates(begin); }

      /**
	 get internal data into an array of 3 Scalar numbers
       */ 
      void GetCoordinates( Scalar * dest ) const 
  			    { fCoordinates.GetCoordinates(dest);  }

#endif

      /**
	 set the values of the vector from the cartesian components (x,y,z)
	 (if the vector is held in polar or cylindrical eta coordinates, 
	 then (x, y, z) are converted to that form)
       */ 
      void SetXYZ (Scalar x, Scalar y, Scalar z) {
	    fCoordinates =  Cartesian3D<Scalar> (x,y,z);
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
      
      // Other fundamental properties  
      /** 
	  Magnitute squared ( r^2 in spherical coordinate) 
      */ 
      Scalar Mag2() const { return fCoordinates.Mag2();}
      
      /**
	 Transverse component squared (rho^2 in cylindrical coordinates. 
      */ 
      Scalar Perp2() const { return fCoordinates.Perp2();}
      

      // ------ Operations combining two vectors ------



     
      // Cross and Dot products 
      /** 
	  Return the scalar (Dot) product of this point with a displacement
	  vector (in arbitrary Coordinates).
      */   
#ifndef __CINT__
      template< class CoordSystem2 >
      Scalar Dot( const  DisplacementVector3D<CoordSystem2> & v) const 
#else 
      template< class V > 
      Scalar Dot( const  V & v) const 
#endif
      {
	return X()*v.X() + Y()*v.Y() + Z()*v.Z();
      }

      /**
	 Return vector (Cross) product of this point with a displacement, as a 
	 point vector in this coordinate system of the first. 
      */ 
#ifndef __CINT__
      template <class CoordSystem2>
      PositionVector3D Cross( const DisplacementVector3D<CoordSystem2> & v) const 
#else 
      template <class V>
      PositionVector3D Cross( const V & v) const  
#endif
      {
	return PositionVector3D ( Y() * v.Z() - v.Y() * Z(), 
				  Z() * v.X() - v.Z() * X(), 
				  X() * v.Y() - v.X() * Y() );
      }
    
      // The Dot and Cross products of a pair of point vectors are physically
      // meaningless concepts and thus not implemented.
    
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

      

      // specialized rotation functions (here or in  a separate class ? )
      // have specialization according to Coordinates (like optimized RotateZ ? )

      /** 
	  Rotate along X by an Angle.
	  Implement here using X(), Y(), Z(), in principle could be implemented 
	  separatly for every system 
      */ 
      PositionVector3D & RotateX(const Scalar & alpha) { 
	double sina = sin(alpha);
	double cosa = cos(alpha);
	double y2 = Y() * cosa - Z() * sina;
	double z2 = Z() * cosa + Y() * sina; 
	SetXYZ(X(), y2, z2);
	return *this; 
      }


      /** 
	  Rotate along Y by an Angle apha
	  Implement here using X(), Y(), Z(), in principle could be implemented 
	  separatly for every system 
      */ 
      PositionVector3D & RotateY(const Scalar & alpha) { 
	double sina = sin(alpha);
	double cosa = cos(alpha);
	double x2 = X() * cosa + Z() * sina; 
	double z2 = Z() * cosa - X() * sina;
	SetXYZ(x2, Y(), z2);
	return *this; 
      }
      
      /** 
	  Rotate along Z by an Angle alpha
	  Implement here using X(), Y(), Z(), in principle could be implemented 
	  separatly for every system 
      */ 
      PositionVector3D & RotateZ(const Scalar & alpha) { 
	double sina = sin(alpha);
	double cosa = cos(alpha);
	double x2 = X() * cosa - Y() * sina; 
	double y2 = Y() * cosa - X() * sina;
	SetXYZ(x2, y2, Z());
	return *this; 
      }
      

      /** 
	  apply a transformation (rotation + translation). 
	  A transformation for a PositionVector is only a rotation since 
	  they do not translate. 
	  The transformation needs to implement the operator * accepting this type of Vectors
      */ 
      template<class ArbitraryTransformation> 
      PositionVector3D & Transform(const ArbitraryTransformation & t) {
	return *this = t * (*this);
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
	  Assignment operator from a displacement vector is inappropriate  
      */
      template <class OtherCoords>
      PositionVector3D & operator= 
      			( const DisplacementVector3D<OtherCoords> & v); 

    };

 

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
									       v1.x()-v2.x(), v1.y()-v2.y(),v1.z()-v2.z() ) 
					     );
    }

    /** 
	Addition of a PositionVector3D and a DisplacementVector3D.
	The return type is a PositionVector3D, 
	of the same (coordinate system) type as the input PositionVector3D.
    */
    template <class CoordSystem1, class CoordSystem2>
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
    PositionVector3D<CoordSystem2> 
    operator-( PositionVector3D<CoordSystem2> p1, 
	       DisplacementVector3D<CoordSystem1> const & v2)        {
      return p1 -= v2;
    }
    
    // Scaling of a position vector with a real number is not physically meaningful

    // ------------- I/O to streams -------------

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

      os << detail::get_manip( os, detail::open  ) << a;
      os << detail::get_manip( os, detail::sep   ) << b;
      os << detail::get_manip( os, detail::sep   ) << c;
      os << detail::get_manip( os, detail::close );

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

      detail::require_delim( is, detail::open );   	is >> a;
      detail::require_delim( is, detail::sep );	is >> b;
      detail::require_delim( is, detail::sep );	is >> c;
      detail::require_delim( is, detail::close );
      v.SetCoordinates(a, b, c);

      return is;

    }  // op>> <>()



  } // end namespace Math
  
} // end namespace ROOT


#endif /* ROOT_MATH_POSITIONVECTOR3D */
