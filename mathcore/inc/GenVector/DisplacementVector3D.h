// @(#)root/mathcore:$Name:  $:$Id: DisplacementVector3D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class DisplacementVector3D
// 
// Created by: Lorenzo Moneta  at Mon May 30 12:21:43 2005
// 
// Last update: mf Tue Jun 21 2005
// 
#ifndef ROOT_MATH_DISPLACEMENTVECTOR3D
#define ROOT_MATH_DISPLACEMENTVECTOR3D 1

#ifndef ROOT_MATH_CARTESIAN3D
#include "GenVector/Cartesian3D.h"
#endif
#ifndef ROOT_MATH_POLAR3D
#include "GenVector/Polar3D.h"
#endif
#ifndef ROOT_MATH_CYLINDRICALETA3D
#include "GenVector/CylindricalEta3D.h"
#endif

#include "GenVector/PositionVector3Dfwd.h"
#include "GenVector/GenVectorIO.h"


// use now LA conversion methods (To DO improve to use iterator with begin-end )
#define LATER 1 
#ifdef LATER
#include <stdlib.h>    // for size_t
#endif




namespace ROOT { 

  namespace Math { 


    /**       
	      Class describing a generic displacement vector in 3 dimensions.
	      This class is templated on the type of Coordinate system. 
	      One example is the XYZVector which is a vector based on 
	      double precision x,y,z data members by using the 
	      Cartesian3D<double> Coordinate system.     
    */

    template <class CoordSystem>
    class DisplacementVector3D {

    public: 

      typedef typename CoordSystem::Scalar Scalar;
      typedef CoordSystem CoordinateType;

      // ------ ctors ------

      /** 
	  Default constructor. Construct an empty object with zero values
      */ 
      DisplacementVector3D () :   fCoordinates()  { }  

      /**
	 Construct from three values of type <em>Scalar</em>.
	 In the case of a XYZVector the values are x,y,z 
	 In the case of  a polar vector they are r,theta, phi
      */   
      DisplacementVector3D(Scalar a, Scalar b, Scalar c) : 
	fCoordinates ( a , b,  c )  { }

     /** 
	  Construct from a displacement vector expressed in different
	  coordinates, or using a different Scalar type
      */
      template <class T>
      explicit DisplacementVector3D( const DisplacementVector3D<T> & v) : 
	fCoordinates ( v.Coordinates() ) { }

     /** 
	  Construct from an arbitrary position vector 
      */
      template <class T>
      explicit DisplacementVector3D( const PositionVector3D<T> & p) : 
	fCoordinates ( p.Coordinates() ) { }

      /** 
	  Construct from a foreign 3D vector type, for example, Hep3Vector 
	  Precondition: v must implement methods x(), y() and z()
      */
      template <class ForeignVector>
      explicit DisplacementVector3D( const ForeignVector & v) : 
	fCoordinates ( Cartesian3D<Scalar>( v.x(), v.y(), v.z() ) ) { }

     
// #ifdef LATER
      /**
	 construct from a generic linear algebra  vector of at least size 3 
	 implementing operator []. 
	 \par v  LAVector
	 \par index0   index where coordinates starts (typically zero)
	 It works for all Coordinates types, 
	 ( x= v[index0] for Cartesian and r=v[index0] for Polar ) 
      */ 
      template <class LAVector> 
      DisplacementVector3D(const LAVector & v, size_t index0 ) { 
	fCoordinates = CoordSystem ( v[index0], v[index0+1], v[index0+2] ); 
      }
// #endif
      
      // compiler-generated copy ctor and dtor are fine.
 
      // ------ assignment ------

      /** 
	  Assignment operator from a displacement vector of arbitrary type  
      */
      template <class OtherCoords>
      DisplacementVector3D & operator= 
      			( const DisplacementVector3D<OtherCoords> & v) {       
	fCoordinates = v.Coordinates();
	return *this;
      }

      // Assignment operator from a position vector is inappropriate.  

      /** 
	  Assignment from a foreign 3D vector type, for example, Hep3Vector 
	  Precondition: v must implement methods x(), y() and z()
      */
      template <class ForeignVector>
      DisplacementVector3D & operator= ( const ForeignVector & v) {       
	SetXYZ( v.x(),  v.y(), v.z() );  
	return *this; 
      }


#ifdef LATER
      /**
	 assign from a generic linear algebra  vector of at least size 3 
	 implementing operator []. This could be also a C array  
	 \par v  LAVector
	 \par index0   index where coordinates starts (typically zero)
	 It works for all Coordinates types, 
	 ( x= v[index0] for Cartesian and r=v[index0] for Polar ) 
      */ 
      template <class LAVector> 
      DisplacementVector3D & AssignFrom(const LAVector & v, size_t index0 = 0) { 
	fCoordinates = CoordSystem  ( v[index0], v[index0+1], v[index0+2] );
	return *this;
      }
#endif

      // ------ Set, Get, and access coordinate data ------
      
      /** 
	  Retrieve a copy of the coordinates object
      */
      const CoordSystem Coordinates() const {
	return fCoordinates;
      }     

#ifndef __CINT__
      //    disable these in CINT otherwise they will be instantiated also for 
      //    non Cartesian based vectors where thesemethods are not defined
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
     
      // cross and dot products 
      /** 
	  Return the scalar (dot) product of two vectors.
	  It is possible to perform the product for any classes 
	  implementing X(), Y() and Z() member functions 
      */   
      template< class OtherVector >
      Scalar Dot( const  OtherVector & v) const { 
	return X()*v.X() + Y()*v.Y() + Z()*v.Z();
      }
      

// NOTE TO SELF:
//
// DOT AND CROSS SHOULD PERHAPS BE FREE METHODS? NO, WE LIKE a.cross(b) syntax
// 
// Data precision is moot -- if you use higher, actual precision is that of 
// lower type anyway.
//
// CROSS SHOULD ALWAYS RETURN CARTESIAN VECTOR (which can then be converted)


//#if 0
      /**
	 Return vector (cross) product of two vectors, as a vector in
	 the coordinate system of the first. 
	 It is possible to perform the cross product for any classes 
	 implementing x(), y() and z() member functions 
      */ 
      template <class OtherVector>
      DisplacementVector3D Cross( const  OtherVector  & v) const {
	return DisplacementVector3D(  Y() * v.Z() - v.Y() * Z(), 
				      Z() * v.X() - v.Z() * X(), 
				      X() * v.Y() - v.X() * Y() );
      }
     
      /**
	 return Unit vector parallel to this
      */
      DisplacementVector3D Unit() const { 
	Scalar tot = R();
	return tot == 0 ? *this : DisplacementVector3D(*this) / tot;
      }
      
      /** 
	  Self Addition with a displacement vector. 
      */
#ifndef __CINT__
      template <class OtherCoords>    
      DisplacementVector3D & operator+= (const  DisplacementVector3D<OtherCoords> & v) { 
	SetXYZ(  X() + v.X(), Y() + v.Y(), Z() + v.Z() ); 
	return *this;
      }
#else 
      template<class V>
      DisplacementVector3D & operator+= (const  V & v) { 
	SetXYZ(  X() + v.X(), Y() + v.Y(), Z() + v.Z() ); 
	return *this;
      }
#endif

      /** 
	  Self Difference with a displacement vector. 
      */
#ifndef __CINT__
      template <class OtherCoords>    
      DisplacementVector3D & operator-= (const  DisplacementVector3D<OtherCoords> & v) { 
	SetXYZ(  X() - v.X(), Y() - v.Y(), Z() - v.Z() ); 
	return *this;
      }
#else 
      template<class V>
      DisplacementVector3D & operator-= (const  V & v) { 
	SetXYZ(  X() - v.X(), Y() - v.Y(), Z() - v.Z() ); 
	return *this;
      }
#endif

      /**
	 multiply this vector by a scalar quantity
      */
      DisplacementVector3D & operator*= (const Scalar & a) { 
	fCoordinates.Scale(a); 
	return *this; 
      }
      
      /**
	 divide this vector by a scalar quantity
      */
      DisplacementVector3D & operator/= (const Scalar & a) { 
	Scalar tmp(a);
	//if (tmp==0) tmp = 1.0 / std::numeric_limits<Scalar>::max();
	fCoordinates.Scale(1/tmp); 
	return *this; 
      }

      // specialized rotation functions (here or in  a separate class ? )
      // have specialization according to Coordinates (like optimized RotateZ ? )

      /** 
	  Rotate along X by an Angle.
	  Implement here using X(), Y(), Z(), in principle could be implemented 
	  separatly for every system 
      */ 
      DisplacementVector3D & RotateX(const Scalar & alpha) { 
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
      DisplacementVector3D & RotateY(const Scalar & alpha) { 
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
      DisplacementVector3D & RotateZ(const Scalar & alpha) { 
	double sina = sin(alpha);
	double cosa = cos(alpha);
	double x2 = X() * cosa - Y() * sina; 
	double y2 = Y() * cosa - X() * sina;
	SetXYZ(x2, y2, Z());
	return *this; 
      }
      

      /** 
	  apply a transformation (in this case only rotation). 
	  A transformation for a DisplacementVector is only a rotation since 
	  they do not translate. 
	  The transformation needs to implement the operator * accepting this type of Vectors
      */ 
      template<class ArbitraryRotation> 
      DisplacementVector3D & Transform(const ArbitraryRotation & R) {
	return *this = R * (*this);
      }

      /**                                                                                                                                                           
	Scaling of a vector with a real number
      */
      DisplacementVector3D  operator * ( const Scalar & a) {
        DisplacementVector3D tmp(*this);
        return tmp*=a;
      }
      /**
	 Division of a vector with a real number 
       */
      DisplacementVector3D operator / (const Scalar & a) {
	DisplacementVector3D tmp(*this);
	tmp /= a;
	return tmp;
      }
//#endif // 0

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
	  Assignment operator from a position vector is inappropriate  
      */
      template <class OtherCoords>
      DisplacementVector3D & operator= 
      ( const PositionVector3D<OtherCoords> & v) { return *this;} 

    };  
    
    // ---------- DisplacementVector3D class template ends here ------------






   /** 
	Addition of DisplacementVector3D vectors.
	The (coordinate system) type of the returned vector is defined to 
	be identical to that of the first vector which is passed by value
    */
#ifndef __CINT__
    template <class CoordSystem1, class CoordSystem2>
    DisplacementVector3D<CoordSystem1>     
    operator+( DisplacementVector3D<CoordSystem1> v1, const DisplacementVector3D<CoordSystem2>  & v2) {
      return v1 += v2;
    }
#else
    template <class V1, class V2>
    V1 operator+( V1 v1, const V2  & v2) {
      return v1 += v2;
    }
#endif
    
    /** 
	Difference between two DisplacementVector3D vectors.
	The (coordinate system) type of the returned vector is defined to 
	be identical to that of the first vector. 
    */
#ifndef __CINT__
    template <class CoordSystem1, class CoordSystem2>
    DisplacementVector3D<CoordSystem1> 
    operator-( DisplacementVector3D<CoordSystem1> v1, 
	       DisplacementVector3D<CoordSystem2> const & v2) {
      return v1 -= v2;
    }
#else
    template <class V1, class V2>
    V1 operator-( V1 v1, const V2  & v2) {
      return v1 -= v2;
    }
#endif

    /**
       Scaling of a displacement vector with a real number v = a*v
    */
    template <class CoordSystem> 
    DisplacementVector3D<CoordSystem> 
    operator * ( const typename DisplacementVector3D<CoordSystem>::Scalar & a, DisplacementVector3D<CoordSystem> v) { 
      return v *= a; 
    }
    // move this in the class to solve problem on AIX
//     template <class CoordSystem> 
//     DisplacementVector3D<CoordSystem> 
//     operator * ( DisplacementVector3D<CoordSystem> v, const typename DisplacementVector3D<CoordSystem>::Scalar & a) { 
//       return v *= a; 
//     }


    // v1*v2 notation for Cross product of two vectors is omitted,
    // since it is always confusing as to whether dot product is meant.



    // ------------- I/O to streams -------------

    template< class char_t, class traits_t, class T >
      inline
      std::basic_ostream<char_t,traits_t> & 
      operator << ( std::basic_ostream<char_t,traits_t> & os
        	  , DisplacementVector3D<T> const & v
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
        	  , DisplacementVector3D<T> & v
        	  )
    {
      if( !is )  return is;

      typename T::Scalar a, b, c;

      detail::require_delim( is, detail::open ); is >> a;
      detail::require_delim( is, detail::sep );	 is >> b;
      detail::require_delim( is, detail::sep );	 is >> c;
      detail::require_delim( is, detail::close );
      v.SetCoordinates(a, b, c);

      return is;

    }  // op>> <>()

  } // end namespace Math
  
} // end namespace ROOT


#endif /* ROOT_MATH_DISPLACEMENTVECTOR3D */

