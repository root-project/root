// @(#)root/mathcore:$Name:  $:$Id: LorentzVector.h,v 1.3 2005/06/28 05:00:39 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class LorentzVector
// 
// Created by: moneta  at Tue May 31 17:06:09 2005
// 
// Last update: Jun 24 2005
// 
#ifndef ROOT_Math_LorentzVector 
#define ROOT_Math_LorentzVector 1


#include "MathCore/Cartesian4D.h"
#include "MathCore/Vector3Dfwd.h"
#ifdef LATER
// needed for definition of size_t
#include <stdlib.h>
#endif

#include "MathCore/GenVectorIO.h"

namespace ROOT { 

  namespace Math { 




    /** 
	Class describing a generic LorentzVector in the 4D space-time. 
	The metric used for the LorentzVector is (-,-,-,+).
	In the case of LorentzVector we don't distinguish the concept of points and vectors as in the 3D case, 
	since the main use case for 4D Vectors is to describe the kinematics of relativistic particles. A LorentzVector behaves like a DisplacementVector in 4D.       

	@ingroup GenVector
    */

    template <class CoordSystem > 
    class LorentzVector { 

    public: 

      typedef typename CoordSystem::Scalar Scalar;
      typedef CoordSystem CoordinateType;

      /**
	 default constructor of an empty vector (Px = Y = Z = E = 0 ) 
      */
      
      LorentzVector() {}

      /**
	 generic constructors from four scalar values. 
	 The association between values and coordinate depends on the coordinate system
	 \param a scalar value (Px for Cartesian or  Rho for cylindrical coordinate system)  
	 \param b scalar value (Y for Cartesian or  Eta for cylindrical coordinate system)  
	 \param c scalar value (Z for Cartesian or  Phi for cylindrical coordinate system)  
	 \param d scalar value (E for Cartesian  or  M  for  PxYPZM coordinate system)  
      */  
      LorentzVector(const Scalar & a, const Scalar & b, const Scalar & c, const Scalar & d) : 
	fCoordinates(a , b,  c, d)  {}


      /**
	 constructor from a LorentzVector expressed in different
	 coordinates, or using a different Scalar type
      */
      template <class OtherType>
      explicit LorentzVector(const LorentzVector<OtherType> & v ) : 
	fCoordinates( v.Coordinates() ) {}

      /**
	 constructor from any other Lorentz vector like Hep3Vector
	 Precondition: v must implement x(), y(), z() and t() 
      */
      template<class ForeignLorentzVector>
      explicit LorentzVector( const ForeignLorentzVector & v) : 
	fCoordinates(Cartesian4D<Scalar>(v.x(), v.y(), v.z(), v.t()  ) ) 
      {}

#ifdef LATER
      /**
	 construct from a generic linear algebra  vector implementing operator []
	 and with a size of at least 4. This could be also a C array 
	 In this case v[0] is the first data member 
	 ( Px for a Cartesian4D base) 
       \param v LA vector
       \param index0 index of first vector element (Px)
      */ 
      template <class LAVector> 
      explicit LorentzVector(const LAVector & v, size_t index0 ) { 
	fCoordinates = CoordSystem ( v[index0], v[index0+1], v[index0+2], v[index0+3] ); 
      }
#endif
      
      // compiler-generated copy ctor and dtor are fine.
      
      // ------ assignment ------
      
      /**
	 Assignment operator from a displacement vector of arbitrary type
      */
      template <class OtherCoords>
      LorentzVector & operator=
      ( const LorentzVector<OtherCoords> & v) {
	fCoordinates = v.Coordinates();
	return *this;
      }
      
      /**
	 assignment from any other Lorentz vector  implementing 
	 x(), y(), z() and t()
      */
      template<class ForeignLorentzVector>
      LorentzVector & operator = ( const ForeignLorentzVector & v) { 
	SetXYZT(v.x(), v.y(), v.z(), v.t() ); 
	return *this; 
      }

#ifdef LATER
      /**
	 assign from a generic linear algebra  vector implementing operator []
	 and with a size of at least 4
	 In this case v[0] is the first data member 
	 ( Px for a Cartesian4D base) 
	 \param v LA vector
	 \param index0 index of first vector element (Px)
      */ 
      template <class LAVector> 
      LorentzVector & AssignFrom(const LAVector & v, size_t index0=0 ) { 
	fCoordinates.SetCoordinates( v[index0], v[index0+1], v[index0+2], v[index0+3] ); 
	return *this;
      }
#endif
  
      // ------ Set, Get, and access coordinate data ------
      
      /**
	 Retrieve a copy of of the coordinates system object
      */
      CoordSystem Coordinates() const {
        return fCoordinates;
      }

      /**
	 Retrieve a non const reference to  the coordinates object
      */
      CoordSystem & Coordinates()  {
        return fCoordinates;
      }

      /**
         Set internal data based on 4 Scalar numbers
       */
      void SetCoordinates( Scalar a, Scalar b, Scalar c, Scalar d )
                            { fCoordinates.SetCoordinates(a, b, c, d);  }

      /**
         Set internal data based on 4 Scalars at *begin to *end
       */
      template <class IT>
      void SetCoordinates( IT begin, IT end  ) {  	
	assert( begin != end && begin+1 != end && begin+2 != end && begin+3 != end);
	fCoordinates.SetCoordinates(*begin, *(begin+1), *(begin+2), *(begin+3)); 
      }

      /**
         Set internal data based on an array of 4 Scalar numbers
       */
      void SetCoordinates( const Scalar * src )
                            { fCoordinates.SetCoordinates(src);  }

      /**
         Set internal data into 4 Scalar numbers
       */
      void GetCoordinates( Scalar& a, Scalar& b, Scalar& c, Scalar & d ) const
                            { fCoordinates.GetCoordinates(a, b, c, d);  }

      /**
         get internal data into 4 Scalars at *begin to *end
       */
      template <class IT>
      void GetCoordinates( IT begin, IT end ) const
                            { fCoordinates.GetCoordinates(&(*begin)); }

      /**
         get internal data into an array of 4 Scalar numbers
       */
      void GetCoordinates( Scalar * dest ) const
                            { fCoordinates.GetCoordinates(dest);  }

      /**
         set the values of the vector from the cartesian components (x,y,z,t)
         (if the vector is held in another coordinates, like cylindrical eta,
         then (x, y, z, t) are converted to that form)
      */
      void SetXYZT (Scalar x, Scalar y, Scalar z, Scalar t) {
            fCoordinates =  Cartesian4D<Scalar> (x,y,z,t);
      }


      // individual coordinate accessors in various coordinate systems

      /**
	 spatial X component 
      */ 
      Scalar X() const { return fCoordinates.X(); } 
      Scalar Px() const { return fCoordinates.X();}
      /** 
	  spatial Y component
      */ 
      Scalar Y() const { return fCoordinates.Y(); } 
      Scalar Py() const { return fCoordinates.Y();}
      /** 
	  spatial Z component
      */ 
      Scalar Z() const { return fCoordinates.Z(); } 
      Scalar Pz() const { return fCoordinates.Z();}
      /** 
	  return 4-th component (time or energy for a 4-momentum vector) 
      */ 
      Scalar T() const { return fCoordinates.T(); } 
      Scalar E() const { return fCoordinates.T(); }


      /**
	 return magnitude (mass ) square  M2 = T**2 - X**2 - Y**2 - Z**2 (we use -,-,-,+ metric)
      */ 
      Scalar M2() const { return fCoordinates.M2();}

      /**
	 return magnitude (mass ) using the  (-,-,-,+)  metric. 
	 If Mag2 is negative (space-like vector) M = - sqrt( -M2) and is negative
      */ 
      Scalar M() const    { return fCoordinates.M();}

      /**
	 return the spatial (3D) magnitude ( sqrt(X**2 + Y**2 + Z**2) )
      */ 
      Scalar R() const { return fCoordinates.R(); } 
      Scalar P() const { return fCoordinates.R(); } 

      /**
	 return the square of the transverse spatial component ( X**2 + Y**2 )
      */ 
      Scalar Perp2() const { return fCoordinates.Perp2();}

      /**
	 return the  transverse spatial component sqrt ( X**2 + Y**2 )
      */ 
      Scalar Rho() const { return fCoordinates.Rho();}
      Scalar Pt() const { return fCoordinates.Rho();}

      /** 
	  transverse mass squared
	  \f[ Mt2 = E^2 - p{_z}^2 \f]
      */
      Scalar Mt2() const { return fCoordinates.Mt2(); } 

      /**
	 transverse mass
	 \f[ \sqrt{ Mt2 = E^2 - p{_z}^2} X sign(E^ - p{_z}^2) \f]
      */
      Scalar Mt() const { return fCoordinates.Mt(); } 

      /**
	 transverse energy squared
	 \f[ et = \frac{E^2 p_{\perp}^2 }{ |p|^2 }
      */
      Scalar Et2() const { return fCoordinates.Et2(); } 
      
      /**
	 transverse energy 
	 \f[ et = \sqrt{ \frac{E^2 p_{\perp}^2 }{ |p|^2 } } X sign(E)
      */
      Scalar Et() const { return fCoordinates.Et(); } 

      /**
	 aximuthal  Angle
      */
      Scalar Phi() const  { return fCoordinates.Phi();}
    
      /**
	 polar Angle
      */ 
      Scalar Theta() const { return fCoordinates.Theta(); } 

      /**
	 pseudorapidity
	 \f[ \eta = - \ln { \tan { \frac { \theta} {2} } } \f]
      */
      Scalar Eta() const { return fCoordinates.Eta(); }  


      /** 
	  get the spatial components of the Vector in a 
	  DisplacementVector based on Cartesian Coordinates
      */ 
      ::ROOT::Math::DisplacementVector3D<Cartesian3D<Scalar> > Vec() { 
	return ::ROOT::Math::DisplacementVector3D<Cartesian3D<Scalar> >( X(), Y(), Z() ); 
      }



      // ------ Operations combining two Lorentz vectors ------
      // use in the template method E instead of T for problem with T definition

 
    /** 
	scalar (Dot) product of two LorentzVector vectors (metric is -,-,-,+)
	Enable the product using any other LorentzVector implementing the X(), Y() , Z() and t() member functions
	\param  q  any LorentzVector implementing the X(), Y() , Z() and t() member functions
	\return the result of v.q of type according to the base scalar type of v
     */ 
    
      template <class OtherLorentzVector>
      Scalar Dot(const OtherLorentzVector & q) const { 
	return E()*q.E() - X()*q.X() - Y()*q.Y() - Z()*q.Z();
      } 


      /**
	 add another Vector to itself ( v+= q )
	 Enable the addition with any other LorentzVector implementing the X(), Y() , Z() and E() member functions
	 \param q  any LorentzVector implementing the X(), Y() , Z() and E() member functions
      */
      template <class OtherLorentzVector>
      LorentzVector & operator += ( const OtherLorentzVector & q) { 
	SetXYZT( X() + q.X(), Y() + q.Y(), Z() + q.Z(), E() + q.E()  ); 
	return *this; 
      }


      /**
	 addition between LorentzVectors (v3 = v1 + v2) 
	 Enable the addition with any other LorentzVector implementing the X(), Y() , Z() and E() member functions
	 \param v2   any LorentzVector implementing the X(), Y() , Z() and E() member functions
	 \return a new LorentzVector of the same type of v1  
      */
      template<class OtherLorentzVector>
      LorentzVector  operator +  ( const OtherLorentzVector & v2) { 
	return LorentzVector(Cartesian4D<Scalar>( X() + v2.X(), Y() + v2.Y(), Z() + v2.Z(), E() + v2.E() ) );  
      }


      /**
	 subtract another Vector to itself ( v-= q )
	 Enable the subtraction with any other LorentzVector implementing the X(), Y() , Z() and E() member functions
	 \param q  any LorentzVector implementing the X(), Y() , Z() and E() member functions
      */
      template <class OtherLorentzVector>
      LorentzVector & operator -= ( const OtherLorentzVector & q) { 
	SetXYZT( X() - q.X(), Y() - q.Y(), Z() - q.Z(), E() - q.E()  ); 
	return *this; 
      }


      /**
	 subtraction between LorentzVectors (v3 = v1 - v2) 
	 Enable the subtraction with any other LorentzVector implementing the X(), Y() , Z() and E() member functions
	 \param v2   any LorentzVector implementing the X(), Y() , Z() and E() member functions
	 \return a new LorentzVector of the same type of v1  
      */
      template<class OtherLorentzVector>
      LorentzVector  operator -  ( const OtherLorentzVector & v2) { 
	return LorentzVector(Cartesian4D<Scalar>( X() - v2.X(), Y() - v2.Y(), Z() - v2.Z(), E() - v2.E() ) );  
      }


      /**
	 return unary minus ( q = - v )
	 \return a new LorentzVector with opposite direction and time
      */ 
      LorentzVector operator - () { 
	return LorentzVector(Cartesian4D<Scalar>( - X(), - Y(), - Z(), - E()  ) );  
      }

      //--- scaling operations ------

      /**
	 multiplication by a scalar quantity v *= a
      */ 
      LorentzVector & operator *= ( const Scalar & a) { 
	fCoordinates.Scale(a);
	return *this;
      }

      /**
	 division by a scalar quantity v /= a
      */ 
      LorentzVector & operator /= ( const Scalar & a) { 
	Scalar tmp(a);
	fCoordinates.Scale(1/tmp);
	return *this;
      }
      
      

      /**
	 Scale of a LorentzVector with a scalar quantity a
	 \param v  mathcore::LorentzVector based on any coordinate system  
	 \param a  scalar quantity of typpe a
	 \return a new mathcoreLorentzVector q = v * a same type as v
      */ 
      LorentzVector operator * ( const Scalar & a) { 
	LorentzVector tmp(*this);
	tmp *= a; 
	return tmp;
      }

      /**
	 Divide a LorentzVector with a scalar quantity a
	 \param v  mathcore::LorentzVector based on any coordinate system  
	 \param a  scalar quantity of typpe a
	 \return a new mathcoreLorentzVector q = v / a same type as v
      */ 
      LorentzVector<CoordSystem> operator / ( const Scalar & a) { 
	LorentzVector<CoordSystem> tmp(*this);
	tmp /= a; 
	return tmp;
      }


      // Limited backward name compatibility with CLHEP

      Scalar x()     const { return X();     }
      Scalar y()     const { return Y();     }
      Scalar z()     const { return Z();     }
      Scalar t()     const { return E();     }
      Scalar px()    const { return X();     }
      Scalar py()    const { return Y();     }
      Scalar pz()    const { return Z();     }
      Scalar e()     const { return E();     }
      Scalar r()     const { return R();     }
      Scalar theta() const { return Theta(); }
      Scalar phi()   const { return Phi();   }
      Scalar eta()   const { return Eta();   }
      Scalar rho()   const { return Rho();   }
      Scalar perp2() const { return Perp2(); }
      Scalar mag2()  const { return M2();    }
      Scalar mag()   const { return M();     }

  private: 

    CoordSystem  fCoordinates; 


  }; 



  // global nethods 

  /**
     Scale of a LorentzVector with a scalar quantity a
     \param a  scalar quantity of typpe a
     \param v  mathcore::LorentzVector based on any coordinate system  
     \return a new mathcoreLorentzVector q = v * a same type as v
   */ 
    template <class CoordSystem>
    LorentzVector<CoordSystem> operator * ( const typename  LorentzVector<CoordSystem> ::Scalar & a, const LorentzVector<CoordSystem>& v) {  
      LorentzVector<CoordSystem> tmp(v);
      tmp *= a; 
      return tmp;
    }



    // ------------- I/O to/from streams -------------

    template< class char_t, class traits_t, class CoordSystem >
      inline
      std::basic_ostream<char_t,traits_t> &
      operator << ( std::basic_ostream<char_t,traits_t> & os
                  , LorentzVector<CoordSystem> const & v
                  )
    {
      if( !os )  return os;

      typename CoordSystem::Scalar a, b, c, d;
      v.GetCoordinates(a, b, c, d);

      if( detail::get_manip( os, detail::bitforbit ) )  {
        detail::set_manip( os, detail::bitforbit, '\00' );
        // TODO: call MF's bitwise-accurate functions on each of a, b, c, d
      }
      else  {
        os << detail::get_manip( os, detail::open  ) << a
           << detail::get_manip( os, detail::sep   ) << b
           << detail::get_manip( os, detail::sep   ) << c
           << detail::get_manip( os, detail::sep   ) << d
           << detail::get_manip( os, detail::close );
      }

      return os;

    }  // op<< <>()


    template< class char_t, class traits_t, class CoordSystem >
      inline
      std::basic_istream<char_t,traits_t> &
      operator >> ( std::basic_istream<char_t,traits_t> & is
                  , LorentzVector<CoordSystem> & v
                  )
    {
      if( !is )  return is;

      typename CoordSystem::Scalar a, b, c, d;

      if( detail::get_manip( is, detail::bitforbit ) )  {
        detail::set_manip( is, detail::bitforbit, '\00' );
        // TODO: call MF's bitwise-accurate functions on each of a, b, c
      }
      else  {
        detail::require_delim( is, detail::open  );  is >> a;
        detail::require_delim( is, detail::sep   );  is >> b;
        detail::require_delim( is, detail::sep   );  is >> c;
        detail::require_delim( is, detail::sep   );  is >> d;
        detail::require_delim( is, detail::close );
      }

      if( is )
        v.SetCoordinates(a, b, c, d);
      return is;

    }  // op>> <>()


  } // end namespace Math

} // end namespace ROOT


#endif




