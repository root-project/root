// @(#)root/mathcore:$Name:  $:$Id: BasicLorentzVector.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class BasicLorentzVector
// 
// Created by: moneta  at Tue May 31 17:06:09 2005
// 
// Last update: Tue May 31 17:06:09 2005
// 
#ifndef ROOT_MATH_BASICLORENTZVECTOR
#define ROOT_MATH_BASICLORENTZVECTOR 1


#include "GenVector/Cartesian4D.h"
#include "GenVector/DisplacementVector3D.h"
// needed for definition of size_t
#include <stdlib.h>


namespace ROOT { 

  namespace Math { 




    /** 
	Class describing a generic LorentzVector in the 4D space-time. 
	The metric used for the LorentzVector is (-,-,-,+).
	In the case of LorentzVector we don't distinguish the concept of points and vectors as in the 3D case, 
	since the main use case for 4D Vectors is to describe the kinematics of relativistic particles. A LorentzVector behaves like a DisplacementVector in 4D.       
    */

    template <class CoordSystem > 
    class BasicLorentzVector { 

    public: 

      typedef typename CoordSystem::Scalar Scalar;
      typedef CoordSystem CoordinateType;

      /**
	 default constructor of an empty vector (Px = Py = Pz = E = 0 ) 
      */
      
      BasicLorentzVector() {}

      /**
	 generic constructors from four scalar values. 
	 The association between values and coordinate depends on the coordinate system
	 \param a scalar value (Px for Cartesian or  Rho for cylindrical coordinate system)  
	 \param b scalar value (Py for Cartesian or  Eta for cylindrical coordinate system)  
	 \param c scalar value (Pz for Cartesian or  Phi for cylindrical coordinate system)  
	 \param d scalar value (E for Cartesian  or  M  for  PxPyPZM coordinate system)  
      */  
      BasicLorentzVector(const Scalar & a, const Scalar & b, const Scalar & c, const Scalar & d) : 
	fCoordinates(a , b,  c, d)  {}


    /**
       constructor from cartesian coordinate based on arbitrary scalar values 
    */
    template <class T>
    explicit BasicLorentzVector(const Cartesian4D<T> & coord ) : 
      fCoordinates(Cartesian4D<T>(coord)) {}



    /**
       constructor from any other Lorentz vector.  
       Precondition: vector must implment Px(), Py(), Pz() and E()
    */
    template<class OtherLorentzVector>
    explicit BasicLorentzVector( const OtherLorentzVector & v) : 
      fCoordinates(Cartesian4D<Scalar>(v.Px(), v.Py(), v.Pz(), v.E() ) ) 
    {}


    /**
       construct from a generic linear algebra  vector implementing operator []
       and with a size of at least 4. This could be also a C array 
       In this case v[0] is the first data member 
       ( Px for a Cartesian4D base) 
       \param v LA vector
       \param index0 index of first vector element (Px)
    */ 
    template <class LAVector> 
    explicit BasicLorentzVector(const LAVector & v, size_t index0 ) { 
      fCoordinates = CoordSystem ( v[index0], v[index0+1], v[index0+2], v[index0+3] ); 
    }
    

    /**
       copy constructor
    */
    BasicLorentzVector( const BasicLorentzVector & v) : 
      fCoordinates(v.fCoordinates) 
    {}
  

    // coordinate accessors

    /**
       spatial X component 
     */ 
    Scalar Px() const { return fCoordinates.Px(); } 
    /** 
	spatial Y component
     */ 
    Scalar Py() const { return fCoordinates.Py(); } 
    /** 
	spatial Z component
     */ 
    Scalar Pz() const { return fCoordinates.Pz(); } 
    /** 
	return 4-th component (time or energy for a 4-momentum vector) 
     */ 
    Scalar E() const { return fCoordinates.E(); } 

    // repeat for Vector3D compatibility
    /** 
	spatial X component
     */ 
    Scalar X() const { return fCoordinates.Px();}
    /** 
	spatial Y component
     */ 
    Scalar Y() const { return fCoordinates.Py();}
    /** 
	spatial Z component
     */ 
    Scalar Z() const { return fCoordinates.Pz();}
    /** 
	return 4-th component (time or energy for a 4-momentum vector) 
     */ 
    Scalar t() const { return fCoordinates.E(); }




    // vector 
    /**
       return magnitude (mass ) square  M2 = E**2 - Px**2 - Py**2 - Pz**2 (we use -,-,-,+ metric)
     */ 
    Scalar M2() const { return fCoordinates.M2();}

    /**
       return magnitude (mass ) using the  (-,-,-,+)  metric. 
       If M2 is negative (space-like vector) M = - sqrt( -M2) and is negative
     */ 
    Scalar M() const    { return fCoordinates.M();}

    /**
       return the spatial (3D) magnitude ( sqrt(Px**2 + Py**2 + Pz**2) )
     */ 
    Scalar p() const { return fCoordinates.p(); } 

    /**
       return the square of the transverse spatial component ( Px**2 + Py**2 )
     */ 
    Scalar Perp2() const { return fCoordinates.Perp2();}

    /**
       return the  transverse spatial component sqrt ( Px**2 + Py**2 )
     */ 
    Scalar Pt() const { return fCoordinates.Pt();}
    /**
       return the  transverse spatial component sqrt ( Px**2 + Py**2 )
       ( for 3D compatibility )
     */ 
    Scalar Rho() const { return fCoordinates.Pt();}

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
    Scalar et2() const { return fCoordinates.et2(); } 

    /**
       transverse energy 
       \f[ et = \sqrt{ \frac{E^2 p_{\perp}^2 }{ |p|^2 } } X sign(E)
    */
    Scalar et() const { return fCoordinates.et(); } 

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
      return ::ROOT::Math::DisplacementVector3D<Cartesian3D<Scalar> >( Px(), Py(), Pz() ); 
    }


    /**
       access the coordinate representation object (non const version)
     */ 
    CoordSystem & Coordinates() { return fCoordinates;}    

    /**
       access the coordinate representation object (const version) 
     */ 
    const CoordSystem & Coordinates() const { return fCoordinates;}    


    // setter methods  )

    /**
       Set the vector components from cartesian Coordinates X,y,z and t 
     */ 
    void SetPxPyPzE(const Scalar & vpx, const Scalar & vpy, const Scalar & vpz, const Scalar & ve) { 
      // not very efficient, maybe need to implement SetPxPyPzE in System classes
      fCoordinates = Cartesian4D<Scalar>(vpx,vpy,vpz,ve);
    }

    /**
       Set the vector components from the scalar Coordinates and the order is defined according to the coordinate system used  
       \param a Px for a mathcore::Cartesian4D system or Pt for a mathcore::CylndricalEta4D
       \param b Py for a mathcore::Cartesian4D system or Eta for a mathcore::CylndricalEta4D
       \param c Pz for a mathcore::Cartesian4D system or Phi for a mathcore::CylndricalEta4D
       \param d E for a mathcore::Cartesian4D system and a mathcore::CylndricalEta4D
     */ 
    void Set(const Scalar & a, const Scalar & b, const Scalar & c, const Scalar & d) { 
      fCoordinates.SetValues( a, b, c , d); 
    }
 
    /** 
	scalar (Dot) product of two LorentzVector vectors (metric is -,-,-,+)
	Enable the product using any other LorentzVector implementing the X(), Y() , Z() and t() member functions
	\param  q  any LorentzVector implementing the X(), Y() , Z() and t() member functions
	\return the result of v.q of type according to the base scalar type of v
     */ 
    
    template <class OtherLorentzVector>
    Scalar Dot(const OtherLorentzVector & q) const { 
      return E()*q.E() - Px()*q.Px() - Py()*q.Py() - Pz()*q.Pz();
    } 

    /**
       assignment from any other Lorentz vector  implementing 
       Px(), Py(), Pz() and E()
    */
    template<class OtherLorentzVector>
    BasicLorentzVector & operator = ( const OtherLorentzVector & v) { 
      SetPxPyPzE(v.Px(), v.Py(), v.Pz(), v.E() ); 
      return *this; 
    }


    /**
       assignment from a Lorentz vector of the same type (more efficient)  
    */
    BasicLorentzVector & operator = ( const BasicLorentzVector & v) { 
      if (this != &v)  fCoordinates = v.fCoordinates; 
      return *this;
    }

    /**
       assign from a generic linear algebra  vector implementing operator []
       and with a size of at least 4
       In this case v[0] is the first data member 
       ( Px for a Cartesian4D base) 
       \param v LA vector
       \param index0 index of first vector element (Px)
    */ 
    template <class LAVector> 
    BasicLorentzVector & AssignFrom(const LAVector & v, size_t index0=0 ) { 
      fCoordinates = CoordSystem ( v[index0], v[index0+1], v[index0+2], v[index0+3] ); 
      return *this;
    }


    /**
       add another Vector to itself ( v+= q )
       Enable the addition with any other LorentzVector implementing the Px(), Py() , Pz() and E() member functions
       \param q  any LorentzVector implementing the Px(), Py() , Pz() and E() member functions
    */
    template <class OtherLorentzVector>
    BasicLorentzVector & operator += ( const OtherLorentzVector & q) { 
      SetPxPyPzE( Px() + q.Px(), Py() + q.Py(), Pz() + q.Pz(), E() + q.E()  ); 
      return *this; 
    }


    /**
       addition between LorentzVectors (v3 = v1 + v2) 
       Enable the addition with any other LorentzVector implementing the Px(), Py() , Pz() and E() member functions
       \param v2   any LorentzVector implementing the Px(), Py() , Pz() and E() member functions
       \return a new LorentzVector of the same type of v1  
    */
    template<class OtherLorentzVector>
    BasicLorentzVector  operator +  ( const OtherLorentzVector & v2) { 
      return BasicLorentzVector(Cartesian4D<Scalar>( Px() + v2.Px(), Py() + v2.Py(), Pz() + v2.Pz(), E() + v2.E() ) );  
    }


    /**
       subtract another Vector to itself ( v-= q )
       Enable the subtraction with any other LorentzVector implementing the Px(), Py() , Pz() and E() member functions
       \param q  any LorentzVector implementing the Px(), Py() , Pz() and E() member functions
    */
    template <class OtherLorentzVector>
    BasicLorentzVector & operator -= ( const OtherLorentzVector & q) { 
      SetPxPyPzE( Px() - q.Px(), Py() - q.Py(), Pz() - q.Pz(), E() - q.E()  ); 
      return *this; 
    }


    /**
       subtraction between LorentzVectors (v3 = v1 - v2) 
       Enable the subtraction with any other LorentzVector implementing the Px(), Py() , Pz() and E() member functions
       \param v2   any LorentzVector implementing the Px(), Py() , Pz() and E() member functions
       \return a new LorentzVector of the same type of v1  
    */
    template<class OtherLorentzVector>
    BasicLorentzVector  operator -  ( const OtherLorentzVector & v2) { 
      return BasicLorentzVector(Cartesian4D<Scalar>( Px() - v2.Px(), Py() - v2.Py(), Pz() - v2.Pz(), E() - v2.E() ) );  
    }

    /**
       multiplication by a scalar quantity v *= a
    */ 
    BasicLorentzVector & operator *= ( const Scalar & a) { 
      fCoordinates.Scale(a);
      return *this;
    }

    /**
       division by a scalar quantity v /= a
    */ 
    BasicLorentzVector & operator /= ( const Scalar & a) { 
      // not efficent implementation need to be reimplemented in system classes 
      // need to check that a != 0
      fCoordinates.Scale(1/a);
      return *this;
    }

    /**
       return unary minus ( q = - v )
       \return a new LorentzVector with opposite direction and time
     */ 
    BasicLorentzVector operator - () { 
      return BasicLorentzVector(Cartesian4D<Scalar>( - Px(), - Py(), - Pz(), - E()  ) );  
    }

    // rotations and boost 

    
    /** 
	Rotate along X by an Angle alpha
	Implement using the 3D vector.
     */ 
    BasicLorentzVector & RotateX(const Scalar & alpha) { 
      DisplacementVector3D<Cartesian3D<Scalar> > v = Vec();
      v.RotateX(alpha);
      SetPxPyPzE( v.X(), v.Y(), v.Z(), E() );
      return *this;
    }

    /** 
	Rotate along Y by an Angle alpha
	Implement using the 3D vector.
     */ 
    BasicLorentzVector & RotateY(const Scalar & alpha) { 
      // not very efficient
      DisplacementVector3D<Cartesian3D<Scalar> > v = Vec();
      v.RotateY(alpha);
      SetPxPyPzE( v.X(), v.Y(), v.Z(), E() );
      return *this;
    }

    /** 
	Rotate along Z by an Angle alpha
	Implement using the 3D vector.
     */ 
    BasicLorentzVector & RotateZ(const Scalar & alpha) { 
      // not very efficient
      DisplacementVector3D<Cartesian3D<Scalar> > v = Vec();
      v.RotateZ(alpha);
      SetPxPyPzE( v.X(), v.Y(), v.Z(), E() );
      return *this;
    }

    /**
       boost along X by a value beta (beta < 1)
     */      
    BasicLorentzVector & BoostX(const Scalar &  b) { 
      Scalar beta2 = b * b;
      if (beta2 > 1) { 
	// what to do if beta > 1 do nothing or boost with 1 ??
	beta2 = 1; 
      }
      Scalar gamma = sqrt(1./(1-beta2));
      Scalar x2 = gamma*(Px() + b*E() );
      Scalar e2 = gamma*(E() +  b*Px() );
      SetPxPyPzE( x2, Py(), Pz(), e2  );
      return *this;
    }


    /**
       boost along Y by a value beta (beta < 1)
     */      
    BasicLorentzVector & BoostY(const Scalar &  b) { 
      Scalar beta2 = b*b;
      if (beta2 > 1) { 
	// what to do if beta > 1 do nothing or boost with 1 ??
	beta2 = 1; 
      }
      Scalar gamma = sqrt(1./(1-beta2));
      Scalar y2 = gamma*(Py() + b*E() );
      Scalar e2 = gamma*(E() + b*Py() );
      SetPxPyPzE( Px(), y2, Pz(), e2  );
      return *this;
    }


    /**
       boost along Z by a value beta (beta < 1)
     */      
    BasicLorentzVector & BoostZ(const Scalar &  b) { 
      Scalar beta2 = b*b;
      if (beta2 > 1) { 
	// what to do if beta > 1 do nothing or boost with 1 ??
	beta2 = 1; 
      }
      Scalar gamma = sqrt(1./(1-beta2));
      Scalar z2 = gamma*(Pz() + b*E() );
      Scalar e2 = gamma*(E() + b*Pz() );
      SetPxPyPzE( Px(), Py(), z2, e2  );
      return *this;
    }


    /** 
	apply a transformation (3D rotations , 4D rotations, etc...) 
	precondition: transformation needs to implement operator * accepting this 
	type of LorentzVector
    */
    template <class ArbitraryTransformation> 
    BasicLorentzVector & Transform(const ArbitraryTransformation & t) { 
      return *this = t * (*this); 
    }


      /**
	 Scale of a LorentzVector with a scalar quantity a
	 \param v  mathcore::BasicLorentzVector based on any coordinate system  
	 \param a  scalar quantity of typpe a
	 \return a new mathcoreBasicLorentzVector q = v * a same type as v
      */ 
      BasicLorentzVector operator * ( const Scalar & a) { 
	BasicLorentzVector tmp(*this);
	tmp *= a; 
	return tmp;
      }

      /**
	 Divide a LorentzVector with a scalar quantity a
	 \param v  mathcore::BasicLorentzVector based on any coordinate system  
	 \param a  scalar quantity of typpe a
	 \return a new mathcoreBasicLorentzVector q = v / a same type as v
      */ 
      BasicLorentzVector<CoordSystem> operator / ( const Scalar & a) { 
	BasicLorentzVector<CoordSystem> tmp(*this);
	tmp /= a; 
	return tmp;
      }


  private: 

    CoordSystem  fCoordinates; 


  }; 



  // global nethods 

  /**
     Scale of a LorentzVector with a scalar quantity a
     \param a  scalar quantity of typpe a
     \param v  mathcore::BasicLorentzVector based on any coordinate system  
     \return a new mathcoreBasicLorentzVector q = v * a same type as v
   */ 
    template <class CoordSystem>
    BasicLorentzVector<CoordSystem> operator * ( const typename  BasicLorentzVector<CoordSystem> ::Scalar & a, const BasicLorentzVector<CoordSystem>& v) {  
      BasicLorentzVector<CoordSystem> tmp(v);
      tmp *= a; 
      return tmp;
    }

    
  
//   // transformation

//   // vector transformation 
  
//   /**
//      3D rotations for Lorentz Vector
//   */ 
//   template<typename Scalar, 
// 	   template <typename> class CoordSystem, 
// 	   template <typename> class RotationType> 
//   BasicLorentzVector<Scalar, CoordSystem> operator * (const  Basic3DRotation<Scalar, RotationType> & R, const BasicLorentzVector<Scalar, CoordSystem > & v) { 
//     Scalar vx = v.Px();
//     Scalar vy = v.Py(); 
//     Scalar vz = v.Pz();   
//     Cartesian4D<T> c( R.xx()*vX + R.xy()*vY + R.xz()*vZ,
// 		   R.yx()*vX + R.yy()*vY + R.yz()*vZ,
// 		   R.zx()*vX + R.zy()*vY + R.zz()*vZ, 
// 		   v.E() );
//     return BasicLorentzVector<Scalar, CoordSystem>(c); 
//   }

//   /**
//      4D Transformation on a LorentzVector (Rotation plus boost)
//    */
//   template<typename Scalar, 
// 	   template <typename> class CoordSystem, 
// 	   template <typename> class LRType> 
//   BasicLorentzVector<Scalar, CoordSystem> operator * (const BasicLorentzRotation<Scalar, LRType> & R, const BasicLorentzVector<Scalar, CoordSystem > & v) { 
//     Scalar vx = v.Px();
//     Scalar vy = v.Py(); 
//     Scalar vz = v.Pz();   
//     Scalar vt = v.E();
//     Cartesian4D<T> c( R.xx()*vX + R.xy()*vY + R.xz()*vZ + R.xt()*vt,
// 		   R.yx()*vX + R.yy()*vY + R.yz()*vZ + R.yt()*vt,
// 		   R.zx()*vX + R.zy()*vY + R.zz()*vZ + R.zt()*vt, 
// 		   R.tx()*vX + R.ty()*vY + R.tz()*vZ + R.tt()*vt ); 

//     return BasicLorentzVector<Scalar, CoordSystem>(c); 
//   }



  } // end namespace Math

} // end namespace ROOT


#endif




