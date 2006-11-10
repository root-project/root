// @(#)root/mathcore:$Name:  $:$Id: Transform3D.h,v 1.16 2006/11/09 21:22:53 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class Transform3D
// 
// Created by: Lorenzo Moneta  October 21 2005
// 
// 
#ifndef ROOT_Math_GenVector_Transform3D 
#define ROOT_Math_GenVector_Transform3D  1



#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/AxisAnglefwd.h"
#include "Math/GenVector/EulerAnglesfwd.h"
#include "Math/GenVector/Quaternionfwd.h"
#include "Math/GenVector/RotationXfwd.h"
#include "Math/GenVector/RotationYfwd.h"
#include "Math/GenVector/RotationZfwd.h"

#include <iostream>

//#include "Math/Vector3Dfwd.h"



namespace ROOT { 

  namespace Math { 

    class Plane3D; 



  /** 
     Basic 3D Transformation class describing  a rotation and then a translation
     The internal data are a rotation data and a 3D vector data and they can be represented 
     like a 3x4 matrix
     The class has a template parameter the coordinate system tag of the reference system 
     to which the transformatioon will be applied. For example for transforming from 
     global to local coordinate systems, the transfrom3D has to be instantiated with the 
     coordinate of the traget system

     @ingroup GenVector

  */ 

  class Transform3D { 
    

  public: 

    typedef  DisplacementVector3D<Cartesian3D<double>, DefaultCoordinateSystemTag >  Vector; 
    typedef  PositionVector3D<Cartesian3D<double>, DefaultCoordinateSystemTag >      Point; 


    enum ETransform3DMatrixIndex {
      kXX = 0, kXY = 1, kXZ = 2, kDX = 3, 
      kYX = 4, kYY = 5, kYZ = 6, kDY = 7,
      kZX = 8, kZY = 9, kZZ =10, kDZ = 11
    };



    /** 
	Default constructor (identy rotation) + zero translation
    */ 
    Transform3D()
    {
      SetIdentity();
    }
    
    /**
       Construct given a pair of pointers or iterators defining the
       beginning and end of an array of 12 Scalars
    */
    template<class IT>
    Transform3D(IT begin, IT end) 
    { 
      SetComponents(begin,end); 
    }

    /**
       Construct from a rotation and then a translation described by a Vector 
    */
    Transform3D( const Rotation3D & r, const Vector & v) 
    {
      AssignFrom( r, v ); 
    }
    /**
       Construct from a translation and then a rotation (inverse assignment) 
    */
    Transform3D( const Vector & v, const Rotation3D & r) 
    {
      // is equivalent from having first the rotation and then the translation vector rotated
      AssignFrom( r, r(v) ); 
    }

    /**
       Construct from a 3D Rotation only with zero translation
    */
    explicit Transform3D( const Rotation3D & r) { 
      AssignFrom(r);
    } 
    // convenience methods for the other rotations (cannot use templates for conflict with LA)
    explicit Transform3D( const AxisAngle & r) { 
      AssignFrom(Rotation3D(r));
    } 
    explicit Transform3D( const EulerAngles & r) { 
      AssignFrom(Rotation3D(r));
    } 
    explicit Transform3D( const Quaternion & r) { 
      AssignFrom(Rotation3D(r));
    } 
    // TO DO: implement direct methods for axial rotations without going through Rotation3D
    explicit Transform3D( const RotationX & r) { 
      AssignFrom(Rotation3D(r));
    } 
    explicit Transform3D( const RotationY & r) { 
      AssignFrom(Rotation3D(r));
    } 
    explicit Transform3D( const RotationZ & r) { 
      AssignFrom(Rotation3D(r));
    } 

    /**
       Construct from a translation only, represented by any DisplacementVector3D 
       and with an identity rotation
    */
    template<class CoordSystem, class Tag>
    explicit Transform3D( const DisplacementVector3D<CoordSystem,Tag> & v) { 
      AssignFrom(Vector(v.X(),v.Y(),v.Z()));
    }
    /**
       Construct from a translation only, represented by a Cartesian 3D Vector,  
       and with an identity rotation
    */
    explicit Transform3D( const Vector & v) { 
      AssignFrom(v);
    }



    //#if !defined(__MAKECINT__) && !defined(G__DICTIONARY)  // this is ambigous with double * , double *   
    /**
       Construct from a rotation (any rotation object)  and then a translation 
       (represented by any DisplacementVector)
       The requirements on the rotation and vector objects are that they can be transformed in a 
       Rotation3D class and in a Vector
    */
    // to do : change to displacement vector3D
    template <class ARotation, class CoordSystem, class Tag>
    Transform3D( const ARotation & r, const DisplacementVector3D<CoordSystem,Tag> & v) 
    {
      AssignFrom( Rotation3D(r), Vector (v.X(),v.Y(),v.Z()) ); 
    }
    /**
       Construct from a translation (using any type of DisplacementVector ) 
       and then a rotation (any rotation object). 
       Requirement on the rotation and vector objects are that they can be transformed in a 
       Rotation3D class and in a Vector 
    */
    template <class ARotation, class CoordSystem, class Tag>
    Transform3D(const DisplacementVector3D<CoordSystem,Tag> & v , const ARotation & r) 
    {
      // is equivalent from having first the rotation and then the translation vector rotated
      Rotation3D r3d(r);
      AssignFrom( r3d, r3d( Vector(v.X(),v.Y(),v.Z()) ) ); 
    }

    //#endif

    /**
       Construct transformation from one coordinate system defined by three 
       points (origin + two axis) to 
       a new coordinate system defined by other three points (origin + axis) 
       @param fr0  point defining origin of original reference system 
       @param fr1  point defining first axis of original reference system 
       @param fr2  point defining second axis of original reference system 
       @param to0  point defining origin of transformed reference system 
       @param to1  point defining first axis transformed reference system 
       @param to2  point defining second axis transformed reference system 

     */
    Transform3D
    (const Point & fr0, const Point & fr1, const Point & fr2,  
     const Point & to0, const Point & to1, const Point & to2 );  


    // use compiler generated copy ctor, copy assignmet and dtor

    /**
       Construct from a linear algebra matrix of size at least 3x4,
       which must support operator()(i,j) to obtain elements (0,0) thru (2,3).
       The 3x3 sub-block is assumed to be the rotation part and the translations vector 
       are described by the 4-th column
    */
    template<class ForeignMatrix>
    explicit Transform3D(const ForeignMatrix & m) { 
      SetComponents(m); 
    }

    /**
       Raw constructor from 12 Scalar components
    */
    Transform3D(double  xx, double  xy, double  xz, double dx, 
		double  yx, double  yy, double  yz, double dy,
		double  zx, double  zy, double  zz, double dz)
    {
      SetComponents (xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz);
    }


    /**
       Construct from a linear algebra matrix of size at least 3x4,
       which must support operator()(i,j) to obtain elements (0,0) thru (2,3).
       The 3x3 sub-block is assumed to be the rotation part and the translations vector 
       are described by the 4-th column
    */
    template<class ForeignMatrix>
    Transform3D & operator= (const ForeignMatrix & m) { 
      SetComponents(m); 
      return *this; 
    }


    // ======== Components ==============


    /**
       Set the 12 matrix components given an iterator to the start of
       the desired data, and another to the end (12 past start).
    */
    template<class IT>
    void SetComponents(IT begin, IT end) {
     for (int i = 0; i <12; ++i) { 
        fM[i] = *begin;
        ++begin; 
     }
     assert (end==begin);
    }

    /**
       Get the 12 matrix components into data specified by an iterator begin
       and another to the end of the desired data (12 past start).
    */
    template<class IT>
    void GetComponents(IT begin, IT end) const {
       for (int i = 0; i <12; ++i) { 
          *begin = fM[i];
          ++begin;  
       }
       assert (end==begin);
    }

    /**
       Get the 12 matrix components into data specified by an iterator begin
    */
    template<class IT>
    void GetComponents(IT begin) const {
      std::copy ( fM, fM+12, begin );
    }

    /**
       Set components from a linear algebra matrix of size at least 3x4,
       which must support operator()(i,j) to obtain elements (0,0) thru (2,3).
       The 3x3 sub-block is assumed to be the rotation part and the translations vector 
       are described by the 4-th column
    */
    template<class ForeignMatrix>
    void
    SetTransformMatrix (const ForeignMatrix & m) {
      fM[kXX]=m(0,0);  fM[kXY]=m(0,1);  fM[kXZ]=m(0,2); fM[kDX]=m(0,3);
      fM[kYX]=m(1,0);  fM[kYY]=m(1,1);  fM[kYZ]=m(1,2); fM[kDY]=m(1,3);
      fM[kZX]=m(2,0);  fM[kZY]=m(2,1);  fM[kZZ]=m(2,2); fM[kDZ]=m(2,3);
    }

    /**
       Get components into a linear algebra matrix of size at least 3x4,
       which must support operator()(i,j) for write access to elements
       (0,0) thru (2,3).
    */
    template<class ForeignMatrix>
    void
    GetTransformMatrix (ForeignMatrix & m) const {
      m(0,0)=fM[kXX];  m(0,1)=fM[kXY];  m(0,2)=fM[kXZ];  m(0,3)=fM[kDX];
      m(1,0)=fM[kYX];  m(1,1)=fM[kYY];  m(1,2)=fM[kYZ];  m(1,3)=fM[kDY];
      m(2,0)=fM[kZX];  m(2,1)=fM[kZY];  m(2,2)=fM[kZZ];  m(2,3)=fM[kDZ];
    }


    /**
       Set the components from 12 scalars 
    */
    void
    SetComponents (double  xx, double  xy, double  xz, double dx,  
		   double  yx, double  yy, double  yz, double dy, 
		   double  zx, double  zy, double  zz, double dz) {
      fM[kXX]=xx;  fM[kXY]=xy;  fM[kXZ]=xz;  fM[kDX]=dx;
      fM[kYX]=yx;  fM[kYY]=yy;  fM[kYZ]=yz;  fM[kDY]=dy;
      fM[kZX]=zx;  fM[kZY]=zy;  fM[kZZ]=zz;  fM[kDZ]=dz;
    }

    /**
       Get the nine components into 12 scalars
    */
    void
    GetComponents (double &xx, double &xy, double &xz, double &dx,
		   double &yx, double &yy, double &yz, double &dy,
		   double &zx, double &zy, double &zz, double &dz) const {
      xx=fM[kXX];  xy=fM[kXY];  xz=fM[kXZ];  dx=fM[kDX];
      yx=fM[kYX];  yy=fM[kYY];  yz=fM[kYZ];  dy=fM[kDY];
      zx=fM[kZX];  zy=fM[kZY];  zz=fM[kZZ];  dz=fM[kDZ];
    }

    

    /**
       Get the rotation and translation vector representing the 3D transformation
    */    
    void GetDecomposition(Rotation3D &r, Vector &v) const;



    // operations on points and vectors 

    /**
       Transformation operation for Position Vector in Cartesian coordinate 
    */
    Point operator() (const Point & p) const;  


    /**
       Transformation operation for Displacement Vectors in Cartesian coordinate 
       For the Displacement Vectors only the rotation applies - no translations
    */
    Vector operator() (const Vector & v) const;


    /**
       Transformation operation for Position Vector in any coordinate system 
    */
    template<class CoordSystem > 
    PositionVector3D<CoordSystem> operator() (const PositionVector3D <CoordSystem> & p) const { 
      Point xyzNew = operator() ( Point(p) );
      return  PositionVector3D<CoordSystem> (xyzNew);
    }

    /**
       Transformation operation for Displacement Vector in any coordinate system 
    */
    template<class CoordSystem > 
    DisplacementVector3D<CoordSystem> operator() (const DisplacementVector3D <CoordSystem> & v) const { 
      Vector xyzNew = operator() ( Vector(v) );
      return  DisplacementVector3D<CoordSystem> (xyzNew);
    }

    /**
       Transformation operation for points between different coordinate system tags 
    */
    template<class CoordSystem, class Tag1, class Tag2 > 
    void Transform (const PositionVector3D <CoordSystem,Tag1> & p1, PositionVector3D <CoordSystem,Tag2> & p2  ) const { 
      Point xyzNew = operator() ( Point(p1.X(), p1.Y(), p1.Z()) );
      p2.SetXYZ( xyzNew.X(), xyzNew.Y(), xyzNew.Z() ); 
    }


    /**
       Transformation operation for Displacement Vector of different coordinate systems 
    */
    template<class CoordSystem,  class Tag1, class Tag2 > 
    void Transform (const DisplacementVector3D <CoordSystem,Tag1> & v1, DisplacementVector3D <CoordSystem,Tag2> & v2  ) const { 
      Vector xyzNew = operator() ( Vector(v1.X(), v1.Y(), v1.Z() ) );
      v2.SetXYZ( xyzNew.X(), xyzNew.Y(), xyzNew.Z() ); 
    }

    /**
       Transformation operation for a Lorentz Vector in any  coordinate system 
    */
    template <class CoordSystem > 
    LorentzVector<CoordSystem> operator() (const LorentzVector<CoordSystem> & q) const { 
      Vector xyzNew = operator() ( Vector(q.Vect() ) );
      return  LorentzVector<CoordSystem> (xyzNew.X(), xyzNew.Y(), xyzNew.Z(), q.E() );
    }

    /**
       Transformation on a 3D plane
    */
    Plane3D operator() (const Plane3D & plane) const; 
          

    // skip transformation for arbitrary vectors - not really defined if point or displacement vectors

    // same but with operator * 
    /**
       Transformation operation for Vectors. Apply same rules as operator() 
       depending on type of vector. 
       Will work only for DisplacementVector3D, PositionVector3D and LorentzVector
    */
    template<class AVector > 
    AVector operator * (const AVector & v) const { 
      return operator() (v);
    }



    /**
       multiply (combine) with another transformation in place
     */
    Transform3D & operator *= (const Transform3D  & t);

    /**
       multiply (combine) two transformations
     */ 
    Transform3D operator * (const Transform3D  & t) const { 
      Transform3D tmp(*this);
      tmp*= t;
      return tmp;
    }

    /** 
	Invert the transformation in place
    */
    void Invert();

    /**
       Return the inverse of the transformation.
    */
    Transform3D Inverse() const { 
      Transform3D t(*this);
      t.Invert();
      return t;
    }


    /**
       Equality/inequality operators
    */
    bool operator == (const Transform3D & rhs) const {
      if( fM[0] != rhs.fM[0] )  return false;
      if( fM[1] != rhs.fM[1] )  return false;
      if( fM[2] != rhs.fM[2] )  return false;
      if( fM[3] != rhs.fM[3] )  return false;
      if( fM[4] != rhs.fM[4] )  return false;
      if( fM[5] != rhs.fM[5] )  return false;
      if( fM[6] != rhs.fM[6] )  return false;
      if( fM[7] != rhs.fM[7] )  return false;
      if( fM[8] != rhs.fM[8] )  return false;
      if( fM[9] != rhs.fM[9] )  return false;
      if( fM[10]!= rhs.fM[10] ) return false;
      if( fM[11]!= rhs.fM[11] ) return false;
      return true;
    }

    bool operator != (const Transform3D & rhs) const {
      return ! operator==(rhs);
    }


  protected: 

    /**
       make transformation from first a rotation then a translation
     */
    void  AssignFrom( const Rotation3D & r, const Vector & v);  

    /**
       make transformation from only rotations (zero translation)
     */
    void  AssignFrom( const Rotation3D & r);  

    /**
       make transformation from only translation (identity rotations)
     */
    void  AssignFrom( const Vector & v);  

    /**
       Set identity transformation (identity rotation , zero translation)
     */
    void SetIdentity() ; 

  private: 


    double fM[12];

  };


    // global functions 

  // TODO - I/O should be put in the manipulator form 

    std::ostream & operator<< (std::ostream & os, const Transform3D & t);
  
    // need a function Transform = Translation * Rotation ???

   




  } // end namespace Math

} // end namespace ROOT


#endif /* MATHCORE_BASIC3DTRANSFORMATION */
