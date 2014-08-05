// @(#)root/mathcore:$Id$
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



#ifndef ROOT_Math_GenVector_DisplacementVector3D
#include "Math/GenVector/DisplacementVector3D.h"
#endif

#ifndef ROOT_Math_GenVector_PositionVector3D
#include "Math/GenVector/PositionVector3D.h"
#endif

#ifndef ROOT_Math_GenVector_Rotation3D
#include "Math/GenVector/Rotation3D.h"
#endif

#ifndef ROOT_Math_GenVector_Translation3D
#include "Math/GenVector/Translation3D.h"
#endif


#include "Math/GenVector/AxisAnglefwd.h"
#include "Math/GenVector/EulerAnglesfwd.h"
#include "Math/GenVector/Quaternionfwd.h"
#include "Math/GenVector/RotationZYXfwd.h"
#include "Math/GenVector/RotationXfwd.h"
#include "Math/GenVector/RotationYfwd.h"
#include "Math/GenVector/RotationZfwd.h"

#include <iostream>

//#include "Math/Vector3Dfwd.h"



namespace ROOT {

namespace Math {


   class Plane3D;


//_________________________________________________________________________________________
/**
    Basic 3D Transformation class describing  a rotation and then a translation
    The internal data are a 3D rotation data (represented as a 3x3 matrix) and a 3D vector data.
    They are represented and held in this class like a 3x4 matrix (a simple array of 12 numbers).

    The class can be constructed from any 3D rotation object
    (ROOT::Math::Rotation3D, ROOT::Math::AxisAngle, ROOT::Math::Quaternion, etc...) and/or
    a 3D Vector (ROOT::Math::DislacementVector3D or via ROOT::Math::Translation ) representing a Translation.
    The Transformation is defined by applying first the rotation and then the translation.
    A transformation defined by applying first a translation and then a rotation is equivalent to the
    transformation obtained applying first the rotation and then a translation equivalent to the rotated vector.
    The operator * can be used to obtain directly such transformations, in addition to combine various
    transformations.
    Keep in mind that the operator * (like in the case of rotations ) is not commutative.
    The operator * is used (in addition to operator() ) to apply a transformations on the vector
    (DisplacementVector3D and LorentzVector classes) and point (PositionVector3D)  classes.
    In the case of Vector objects the transformation only rotates them and does not translate them.
    Only Point objects are able to be both rotated and translated.


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
      Construct from a rotation and then a translation described by a Translation3D class
   */
   Transform3D( const Rotation3D & r, const Translation3D & t)
   {
      AssignFrom( r, t.Vect() );
   }

   /**
      Construct from a rotation (any rotation object)  and then a translation
      (represented by any DisplacementVector)
      The requirements on the rotation and vector objects are that they can be transformed in a
      Rotation3D class and in a Cartesian3D Vector
   */
   template <class ARotation, class CoordSystem, class Tag>
   Transform3D( const ARotation & r, const DisplacementVector3D<CoordSystem,Tag> & v)
   {
      AssignFrom( Rotation3D(r), Vector (v.X(),v.Y(),v.Z()) );
   }

   /**
      Construct from a rotation (any rotation object)  and then a translation
      represented by a Translation3D class
      The requirements on the rotation is that it can be transformed in a
      Rotation3D class
   */
   template <class ARotation>
   Transform3D( const ARotation & r, const Translation3D & t)
   {
      AssignFrom( Rotation3D(r), t.Vect() );
   }


#ifdef OLD_VERSION
   /**
      Construct from a translation and then a rotation (inverse assignment)
   */
   Transform3D( const Vector & v, const Rotation3D & r)
   {
      // is equivalent from having first the rotation and then the translation vector rotated
      AssignFrom( r, r(v) );
   }
#endif

   /**
      Construct from a 3D Rotation only with zero translation
   */
   explicit Transform3D( const Rotation3D & r) {
      AssignFrom(r);
   }

   // convenience methods for constructing a Transform3D from all the 3D rotations classes
   // (cannot use templates for conflict with LA)

   explicit Transform3D( const AxisAngle & r) {
      AssignFrom(Rotation3D(r));
   }
   explicit Transform3D( const EulerAngles & r) {
      AssignFrom(Rotation3D(r));
   }
   explicit Transform3D( const Quaternion & r) {
      AssignFrom(Rotation3D(r));
   }
   explicit Transform3D( const RotationZYX & r) {
      AssignFrom(Rotation3D(r));
   }

   // Constructors from axial rotations
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
   /**
      Construct from a translation only, represented by a Translation3D class
      and with an identity rotation
   */
   explicit Transform3D( const Translation3D & t) {
      AssignFrom(t.Vect());
   }



   //#if !defined(__MAKECINT__) && !defined(G__DICTIONARY)  // this is ambigous with double * , double *


#ifdef OLD_VERSION
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
#endif


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
#ifndef NDEBUG
   void SetComponents(IT begin, IT end) {
#else
   void SetComponents(IT begin, IT ) {
#endif
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
#ifndef NDEBUG
   void GetComponents(IT begin, IT end) const {
#else
   void GetComponents(IT begin, IT ) const {
#endif
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
      Get the components into 12 scalars
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
      in any rotation and any vector (the Translation class could also be used)
   */
   template<class AnyRotation, class V>
   void GetDecomposition(AnyRotation &r, V &v) const {
      GetRotation(r);
      GetTranslation(v);
   }


   /**
      Get the rotation and translation vector representing the 3D transformation
   */
   void GetDecomposition(Rotation3D &r, Vector &v) const {
      GetRotation(r);
      GetTranslation(v);
   }

   /**
      Get the 3D rotation representing the 3D transformation
   */
   Rotation3D Rotation() const {
      return Rotation3D( fM[kXX], fM[kXY], fM[kXZ],
                         fM[kYX], fM[kYY], fM[kYZ],
                         fM[kZX], fM[kZY], fM[kZZ] );
   }

   /**
      Get the rotation representing the 3D transformation
   */
   template <class AnyRotation>
   AnyRotation Rotation() const {
      return AnyRotation(Rotation3D(fM[kXX], fM[kXY], fM[kXZ],
                                    fM[kYX], fM[kYY], fM[kYZ],
                                    fM[kZX], fM[kZY], fM[kZZ] ) );
   }

   /**
      Get the  rotation (any type) representing the 3D transformation
   */
   template <class AnyRotation>
   void GetRotation(AnyRotation &r) const {
      r = Rotation();
   }

   /**
      Get the translation representing the 3D transformation in a Cartesian vector
   */
   Translation3D Translation() const {
      return Translation3D( fM[kDX], fM[kDY], fM[kDZ] );
   }

   /**
      Get the translation representing the 3D transformation in any vector
      which implements the SetXYZ method
   */
   template <class AnyVector>
   void GetTranslation(AnyVector &v) const {
      v.SetXYZ(fM[kDX], fM[kDY], fM[kDZ]);
   }



   // operations on points and vectors

   /**
      Transformation operation for Position Vector in Cartesian coordinate
      For a Position Vector first a rotation and then a translation is applied
   */
   Point operator() (const Point & p) const {
      return Point ( fM[kXX]*p.X() + fM[kXY]*p.Y() + fM[kXZ]*p.Z() + fM[kDX],
                     fM[kYX]*p.X() + fM[kYY]*p.Y() + fM[kYZ]*p.Z() + fM[kDY],
                     fM[kZX]*p.X() + fM[kZY]*p.Y() + fM[kZZ]*p.Z() + fM[kDZ] );
   }


   /**
      Transformation operation for Displacement Vectors in Cartesian coordinate
      For the Displacement Vectors only the rotation applies - no translations
   */
   Vector operator() (const Vector & v) const {
      return Vector( fM[kXX]*v.X() + fM[kXY]*v.Y() + fM[kXZ]*v.Z() ,
                     fM[kYX]*v.X() + fM[kYY]*v.Y() + fM[kYZ]*v.Z() ,
                     fM[kZX]*v.X() + fM[kZY]*v.Y() + fM[kZZ]*v.Z()  );
   }


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
   inline Transform3D & operator *= (const Transform3D  & t);

   /**
      multiply (combine) two transformations
   */
   inline Transform3D operator * (const Transform3D  & t) const;

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
      Equality operator. Check equality for each element
      To do: use double tolerance
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

   /**
      Inequality operator. Check equality for each element
      To do: use double tolerance
   */
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


   double fM[12];    // transformation elements (3x4 matrix)

};




// inline functions (combination of transformations)

inline Transform3D & Transform3D::operator *= (const Transform3D  & t)
{
   // combination of transformations

   SetComponents(fM[kXX]*t.fM[kXX]+fM[kXY]*t.fM[kYX]+fM[kXZ]*t.fM[kZX],
                 fM[kXX]*t.fM[kXY]+fM[kXY]*t.fM[kYY]+fM[kXZ]*t.fM[kZY],
                 fM[kXX]*t.fM[kXZ]+fM[kXY]*t.fM[kYZ]+fM[kXZ]*t.fM[kZZ],
                 fM[kXX]*t.fM[kDX]+fM[kXY]*t.fM[kDY]+fM[kXZ]*t.fM[kDZ]+fM[kDX],

                 fM[kYX]*t.fM[kXX]+fM[kYY]*t.fM[kYX]+fM[kYZ]*t.fM[kZX],
                 fM[kYX]*t.fM[kXY]+fM[kYY]*t.fM[kYY]+fM[kYZ]*t.fM[kZY],
                 fM[kYX]*t.fM[kXZ]+fM[kYY]*t.fM[kYZ]+fM[kYZ]*t.fM[kZZ],
                 fM[kYX]*t.fM[kDX]+fM[kYY]*t.fM[kDY]+fM[kYZ]*t.fM[kDZ]+fM[kDY],

                 fM[kZX]*t.fM[kXX]+fM[kZY]*t.fM[kYX]+fM[kZZ]*t.fM[kZX],
                 fM[kZX]*t.fM[kXY]+fM[kZY]*t.fM[kYY]+fM[kZZ]*t.fM[kZY],
                 fM[kZX]*t.fM[kXZ]+fM[kZY]*t.fM[kYZ]+fM[kZZ]*t.fM[kZZ],
                 fM[kZX]*t.fM[kDX]+fM[kZY]*t.fM[kDY]+fM[kZZ]*t.fM[kDZ]+fM[kDZ]);

   return *this;
}



inline Transform3D Transform3D::operator * (const Transform3D  & t) const
{
   // combination of transformations

   return Transform3D(fM[kXX]*t.fM[kXX]+fM[kXY]*t.fM[kYX]+fM[kXZ]*t.fM[kZX],
                      fM[kXX]*t.fM[kXY]+fM[kXY]*t.fM[kYY]+fM[kXZ]*t.fM[kZY],
                      fM[kXX]*t.fM[kXZ]+fM[kXY]*t.fM[kYZ]+fM[kXZ]*t.fM[kZZ],
                      fM[kXX]*t.fM[kDX]+fM[kXY]*t.fM[kDY]+fM[kXZ]*t.fM[kDZ]+fM[kDX],

                      fM[kYX]*t.fM[kXX]+fM[kYY]*t.fM[kYX]+fM[kYZ]*t.fM[kZX],
                      fM[kYX]*t.fM[kXY]+fM[kYY]*t.fM[kYY]+fM[kYZ]*t.fM[kZY],
                      fM[kYX]*t.fM[kXZ]+fM[kYY]*t.fM[kYZ]+fM[kYZ]*t.fM[kZZ],
                      fM[kYX]*t.fM[kDX]+fM[kYY]*t.fM[kDY]+fM[kYZ]*t.fM[kDZ]+fM[kDY],

                      fM[kZX]*t.fM[kXX]+fM[kZY]*t.fM[kYX]+fM[kZZ]*t.fM[kZX],
                      fM[kZX]*t.fM[kXY]+fM[kZY]*t.fM[kYY]+fM[kZZ]*t.fM[kZY],
                      fM[kZX]*t.fM[kXZ]+fM[kZY]*t.fM[kYZ]+fM[kZZ]*t.fM[kZZ],
                      fM[kZX]*t.fM[kDX]+fM[kZY]*t.fM[kDY]+fM[kZZ]*t.fM[kDZ]+fM[kDZ]  );

}




//--- global functions resulting in Transform3D -------


// ------ combination of a  translation (first)  and a rotation ------


/**
   combine a translation and a rotation to give a transform3d
   First the translation then the rotation
 */
inline Transform3D operator * (const Rotation3D & r, const Translation3D & t) {
   return Transform3D( r, r(t.Vect()) );
}
inline Transform3D operator * (const RotationX & r, const Translation3D & t) {
   Rotation3D r3(r);
   return Transform3D( r3, r3(t.Vect()) );
}
inline Transform3D operator * (const RotationY & r, const Translation3D & t) {
   Rotation3D r3(r);
   return Transform3D( r3, r3(t.Vect()) );
}
inline Transform3D operator * (const RotationZ & r, const Translation3D & t) {
   Rotation3D r3(r);
   return Transform3D( r3, r3(t.Vect()) );
}
inline Transform3D operator * (const RotationZYX & r, const Translation3D & t) {
   Rotation3D r3(r);
   return Transform3D( r3, r3(t.Vect()) );
}
inline Transform3D operator * (const AxisAngle & r, const Translation3D & t) {
   Rotation3D r3(r);
   return Transform3D( r3, r3(t.Vect()) );
}
inline Transform3D operator * (const EulerAngles & r, const Translation3D & t) {
   Rotation3D r3(r);
   return Transform3D( r3, r3(t.Vect()) );
}
inline Transform3D operator * (const Quaternion & r, const Translation3D & t) {
   Rotation3D r3(r);
   return Transform3D( r3, r3(t.Vect()) );
}

// ------ combination of a  rotation (first)  and then a translation ------

/**
   combine a rotation and a translation to give a transform3d
   First a rotation then the translation
 */
inline Transform3D operator * (const Translation3D & t, const Rotation3D & r) {
   return Transform3D( r, t.Vect());
}
inline Transform3D operator * (const Translation3D & t, const RotationX & r) {
   return Transform3D( Rotation3D(r) , t.Vect());
}
inline Transform3D operator * (const Translation3D & t, const RotationY & r) {
   return Transform3D( Rotation3D(r) , t.Vect());
}
inline Transform3D operator * (const Translation3D & t, const RotationZ & r) {
   return Transform3D( Rotation3D(r) , t.Vect());
}
inline Transform3D operator * (const Translation3D & t, const RotationZYX & r) {
   return Transform3D( Rotation3D(r) , t.Vect());
}
inline Transform3D operator * (const Translation3D & t, const EulerAngles & r) {
   return Transform3D( Rotation3D(r) , t.Vect());
}
inline Transform3D operator * (const Translation3D & t, const Quaternion & r) {
   return Transform3D( Rotation3D(r) , t.Vect());
}
inline Transform3D operator * (const Translation3D & t, const AxisAngle & r) {
   return Transform3D( Rotation3D(r) , t.Vect());
}

// ------ combination of a Transform3D and a pure translation------

/**
   combine a transformation and a translation to give a transform3d
   First the translation then the transform3D
 */
inline Transform3D operator * (const Transform3D & t, const Translation3D & d) {
   Rotation3D r = t.Rotation();
   return Transform3D( r, r( d.Vect() ) + t.Translation().Vect()  );
}

/**
   combine a translation and a transformation to give a transform3d
   First the transformation then the translation
 */
inline Transform3D operator * (const Translation3D & d, const Transform3D & t) {
   return Transform3D( t.Rotation(), t.Translation().Vect() + d.Vect());
}

// ------ combination of a Transform3D and any rotation------


/**
   combine a transformation and a rotation to give a transform3d
   First the rotation then the transform3D
 */
inline Transform3D operator * (const Transform3D & t, const Rotation3D & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}
inline Transform3D operator * (const Transform3D & t, const RotationX & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}
inline Transform3D operator * (const Transform3D & t, const RotationY & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}
inline Transform3D operator * (const Transform3D & t, const RotationZ & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}
inline Transform3D operator * (const Transform3D & t, const RotationZYX & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}
inline Transform3D operator * (const Transform3D & t, const EulerAngles & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}
inline Transform3D operator * (const Transform3D & t, const AxisAngle & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}
inline Transform3D operator * (const Transform3D & t, const Quaternion & r) {
   return Transform3D( t.Rotation()*r ,  t.Translation()  );
}



/**
   combine a rotation and a transformation to give a transform3d
   First the transformation then the rotation
 */
inline Transform3D operator * (const Rotation3D & r, const Transform3D & t) {
   return Transform3D( r * t.Rotation(), r * t.Translation().Vect() );
}
inline Transform3D operator * (const RotationX & r, const Transform3D & t) {
   Rotation3D r3d(r);
   return Transform3D( r3d * t.Rotation(), r3d * t.Translation().Vect() );
}
inline Transform3D operator * (const RotationY & r, const Transform3D & t) {
   Rotation3D r3d(r);
   return Transform3D( r3d * t.Rotation(), r3d * t.Translation().Vect() );
}
inline Transform3D operator * (const RotationZ & r, const Transform3D & t) {
   Rotation3D r3d(r);
   return Transform3D( r3d * t.Rotation(), r3d * t.Translation().Vect() );
}
inline Transform3D operator * (const RotationZYX & r, const Transform3D & t) {
   Rotation3D r3d(r);
   return Transform3D( r3d * t.Rotation(), r3d * t.Translation().Vect() );
}
inline Transform3D operator * (const EulerAngles & r, const Transform3D & t) {
   Rotation3D r3d(r);
   return Transform3D( r3d * t.Rotation(), r3d * t.Translation().Vect() );
}
inline Transform3D operator * (const AxisAngle & r, const Transform3D & t) {
   Rotation3D r3d(r);
   return Transform3D( r3d * t.Rotation(), r3d * t.Translation().Vect() );
}
inline Transform3D operator * (const Quaternion & r, const Transform3D & t) {
   Rotation3D r3d(r);
   return Transform3D( r3d * t.Rotation(), r3d * t.Translation().Vect() );
}


//---I/O functions
// TODO - I/O should be put in the manipulator form

/**
   print the 12 components of the Transform3D
 */
std::ostream & operator<< (std::ostream & os, const Transform3D & t);


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GenVector_Transform3D */
