// @(#)root/gl:$Name:  $:$Id: CsgOps.cxx,v 1.6 2005/04/12 09:29:32 brun Exp $
// Author:  Timur Pocheptsov  01/04/2005
/*
  CSGLib - Software Library for Constructive Solid Geometry
  Copyright (C) 2003-2004  Laurence Bourn

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Please send remarks, questions and bug reports to laurencebourn@hotmail.com
*/

/**
 * I've modified some very nice bounding box tree code from
 * Gino van der Bergen's Free Solid Library below. It's basically
 * the same code - but I've hacked out the transformation stuff as
 * I didn't understand it. I've also made it far less elegant!
 * Laurence Bourn.
 */

/*
  SOLID - Software Library for Interference Detection
  Copyright (C) 1997-1998  Gino van den Bergen

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Please send remarks, questions and bug reports to gino@win.tue.nl,
  or write to:
                  Gino van den Bergen
		  Department of Mathematics and Computing Science
		  Eindhoven University of Technology
		  P.O. Box 513, 5600 MB Eindhoven, The Netherlands
*/

/*
	This file contains compressed version of CSGSolid and SOLID.
	All stuff is in RootCsg namespace (to avoid name clashes),
	I used ROOT's own	typedefs and math functions, removed resulting triangulation
	(I use OpenGL's inner implementation), changed names from MT_xxx CSG_xxx
	to more natural etc.
	Interface to CSGSolid :
	ConvertToMesh(to convert TBuffer3D to mesh)
	BuildUnion
	BuildIntersection
	BuildDifference

	31.03.05 Timur Pocheptsov.
*/

#include <algorithm>
#include <vector>

#include "TBuffer3D.h"
#include "Rtypes.h"
#include "TMath.h"
#include "CsgOps.h"

namespace RootCsg {

   const Double_t epsilon = 1e-10;
   const Double_t epsilon2 = 1e-20;
   const Double_t infinity = 1e50;
   Int_t sign(Double_t x){return x < 0. ? -1 : x > 0. ? 1 : 0;}
   Bool_t fuzzy_zero(Double_t x){return TMath::Abs(x) < epsilon;}
   Bool_t fuzzy_zero2(Double_t x){return TMath::Abs(x) < epsilon2;}

   class Tuple2 {
   protected:
      Double_t fCo[2];
   public:
      Tuple2(){}
      Tuple2(const Double_t *vv){SetValue(vv);}
      Tuple2(Double_t xx, Double_t yy){SetValue(xx, yy);}
      Double_t &operator[](Int_t i){return fCo[i];}
      const Double_t &operator[](Int_t i)const{return fCo[i];}
      Double_t &X(){return fCo[0];}
      const Double_t &X()const{return fCo[0];}
      Double_t &Y(){return fCo[1];}
      const Double_t &Y()const{return fCo[1];}
      Double_t &U(){return fCo[0];}
      const Double_t &U()const{return fCo[0];}
      Double_t &V(){return fCo[1];}
      const Double_t &V()const{return fCo[1];}
      Double_t *GetValue(){return fCo;}
      const Double_t *GetValue()const{return fCo;}
      void GetValue(Double_t *vv)const
      {vv[0] = fCo[0]; vv[1] = fCo[1];}
      void SetValue(const Double_t *vv)
      {fCo[0] = vv[0]; fCo[1] = vv[1];}
      void SetValue(Double_t xx, Double_t yy)
      {fCo[0] = xx; fCo[1] = yy;}
   };

   Bool_t operator == (const Tuple2 &t1, const Tuple2 &t2)
   {return t1[0] == t2[0] && t1[1] == t2[1];}

   class Vector2 : public Tuple2 {
   public:
      Vector2(){}
      Vector2(const Double_t *v) : Tuple2(v) {}
      Vector2(Double_t xx, Double_t yy) : Tuple2(xx, yy) {}
      Vector2 &operator += (const Vector2 &v);
      Vector2 &operator -= (const Vector2 &v);
      Vector2 &operator *= (Double_t s);
      Vector2 &operator /= (Double_t s);
      Double_t Dot(const Vector2 &v)const;
      Double_t Length2()const;
      Double_t Length()const;
      Vector2 Absolute()const;
      void Normalize();
      Vector2 Normalized()const;
      void Scale(Double_t x, Double_t y);
      Vector2 Scaled(Double_t x, Double_t y)const;
      Bool_t FuzzyZero()const;
      Double_t Angle(const Vector2 &v)const;
      Vector2 Cross(const Vector2 &v)const;
      Double_t Triple(const Vector2 &v1, const Vector2 &v2)const;
      Int_t closestAxis()const;
   };

   Vector2 &Vector2::operator+=(const Vector2 &vv)
   {fCo[0] += vv[0]; fCo[1] += vv[1];return *this;}
   Vector2 &Vector2::operator-=(const Vector2 &vv)
   {fCo[0] -= vv[0]; fCo[1] -= vv[1];return *this;}
   Vector2 &Vector2::operator *= (Double_t s)
   {fCo[0] *= s; fCo[1] *= s; return *this;}
   Vector2 &Vector2::operator /= (Double_t s)
   {return *this *= 1. / s;}
   Vector2 operator + (const Vector2 &v1, const Vector2 &v2)
   {return Vector2(v1[0] + v2[0], v1[1] + v2[1]);}
   Vector2 operator - (const Vector2 &v1, const Vector2 &v2)
   {return Vector2(v1[0] - v2[0], v1[1] - v2[1]);}
   Vector2 operator - (const Vector2 &v)
   {return Vector2(-v[0], -v[1]);}
   Vector2 operator * (const Vector2 &v, Double_t s)
   {return Vector2(v[0] * s, v[1] * s);}
   Vector2 operator * (Double_t s, const Vector2 &v){return v * s;}
   Vector2 operator / (const Vector2 & v, Double_t s)
   {return v * (1.0 / s);}
   Double_t Vector2::Dot(const Vector2 &vv)const
   {return fCo[0] * vv[0] + fCo[1] * vv[1];}
   Double_t Vector2::Length2()const{ return Dot(*this); }
   Double_t Vector2::Length()const{ return TMath::Sqrt(Length2());}
   Vector2 Vector2::Absolute()const
   {return Vector2(TMath::Abs(fCo[0]), TMath::Abs(fCo[1]));}
   Bool_t Vector2::FuzzyZero()const
   {return fuzzy_zero2(Length2());}
   void Vector2::Normalize(){ *this /= Length();}
   Vector2 Vector2::Normalized()const{return *this / Length();}
   void Vector2::Scale(Double_t xx, Double_t yy){fCo[0] *= xx; fCo[1] *= yy;}
   Vector2 Vector2::Scaled(Double_t xx, Double_t yy)const
   {return Vector2(fCo[0] * xx, fCo[1] * yy);}
   Double_t Vector2::Angle(const Vector2 &vv)const
   {Double_t s = TMath::Sqrt(Length2() * vv.Length2()); return TMath::ACos(Dot(vv) / s);}
   Double_t  dot(const Vector2 &v1, const Vector2 &v2)
   {return v1.Dot(v2);}
   Double_t length2(const Vector2 &v){return v.Length2();}
   Double_t length(const Vector2 &v){return v.Length();}
   Bool_t fuzzy_zero(const Vector2 &v){return v.FuzzyZero();}
   Bool_t fuzzy_equal(const Vector2 &v1, const Vector2 &v2)
   {return fuzzy_zero(v1 - v2);}
   Double_t Angle(const Vector2 &v1, const Vector2 &v2)
   {return v1.Angle(v2);}

   class Point2 : public Vector2 {
   public:
      Point2(){}
      Point2(const Double_t *v) : Vector2(v){}
      Point2(Double_t x, Double_t y) : Vector2(x, y) {}
      Point2 &operator += (const Vector2 &v);
      Point2 &operator -= (const Vector2 &v);
      Point2 &operator = (const Vector2 &v);
      Double_t Distance(const Point2 &p)const;
      Double_t Distance2(const Point2 &p)const;
      Point2 Lerp(const Point2 &p, Double_t t)const;
   };

   Point2 &Point2::operator += (const Vector2 &v)
   {fCo[0] += v[0]; fCo[1] += v[1];return *this;}
   Point2 &Point2::operator-=(const Vector2& v)
   {fCo[0] -= v[0]; fCo[1] -= v[1];return *this;}
   Point2 &Point2::operator=(const Vector2& v)
   {fCo[0] = v[0]; fCo[1] = v[1];return *this;}
   Double_t Point2::Distance(const Point2& p)const
   {return (p - *this).Length();}
   Double_t Point2::Distance2(const Point2& p)const
   {return (p - *this).Length2();}
   Point2 Point2::Lerp(const Point2 &p, Double_t t)const
   {return Point2(fCo[0] + (p[0] - fCo[0]) * t,
                  fCo[1] + (p[1] - fCo[1]) * t);}
   Point2 operator + (const Point2 &p, const Vector2 &v)
   {return Point2(p[0] + v[0], p[1] + v[1]);}
   Point2 operator - (const Point2 &p, const Vector2 &v)
   {return Point2(p[0] - v[0], p[1] - v[1]);}
   Vector2 operator - (const Point2 &p1, const Point2 &p2)
   {return Vector2(p1[0] - p2[0], p1[1] - p2[1]);}
   Double_t distance(const Point2 &p1, const Point2 &p2)
   {return p1.Distance(p2);}
   Double_t distance2(const Point2 &p1, const Point2 &p2)
   {return p1.Distance2(p2);}
   Point2 lerp(const Point2 &p1, const Point2 &p2, Double_t t)
   {return p1.Lerp(p2, t);}

   class Tuple3 {
   protected:
      Double_t fCo[3];
   public:
      Tuple3(){}
      Tuple3(const Double_t *v){SetValue(v);}
      Tuple3(Double_t xx, Double_t yy, Double_t zz){SetValue(xx, yy, zz);}
      Double_t &operator [] (Int_t i){return fCo[i];}
      const Double_t &operator[](Int_t i)const{return fCo[i];}
      Double_t &X(){return fCo[0];}
      const Double_t &X()const{return fCo[0];}
      Double_t &Y(){return fCo[1];}
      const Double_t &Y()const{return fCo[1];}
      Double_t &Z(){return fCo[2];}
      const Double_t &Z()const{return fCo[2];}
      Double_t *GetValue(){return fCo;}
      const Double_t *GetValue()const{ return fCo; }
      void GetValue(Double_t *v)const
      {v[0] = Double_t(fCo[0]), v[1] = Double_t(fCo[1]), v[2] = Double_t(fCo[2]);}
      void SetValue(const Double_t *v)
      {fCo[0] = Double_t(v[0]), fCo[1] = Double_t(v[1]), fCo[2] = Double_t(v[2]);}
      void SetValue(Double_t xx, Double_t yy, Double_t zz)
      {fCo[0] = xx; fCo[1] = yy; fCo[2] = zz;}
   };

   Bool_t operator==(const Tuple3& t1, const Tuple3& t2)
   {return t1[0] == t2[0] && t1[1] == t2[1] && t1[2] == t2[2];}

   class Vector3 : public Tuple3 {
   public:
      Vector3(){}
      Vector3(const Double_t *v) : Tuple3(v){}
      Vector3(Double_t xx, Double_t yy, Double_t zz) : Tuple3(xx, yy, zz){}
      Vector3 &operator += (const Vector3& v);
      Vector3 &operator -= (const Vector3& v);
      Vector3 &operator *= (Double_t s);
      Vector3 &operator /= (Double_t s);
      Double_t Dot(const Vector3& v)const;
      Double_t Length2()const;
      Double_t Length()const;
      Vector3 Absolute()const;
      void NoiseGate(Double_t threshold);
      void Normalize();
      Vector3 Normalized()const;
      Vector3 SafeNormalized()const;
      void Scale(Double_t x, Double_t y, Double_t z);
      Vector3 Scaled(Double_t x, Double_t y, Double_t z)const;
      Bool_t FuzzyZero()const;
      Double_t Angle(const Vector3 &v)const;
      Vector3 Cross(const Vector3 &v)const;
      Double_t Triple(const Vector3 &v1, const Vector3 &v2)const;
      Int_t ClosestAxis()const;
      static Vector3 Random();
   };

   Vector3 &Vector3::operator += (const Vector3 &v)
   {fCo[0] += v[0]; fCo[1] += v[1]; fCo[2] += v[2]; return *this;}
   Vector3 &Vector3::operator -= (const Vector3 &v)
   {fCo[0] -= v[0]; fCo[1] -= v[1]; fCo[2] -= v[2]; return *this;}
   Vector3 &Vector3::operator *= (Double_t s)
   {fCo[0] *= s; fCo[1] *= s; fCo[2] *= s; return *this;}
   Vector3 &Vector3::operator /= (Double_t s)
   {return *this *= Double_t(1.0) / s;}
   Double_t Vector3::Dot(const Vector3 &v)const
   {return fCo[0] * v[0] + fCo[1] * v[1] + fCo[2] * v[2];}
   Double_t Vector3::Length2()const
   {return Dot(*this);}
   Double_t Vector3::Length()const
   {return TMath::Sqrt(Length2());}
   Vector3 Vector3::Absolute()const
   {return Vector3(TMath::Abs(fCo[0]), TMath::Abs(fCo[1]), TMath::Abs(fCo[2]));}
   Bool_t Vector3::FuzzyZero()const
   {return fuzzy_zero(Length2());}
   void Vector3::NoiseGate(Double_t threshold)
   {if (Length2() < threshold) SetValue(0., 0., 0.);}
   void Vector3::Normalize()
   {*this /= Length();}
   Vector3 operator * (const Vector3 &v, Double_t s)
   {return Vector3(v[0] * s, v[1] * s, v[2] * s);}
   Vector3 operator / (const Vector3 &v, Double_t s)
   {return v * (1. / s);}
   Vector3 Vector3::Normalized()const{ return *this / Length(); }
   Vector3 Vector3::SafeNormalized()const
   {Double_t len = Length(); return fuzzy_zero(len) ? Vector3(1., 0., 0.):*this / len;}
   void Vector3::Scale(Double_t xx, Double_t yy, Double_t zz)
   {fCo[0] *= xx; fCo[1] *= yy; fCo[2] *= zz;}
   Vector3 Vector3::Scaled(Double_t xx, Double_t yy, Double_t zz)const
   {return Vector3(fCo[0] * xx, fCo[1] * yy, fCo[2] * zz);}
   Double_t Vector3::Angle(const Vector3 &v)const
   {Double_t s = TMath::Sqrt(Length2() * v.Length2()); return TMath::ACos(Dot(v) / s);}
   Vector3 Vector3::Cross(const Vector3 &v)const
   {return Vector3(fCo[1] * v[2] - fCo[2] * v[1],
                   fCo[2] * v[0] - fCo[0] * v[2],
                   fCo[0] * v[1] - fCo[1] * v[0]);}
   Double_t Vector3::Triple(const Vector3 &v1, const Vector3 &v2)const
   {return fCo[0] * (v1[1] * v2[2] - v1[2] * v2[1]) +
           fCo[1] * (v1[2] * v2[0] - v1[0] * v2[2]) +
           fCo[2] * (v1[0] * v2[1] - v1[1] * v2[0]);}
   Int_t Vector3::ClosestAxis()const
   {Vector3 a = Absolute();
    return a[0] < a[1] ? (a[1] < a[2] ? 2 : 1) : (a[0] < a[2] ? 2 : 0);}

   Vector3 operator + (const Vector3 &v1, const Vector3 &v2)
   {return Vector3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);}
   Vector3 operator - (const Vector3 &v1, const Vector3 &v2)
   {return Vector3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);}
   Vector3 operator - (const Vector3 &v)
   {return Vector3(-v[0], -v[1], -v[2]);}
   Vector3 operator * (Double_t s, const Vector3 &v)
   {return v * s;}
   Vector3 operator * (const Vector3 &v1, const Vector3 &v2)
   {return Vector3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);}

   Double_t  dot(const Vector3 &v1, const Vector3 &v2)
   {return v1.Dot(v2);}
   Double_t length2(const Vector3 &v){return v.Length2();}
   Double_t length(const Vector3 &v){return v.Length();}
   Bool_t fuzzy_zero(const Vector3 &v){return v.FuzzyZero();}
   Bool_t fuzzy_equal(const Vector3 &v1, const Vector3 &v2)
   {return fuzzy_zero(v1 - v2);}
   Double_t Angle(const Vector3 &v1, const Vector3 &v2){return v1.Angle(v2);}
   Vector3 cross(const Vector3 &v1, const Vector3 &v2){return v1.Cross(v2);}
   Double_t triple(const Vector3 &v1, const Vector3 &v2, const Vector3 &v3)
   {return v1.Triple(v2, v3);}

   class Point3 : public Vector3 {
   public:
      Point3(){}
      Point3(const Double_t *v) : Vector3(v) {}
      Point3(Double_t xx, Double_t yy, Double_t zz) : Vector3(xx, yy, zz) {}
      Point3 &operator += (const Vector3 &v);
      Point3 &operator -= (const Vector3 &v);
      Point3 &operator = (const Vector3 &v);
      Double_t Distance(const Point3 &p)const;
      Double_t Distance2(const Point3 &p)const;
      Point3 Lerp(const Point3 &p, Double_t t)const;
   };

   Point3 &Point3::operator += (const Vector3 &v)
   {fCo[0] += v[0]; fCo[1] += v[1]; fCo[2] += v[2];return *this;}
   Point3 &Point3::operator-=(const Vector3 &v)
   {fCo[0] -= v[0]; fCo[1] -= v[1]; fCo[2] -= v[2];return *this;}
   Point3 &Point3::operator=(const Vector3 &v)
   {fCo[0] = v[0]; fCo[1] = v[1]; fCo[2] = v[2];return *this;}
   Double_t Point3::Distance(const Point3 &p)const
   {return (p - *this).Length();}
   Double_t Point3::Distance2(const Point3 &p)const
   {return (p - *this).Length2();}
   Point3 Point3::Lerp(const Point3 &p, Double_t t)const
   {return Point3(fCo[0] + (p[0] - fCo[0]) * t,
                  fCo[1] + (p[1] - fCo[1]) * t,
                  fCo[2] + (p[2] - fCo[2]) * t);}
   Point3 operator + (const Point3 &p, const Vector3 &v)
   {return Point3(p[0] + v[0], p[1] + v[1], p[2] + v[2]);}
   Point3 operator - (const Point3 &p, const Vector3 &v)
   {return Point3(p[0] - v[0], p[1] - v[1], p[2] - v[2]);}
   Vector3 operator - (const Point3 &p1, const Point3 &p2)
   {return Vector3(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);}
   Double_t distance(const Point3 &p1, const Point3 &p2)
   {return p1.Distance(p2);}
   Double_t distance2(const Point3 &p1, const Point3 &p2)
   {return p1.Distance2(p2);}
   Point3 lerp(const Point3 &p1, const Point3 &p2, Double_t t)
   {return p1.Lerp(p2, t);}

   class Tuple4 {
   protected:
      Double_t fCo[4];
   public:
      Tuple4(){}
      Tuple4(const Double_t *v){SetValue(v);}
      Tuple4(Double_t xx, Double_t yy, Double_t zz, Double_t ww)
      {SetValue(xx, yy, zz, ww);}
      Double_t &operator[](Int_t i){return fCo[i];}
      const Double_t &operator[](Int_t i)const{return fCo[i];}
      Double_t &X(){return fCo[0];}
      const Double_t &X()const{return fCo[0];}
      Double_t &Y(){return fCo[1];}
      const Double_t &Y()const{return fCo[1];}
      Double_t &Z(){return fCo[2];}
      const Double_t &Z()const{return fCo[2];}
      Double_t &W(){return fCo[3];}
      const Double_t &W()const{return fCo[3];}
      Double_t *GetValue(){return fCo;}
      const Double_t *GetValue()const{return fCo;}
      void GetValue(Double_t *v)const
      {v[0] = fCo[0]; v[1] = fCo[1]; v[2] = fCo[2]; v[3] = fCo[3];}
      void SetValue(const Double_t *v)
      {fCo[0] = v[0]; fCo[1] = v[1]; fCo[2] = v[2]; fCo[3] = v[3];}
      void SetValue(Double_t xx, Double_t yy, Double_t zz, Double_t ww)
      {fCo[0] = xx; fCo[1] = yy; fCo[2] = zz; fCo[3] = ww;}
   };

   Bool_t operator == (const Tuple4 &t1, const Tuple4 &t2)
   {return t1[0] == t2[0] && t1[1] == t2[1] && t1[2] == t2[2] && t1[3] == t2[3];}
   class Matrix3x3 {
   protected:
      Vector3 fEl[3];
   public:
      Matrix3x3(){}
      Matrix3x3(const Double_t *m){SetValue(m);}
      Matrix3x3(const Vector3 &euler){SetEuler(euler);}
      Matrix3x3(const Vector3 &euler, const Vector3 &s)
      {SetEuler(euler); Scale(s[0], s[1], s[2]);}
      Matrix3x3(Double_t xx, Double_t xy, Double_t xz,
                Double_t yx, Double_t yy, Double_t yz,
                Double_t zx, Double_t zy, Double_t zz)
      {SetValue(xx, xy, xz, yx, yy, yz, zx, zy, zz);}

      Vector3 &operator [] (Int_t i){return fEl[i];}
      const Vector3 &operator [] (Int_t i)const{return fEl[i];}
      void SetValue(const Double_t *m)
      {fEl[0][0] = *m++; fEl[1][0] = *m++; fEl[2][0] = *m++; m++;
       fEl[0][1] = *m++; fEl[1][1] = *m++; fEl[2][1] = *m++; m++;
       fEl[0][2] = *m++; fEl[1][2] = *m++; fEl[2][2] = *m;}
      void SetValue(Double_t xx, Double_t xy, Double_t xz,
                    Double_t yx, Double_t yy, Double_t yz,
                    Double_t zx, Double_t zy, Double_t zz)
      {fEl[0][0] = xx; fEl[0][1] = xy; fEl[0][2] = xz;
       fEl[1][0] = yx; fEl[1][1] = yy; fEl[1][2] = yz;
       fEl[2][0] = zx; fEl[2][1] = zy; fEl[2][2] = zz;}
	   void SetEuler(const Vector3 &euler)
      {Double_t ci = TMath::Cos(euler[0]);
       Double_t cj = TMath::Cos(euler[1]);
       Double_t ch = TMath::Cos(euler[2]);
       Double_t si = TMath::Sin(euler[0]);
       Double_t sj = TMath::Sin(euler[1]);
       Double_t sh = TMath::Sin(euler[2]);
       Double_t cc = ci * ch;
       Double_t cs = ci * sh;
       Double_t sc = si * ch;
       Double_t ss = si * sh;
       SetValue(cj * ch, sj * sc - cs, sj * cc + ss,
				    cj * sh, sj * ss + cc, sj * cs - sc,
               -sj, cj * si, cj * ci);}
      void Scale(Double_t x, Double_t y, Double_t z)
      {fEl[0][0] *= x; fEl[0][1] *= y; fEl[0][2] *= z;
       fEl[1][0] *= x; fEl[1][1] *= y; fEl[1][2] *= z;
       fEl[2][0] *= x; fEl[2][1] *= y; fEl[2][2] *= z;}
      Matrix3x3 Scaled(Double_t x, Double_t y, Double_t z)const
      {return Matrix3x3(fEl[0][0] * x, fEl[0][1] * y, fEl[0][2] * z,
                        fEl[1][0] * x, fEl[1][1] * y, fEl[1][2] * z,
                        fEl[2][0] * x, fEl[2][1] * y, fEl[2][2] * z);}
      void SetIdentity()
      {SetValue(1., 0., 0., 0., 1., 0., 0., 0., 1.);}
      void GetValue(Double_t *m)const
      {*m++ = fEl[0][0]; *m++ = fEl[1][0]; *m++ = fEl[2][0]; *m++ = 0.0;
       *m++ = fEl[0][1]; *m++ = fEl[1][1]; *m++ = fEl[2][1]; *m++ = 0.0;
       *m++ = fEl[0][2]; *m++ = fEl[1][2]; *m++ = fEl[2][2]; *m   = 0.0;}
      Matrix3x3 &operator *= (const Matrix3x3 &m);
      Double_t Tdot(Int_t c, const Vector3 &v)const
      {return fEl[0][c] * v[0] + fEl[1][c] * v[1] + fEl[2][c] * v[2];}
      Double_t Cofac(Int_t r1, Int_t c1, Int_t r2, Int_t c2)const
      {return fEl[r1][c1] * fEl[r2][c2] - fEl[r1][c2] * fEl[r2][c1];}
      Double_t Determinant()const;
      Matrix3x3 Adjoint()const;
      Matrix3x3 Absolute()const;
      Matrix3x3 Transposed()const;
      void Transpose();
      Matrix3x3 Inverse()const;
      void Invert();
   };

   Matrix3x3 &Matrix3x3::operator *= (const Matrix3x3 &m)
   {SetValue(m.Tdot(0, fEl[0]), m.Tdot(1, fEl[0]), m.Tdot(2, fEl[0]),
             m.Tdot(0, fEl[1]), m.Tdot(1, fEl[1]), m.Tdot(2, fEl[1]),
             m.Tdot(0, fEl[2]), m.Tdot(1, fEl[2]), m.Tdot(2, fEl[2]));
    return *this;}
   Double_t Matrix3x3::Determinant()const
   {return triple((*this)[0], (*this)[1], (*this)[2]);}
   Matrix3x3 Matrix3x3::Absolute()const
   {return Matrix3x3(TMath::Abs(fEl[0][0]), TMath::Abs(fEl[0][1]), TMath::Abs(fEl[0][2]),
                     TMath::Abs(fEl[1][0]), TMath::Abs(fEl[1][1]), TMath::Abs(fEl[1][2]),
                     TMath::Abs(fEl[2][0]), TMath::Abs(fEl[2][1]), TMath::Abs(fEl[2][2]));}

   Matrix3x3 Matrix3x3::Transposed()const
   {return Matrix3x3(fEl[0][0], fEl[1][0], fEl[2][0],
                     fEl[0][1], fEl[1][1], fEl[2][1],
                     fEl[0][2], fEl[1][2], fEl[2][2]);}
   void Matrix3x3::Transpose()
   {*this = Transposed();}
   Matrix3x3 Matrix3x3::Adjoint()const
   {return Matrix3x3(Cofac(1, 1, 2, 2), Cofac(0, 2, 2, 1), Cofac(0, 1, 1, 2),
                     Cofac(1, 2, 2, 0), Cofac(0, 0, 2, 2), Cofac(0, 2, 1, 0),
                     Cofac(1, 0, 2, 1), Cofac(0, 1, 2, 0), Cofac(0, 0, 1, 1));}
   Matrix3x3 Matrix3x3::Inverse()const
   {Vector3 co(Cofac(1, 1, 2, 2), Cofac(1, 2, 2, 0), Cofac(1, 0, 2, 1));
    Double_t det = dot((*this)[0], co);
    Double_t s = 1. / det;
    return Matrix3x3(co[0] * s, Cofac(0, 2, 2, 1) * s, Cofac(0, 1, 1, 2) * s,
                     co[1] * s, Cofac(0, 0, 2, 2) * s, Cofac(0, 2, 1, 0) * s,
                     co[2] * s, Cofac(0, 1, 2, 0) * s, Cofac(0, 0, 1, 1) * s);}
   void Matrix3x3::Invert()
   {*this = Inverse();}
   Vector3 operator * (const Matrix3x3& m, const Vector3& v)
   {return Vector3(dot(m[0], v), dot(m[1], v), dot(m[2], v));}
   Vector3 operator * (const Vector3& v, const Matrix3x3& m)
   {return Vector3(m.Tdot(0, v), m.Tdot(1, v), m.Tdot(2, v));}

   Matrix3x3 operator * (const Matrix3x3 &m1, const Matrix3x3 &m2)
   {return Matrix3x3(m2.Tdot(0, m1[0]), m2.Tdot(1, m1[0]), m2.Tdot(2, m1[0]),
                     m2.Tdot(0, m1[1]), m2.Tdot(1, m1[1]), m2.Tdot(2, m1[1]),
                     m2.Tdot(0, m1[2]), m2.Tdot(1, m1[2]), m2.Tdot(2, m1[2]));}

   Matrix3x3 mmult_transpose_left(const Matrix3x3 &m1, const Matrix3x3 &m2)
   {return Matrix3x3(m1[0][0] * m2[0][0] + m1[1][0] * m2[1][0] + m1[2][0] * m2[2][0],
                     m1[0][0] * m2[0][1] + m1[1][0] * m2[1][1] + m1[2][0] * m2[2][1],
                     m1[0][0] * m2[0][2] + m1[1][0] * m2[1][2] + m1[2][0] * m2[2][2],
                     m1[0][1] * m2[0][0] + m1[1][1] * m2[1][0] + m1[2][1] * m2[2][0],
                     m1[0][1] * m2[0][1] + m1[1][1] * m2[1][1] + m1[2][1] * m2[2][1],
                     m1[0][1] * m2[0][2] + m1[1][1] * m2[1][2] + m1[2][1] * m2[2][2],
                     m1[0][2] * m2[0][0] + m1[1][2] * m2[1][0] + m1[2][2] * m2[2][0],
                     m1[0][2] * m2[0][1] + m1[1][2] * m2[1][1] + m1[2][2] * m2[2][1],
                     m1[0][2] * m2[0][2] + m1[1][2] * m2[1][2] + m1[2][2] * m2[2][2]);}

   Matrix3x3 mmult_transpose_right(const Matrix3x3& m1, const Matrix3x3& m2)
   {return Matrix3x3(m1[0].Dot(m2[0]), m1[0].Dot(m2[1]), m1[0].Dot(m2[2]),
                     m1[1].Dot(m2[0]), m1[1].Dot(m2[1]), m1[1].Dot(m2[2]),
                     m1[2].Dot(m2[0]), m1[2].Dot(m2[1]), m1[2].Dot(m2[2]));}


   class Line3 {
   private :
      Bool_t fBounds[2];
      Double_t fParams[2];
      Point3 fOrigin;
      Vector3 fDir;
   public :
      Line3();
      Line3(const Point3 &p1, const Point3 &p2);
      Line3(const Point3 &p1, const Vector3 &v);
      Line3(const Point3 &p1, const Vector3 &v, Bool_t bound1, Bool_t bound2);
      static Line3 InfiniteRay(const Point3 &p1, const Vector3 &v);
      const Vector3 &Direction()const {return fDir;}
      const Point3 &Origin()const{ return fOrigin;}
      Bool_t Bounds(Int_t i)const
      {return (i == 0 ? fBounds[0] : fBounds[1]);}
      Bool_t &Bounds(Int_t i)
      {return (i == 0 ? fBounds[0] : fBounds[1]);}
      const Double_t &Param(Int_t i)const
      {return (i == 0 ? fParams[0] : fParams[1]);}
      Double_t &Param(Int_t i)
      {return (i == 0 ? fParams[0] : fParams[1]);}
      Vector3 UnboundSmallestVector(const Point3 &point)const
      {Vector3 diff(fOrigin - point);
       return diff - fDir * diff.Dot(fDir);}
      Double_t UnboundClosestParameter(const Point3 &point)const
      {Vector3 diff(fOrigin-point);	return diff.Dot(fDir);}
      Double_t UnboundDistance(const Point3& point)const
      {return UnboundSmallestVector(point).Length();}
      Bool_t IsParameterOnLine(const Double_t &t) const
      {return ((fParams[0] - epsilon < t) || (!fBounds[0])) && ((fParams[1] > t + epsilon) || (!fBounds[1]));}
   };

   Line3::Line3() : fOrigin(0,0,0), fDir(1,0,0)
   {fBounds[0] = kFALSE; fBounds[1] = kFALSE; fParams[0] = 0; fParams[1] = 1;}
   Line3::Line3(const Point3 &p1, const Point3 &p2) : fOrigin(p1), fDir(p2-p1)
   {fBounds[0] = kTRUE; fBounds[1] = kTRUE; fParams[0] = 0; fParams[1] = 1;}
   Line3::Line3(const Point3 &p1, const Vector3 &v): fOrigin(p1),	fDir(v)
   {fBounds[0] = kFALSE; fBounds[1] = kFALSE; fParams[0] = 0; fParams[1] = 1;}
   Line3::Line3(const Point3 &p1, const Vector3 &v, Bool_t bound1, Bool_t bound2)
            : fOrigin(p1),	fDir(v)
   {fBounds[0] = bound1; fBounds[1] = bound2; fParams[0] = 0; fParams[1] = 1;}

   class Plane3 : public Tuple4 {
   public :
      Plane3(const Vector3 &a, const Vector3 &b, const Vector3 &c);
      Plane3(const Vector3 &n, const Vector3 &p);
      Plane3();
      Plane3(const Plane3 & p):Tuple4(p){}
      Vector3 Normal()const;
		Double_t	Scalar()const;
      void Invert();
      Plane3 &operator = (const Plane3 & rhs);
		Double_t SignedDistance(const Vector3 &)const;
   };

   Plane3::Plane3(const Vector3 &a,	const Vector3 &b,	const Vector3 &c)
   {Vector3 l1 = b-a;
    Vector3 l2 = c-b;
    Vector3 n = l1.Cross(l2);
    n = n.SafeNormalized();
    Double_t d = n.Dot(a);
    fCo[0] = n.X(); fCo[1] = n.Y(); fCo[2] = n.Z(); fCo[3] = -d;}
   Plane3::Plane3(const Vector3 &n,	const Vector3 &p)
   {Vector3 mn = n.SafeNormalized();
    Double_t md = mn.Dot(p);
    fCo[0] = mn.X(); fCo[1] = mn.Y(); fCo[2] = mn.Z(); fCo[3] = -md;}
   Plane3::Plane3() : Tuple4()
   {fCo[0] = 1.; fCo[1] = 0.;	fCo[2] = 0.; fCo[3] = 0.;}
   Vector3 Plane3::Normal()const
   {return Vector3(fCo[0],fCo[1],fCo[2]);}
	Double_t Plane3::Scalar()const
   {return fCo[3];}
   void Plane3::Invert()
   {fCo[0] = -fCo[0]; fCo[1] = -fCo[1]; fCo[2] = -fCo[2]; fCo[3] = -fCo[3];}
   Plane3 &Plane3::operator = (const Plane3 &rhs)
   {fCo[0] = rhs.fCo[0]; fCo[1] = rhs.fCo[1]; fCo[2] = rhs.fCo[2]; fCo[3] = rhs.fCo[3]; return *this;}
	Double_t Plane3::SignedDistance(const Vector3 &v)const
   {return Normal().Dot(v) + fCo[3];}

   class BBox {
      friend Bool_t intersect(const BBox &a, const BBox &b);
   private:
	   Point3 fCenter;
	   Vector3 fExtent;
   public:
      BBox(){}
      BBox(const Point3 &mini, const Point3 &maxi)
      {SetValue(mini,maxi);}
		const Point3 &Center()const
      {return fCenter;}
		const Vector3 &Extent()const
      {return fExtent;}
      Point3 &Center()
      {return fCenter;}
      Vector3 &Extent()
      {return fExtent;}
      void SetValue(const Point3 &mini,const Point3 &maxi)
      {fExtent = (maxi - mini) / 2;	fCenter = mini + fExtent;}
		void Enclose(const BBox &a, const BBox &b)
      {
         Point3 lower(
			   TMath::Min(a.Lower(0), b.Lower(0)),
			   TMath::Min(a.Lower(1), b.Lower(1)),
			   TMath::Min(a.Lower(2), b.Lower(2))
		   );
		   Point3 upper(
			   TMath::Max(a.Upper(0), b.Upper(0)),
			   TMath::Max(a.Upper(1), b.Upper(1)),
			   TMath::Max(a.Upper(2), b.Upper(2))
		   );
		   SetValue(lower, upper);
	   }
		void SetEmpty()
      {fCenter.SetValue(0., 0., 0.);
       fExtent.SetValue(-infinity, -infinity, -infinity);}
      void Include(const Point3 &p)
      {
		   Point3 lower(
			   TMath::Min(Lower(0), p[0]),
			   TMath::Min(Lower(1), p[1]),
			   TMath::Min(Lower(2), p[2])
		   );
		   Point3 upper(
			   TMath::Max(Upper(0), p[0]),
			   TMath::Max(Upper(1), p[1]),
			   TMath::Max(Upper(2), p[2])
		   );
		   SetValue(lower, upper);
      }
      void Include(const BBox &b){Enclose(*this, b);}
		Double_t	Lower(Int_t i)const
      {return fCenter[i] - fExtent[i];}
		Double_t Upper(Int_t i)const
      {return fCenter[i] + fExtent[i];}
      Point3 Lower()const
      {return fCenter - fExtent;}
      Point3 Upper()const
      {return fCenter + fExtent;}
      Double_t Size()const
      {return TMath::Max(TMath::Max(fExtent[0], fExtent[1]), fExtent[2]);}
		Int_t LongestAxis()const{return fExtent.ClosestAxis();}
		Bool_t IntersectXRay(const Point3 &xBase)const
      {
         if (xBase[0] <= Upper(0)) {
            if (xBase[1] <= Upper(1) && xBase[1] >= Lower(1)) {
				   if (xBase[2] <= Upper(2) && xBase[2] >= Lower(2)) {return kTRUE;}
			   }
		   }
		   return kFALSE;
	   }
   };

   Bool_t intersect(const BBox &a, const BBox &b)
   {return TMath::Abs(a.fCenter[0] - b.fCenter[0]) <= a.fExtent[0] + b.fExtent[0] &&
           TMath::Abs(a.fCenter[1] - b.fCenter[1]) <= a.fExtent[1] + b.fExtent[1] &&
           TMath::Abs(a.fCenter[2] - b.fCenter[2]) <= a.fExtent[2] + b.fExtent[2];}

   class BBoxNode {
   public:
     enum ETagType {kLeaf, kInternal};
     BBox fBBox;
     ETagType fTag;
   };

   class BBoxLeaf : public BBoxNode {
   public:
      Int_t fPolyIndex;
      BBoxLeaf() {}
      BBoxLeaf(Int_t polyIndex, const BBox &bbox) : fPolyIndex(polyIndex)
      {fBBox = bbox; fTag = kLeaf;}
   };

   typedef BBoxLeaf* LeafPtr;
   typedef BBoxNode* NodePtr;


   class BBoxInternal : public BBoxNode {
   public:
	   NodePtr fLeftSon;
	   NodePtr fRightSon;
	   BBoxInternal() {}
	   BBoxInternal(Int_t n, LeafPtr leafIt);
   };

   typedef BBoxInternal* InternalPtr;

   class BBoxTree {
   private:
	   Int_t fBranch;
	   LeafPtr fLeaves;
	   InternalPtr fInternals;
	   Int_t fNumLeaves;
   public :
	   BBoxTree() {};
	   NodePtr RootNode() const {return fInternals;}
	   ~BBoxTree(){delete[] fLeaves;	delete[] fInternals;}
	   void BuildTree(LeafPtr leaves, Int_t numLeaves);
   private :
	   void RecursiveTreeBuild(Int_t n, LeafPtr leafIt);
   };

   BBoxInternal::BBoxInternal(Int_t n, LeafPtr leafIt)
   {fTag = kInternal; fBBox.SetEmpty();
    Int_t i;
    for (i=0;i<n;i++) {fBBox.Include(leafIt[i].fBBox);}}
   void BBoxTree::BuildTree(LeafPtr leaves, Int_t numLeaves)
   {fBranch = 0; fLeaves = leaves; fNumLeaves = numLeaves;
    fInternals = new BBoxInternal[numLeaves];
    RecursiveTreeBuild(fNumLeaves,fLeaves);}

   void BBoxTree::RecursiveTreeBuild(Int_t n, LeafPtr leafIt)
   {
      fInternals[fBranch] = BBoxInternal(n,leafIt);
      BBoxInternal& aBBox  = fInternals[fBranch];
      fBranch++;

      Int_t axis = aBBox.fBBox.LongestAxis();
      Int_t i = 0, mid = n;

	   while (i < mid) {
		   if (leafIt[i].fBBox.Center()[axis] < aBBox.fBBox.Center()[axis]) {
			   ++i;
		   } else {
			   --mid;
			   std::swap(leafIt[i], leafIt[mid]);
		   }
	   }
	   if (mid == 0 || mid == n) {
		   mid = n / 2;
	   }
	   if (mid >= 2) {
		   aBBox.fRightSon = fInternals + fBranch;
		   RecursiveTreeBuild(mid,leafIt);
	   } else {
		   aBBox.fRightSon = leafIt;
	   }
	   if (n - mid >= 2) {
		   aBBox.fLeftSon = fInternals + fBranch;
		   RecursiveTreeBuild(n - mid, leafIt + mid);
	   } else {
		   aBBox.fLeftSon = leafIt + mid;
	   }
   }

	class BlenderVProp {
	private:
		Int_t fVertexIndex;
	public:
		BlenderVProp(Int_t vIndex) : fVertexIndex(vIndex){}
		BlenderVProp(Int_t vIndex, const BlenderVProp &,
						 const BlenderVProp &, const Double_t &)
		{fVertexIndex = vIndex;}
		BlenderVProp(){};
		operator Int_t()const{return fVertexIndex;}
		BlenderVProp &operator = (Int_t i)
		{fVertexIndex = i; return *this;}
	};

	template <class TMesh>
	class PolygonGeometry {
	public:
		typedef typename TMesh::Polygon TPolygon;
	private:
		const TMesh &fMesh;
		const TPolygon &fPoly;
	public:
		PolygonGeometry(const TMesh &mesh, Int_t pIndex)
			: fMesh(mesh), fPoly(mesh.Polys()[pIndex])
		{}
		PolygonGeometry(const TMesh &mesh, const TPolygon &poly)
			: fMesh(mesh), fPoly(poly)
		{}
		const Point3 &operator[] (Int_t i)const
		{return fMesh.Verts()[fPoly[i]].Pos();}
		Int_t Size()const{return fPoly.Size();}
	private:
		PolygonGeometry(const PolygonGeometry &);
		PolygonGeometry& operator = (PolygonGeometry &);
	};

	template <class TPolygon, class TVertex>
	class Mesh : public BaseMesh {
	public:
		typedef std::vector<TVertex> VLIST;
		typedef std::vector<TPolygon> PLIST;
		typedef TPolygon Polygon;
		typedef TVertex Vertex;
		typedef PolygonGeometry<Mesh> TGBinder;
	private:
		VLIST fVerts;
		PLIST fPolys;
	public:
		VLIST &Verts(){return fVerts;}
		const VLIST &Verts()const{return fVerts;}
		PLIST &Polys(){return fPolys;}
		const PLIST &Polys()const{return fPolys;}
		//BaseMesh's final-overriders
		UInt_t NumberOfPolys()const{return fPolys.size();}
		UInt_t NumberOfVertices()const{return fVerts.size();}
		UInt_t SizeOfPoly(UInt_t polyIndex)const{return fPolys[polyIndex].Size();}
		const Double_t *GetVertex(UInt_t vertexNum)const
		{return fVerts[vertexNum].GetValue();}
		Int_t GetVertexIndex(UInt_t polyNum, UInt_t vertexNum)const
		{return fPolys[polyNum][vertexNum];}
	};

	const Int_t cofacTable[3][2] = {{1,2}, {0,2}, {0,1}};

	Bool_t intersect(const Plane3 &p1, const Plane3 &p2, Line3 &output)
	{Matrix3x3 mat;
	 mat[0] = p1.Normal();
	 mat[1] = p2.Normal();
	 mat[2] = mat[0].Cross(mat[1]);
	 if (mat[2].FuzzyZero()) return kFALSE;
	 Vector3 aPoint(-p1.Scalar(),-p2.Scalar(),0);
	 output = Line3(Point3(0., 0., 0.) + mat.Inverse() * aPoint ,mat[2]);
	 return kTRUE;}
	Bool_t intersect_2d_no_bounds_check(const Line3 &l1, const Line3 &l2, Int_t majAxis,
													Double_t& l1Param, Double_t& l2Param)
	{Int_t ind1 = cofacTable[majAxis][0];
	 Int_t ind2 = cofacTable[majAxis][1];
	 Double_t Zx = l2.Origin()[ind1] - l1.Origin()[ind1];
	 Double_t Zy = l2.Origin()[ind2] - l1.Origin()[ind2];
	 Double_t det =  l1.Direction()[ind1]*l2.Direction()[ind2] -
						  l2.Direction()[ind1]*l1.Direction()[ind2];
	 if (fuzzy_zero(det)) return kFALSE;
	 l1Param = (l2.Direction()[ind2]*Zx - l2.Direction()[ind1]*Zy)/det;
	 l2Param = -(-l1.Direction()[ind2]*Zx + l1.Direction()[ind1]*Zy)/det;
	 return kTRUE;}
	Bool_t intersect_2d_bounds_check(const Line3 &l1, const Line3 &l2, Int_t majAxis,
												Double_t &l1Param, Double_t &l2Param)
	{Bool_t isect = intersect_2d_no_bounds_check(l1, l2, majAxis, l1Param, l2Param);
	if (!isect) return kFALSE;
	return l1.IsParameterOnLine(l1Param) && l2.IsParameterOnLine(l2Param);}

	Int_t compute_classification(const Double_t &distance, const Double_t &epsilon)
	{if (TMath::Abs(distance) < epsilon) return 0;
	 else	return distance < 0 ? 1 : 2;}

	template<typename TGBinder>
	Bool_t intersect_poly_with_line_2d(const Line3 &l,	const TGBinder &p1, const Plane3 &plane,
												  Double_t &a,	Double_t &b)
	{Int_t majAxis = plane.Normal().ClosestAxis();
	 Int_t lastInd = p1.Size()-1;
	 b = (-infinity); a = (infinity);
	 Double_t isectParam(0.), isectParam2(0.);
	 Int_t i;
	 Int_t j = lastInd;
	 Int_t isectsFound(0);
	 for (i=0;i<=lastInd; j=i,i++ ) {
		 Line3 testLine(p1[j],p1[i]);
		 if (intersect_2d_bounds_check(l, testLine, majAxis, isectParam, isectParam2)) {
			 ++isectsFound;
			 b = TMath::Max(isectParam, b);
			 a = TMath::Min(isectParam, a);
		 }
	 }
	 return (isectsFound > 0);}

	template<typename TGBinder>
	Bool_t instersect_poly_with_line_3d(const Line3 &l, const TGBinder &p1,
													const Plane3 &plane,	Double_t &a)
	{Double_t determinant = l.Direction().Dot(plane.Normal());
	 if (fuzzy_zero(determinant)) return kFALSE;
	 a = -plane.Scalar() - plane.Normal().Dot(l.Origin());
	 a /= determinant;
	 if (a <= 0 ) return kFALSE;
	 if (!l.IsParameterOnLine(a)) return kFALSE;
	 Point3 pointOnPlane = l.Origin() + l.Direction() * a;
	 return point_in_polygon_test_3d(p1, plane, l.Origin(), pointOnPlane);}

	template<typename TGBinder>
	Bool_t point_in_polygon_test_3d(const TGBinder& p1, const Plane3& plane, const Point3& origin,
											  const Point3 &pointOnPlane)
	{Bool_t discardSign = plane.SignedDistance(origin) < 0 ? kTRUE : kFALSE;
	 const Int_t polySize = p1.Size();
	 Point3 lastPoint = p1[polySize-1];
	 Int_t i;
	 for (i=0;i<polySize; ++i)	{
		 const Point3& aPoint = p1[i];
		 Plane3 testPlane(origin, lastPoint, aPoint);
		 if ((testPlane.SignedDistance(pointOnPlane) <= 0) == discardSign) return kFALSE;
		 lastPoint = aPoint;
	 }
	 return kTRUE;}

	template <typename TGBinder>
	Point3 polygon_mid_point(const TGBinder &p1)
	{Point3 midPoint(0., 0., 0.);
	 Int_t i;
	 for (i=0; i < p1.Size(); i++) midPoint += p1[i];
	 return Point3(midPoint[0] / i, midPoint[1] / i, midPoint[2] / i);}

	template <typename TGBinder>
	Int_t which_side(const TGBinder &p1, const Plane3 &plane1)
	{Int_t output = 0;
	 Int_t i;
	 for (i=0; i<p1.Size(); i++) {
		 Double_t signedDistance = plane1.SignedDistance(p1[i]);
		 if (!fuzzy_zero(signedDistance))
			 signedDistance < 0 ? (output |= 1) : (output |=2);
	 }
	 return output;}

	template <typename TGBinder>
	Line3 polygon_mid_point_ray(const TGBinder &p1,	const Plane3 &plane)
	{return Line3(polygon_mid_point(p1),plane.Normal(),kTRUE,kFALSE);}

	template <typename TGBinder>
	Plane3 compute_plane(const TGBinder &poly)
	{
	 Point3 plast(poly[poly.Size()-1]);
	 Point3 pivot;
	 Vector3 edge;
	 Int_t j;
	 for (j = 0; j < poly.Size(); j++) {
		 pivot = poly[j];
		 edge =  pivot - plast;
		 if (!edge.FuzzyZero()) break;
	 }
	 for (; j < poly.Size(); j++) {
		 Vector3 v2 = poly[j] - pivot;
		 Vector3 v3 = edge.Cross(v2);
		 if (!v3.FuzzyZero()) return Plane3(v3,pivot);
	 }
	 return Plane3();}

	template <typename TGBinder>
	BBox fit_bbox(const TGBinder &p1)
	{BBox bbox;	bbox.SetEmpty();
	 for (Int_t i = 0; i < p1.Size(); ++i)	bbox.Include(p1[i]);
	 return bbox;}

	template<typename TGBinderA, typename TGBinderB>
	Bool_t intersect_polygons (const TGBinderA &p1,	const TGBinderB &p2,
										const Plane3 &plane1, const Plane3 &plane2)
	{Line3 intersectLine;
	 if (!intersect(plane1, plane2, intersectLine))	return kFALSE;
	 Double_t p1A, p1B;
	 Double_t p2A, p2B;
	 if (
		!intersect_poly_with_line_2d(intersectLine,p1,plane1,p1A,p1B) ||
		!intersect_poly_with_line_2d(intersectLine,p2,plane2,p2A,p2B))
	 {
		 return kFALSE;
	 }
	 Double_t maxOMin = TMath::Max(p1A,p2A);
	 Double_t minOMax = TMath::Min(p1B,p2B);
	 return (maxOMin <= minOMax);}

	template <class TMesh, class TSplitFunctionBinder>
	class SplitFunction {
	private:
		TMesh &fMesh;
		TSplitFunctionBinder &fFunctionBinder;
	public:
		SplitFunction(TMesh &mesh, TSplitFunctionBinder &functionBindor)
			: fMesh(mesh), fFunctionBinder(functionBindor)
		{}
		void SplitPolygon(const Int_t p1Index,	const Plane3 &plane,
								Int_t &inPiece, Int_t &outPiece,
								const Double_t onEpsilon)
		{
			const typename TMesh::Polygon &p = fMesh.Polys()[p1Index];
			typename TMesh::Polygon inP(p),outP(p);
			inP.Verts().clear();
			outP.Verts().clear();
			fFunctionBinder.DisconnectPolygon(p1Index);
			Int_t lastIndex = p.Verts().back();
			Point3 lastVertex = fMesh.Verts()[lastIndex].Pos();
			Int_t lastClassification = compute_classification(plane.SignedDistance(lastVertex),onEpsilon);
			Int_t totalClassification(lastClassification);
			Int_t i;
			Int_t j=p.Size()-1;
			for (i = 0; i < p.Size(); j = i, ++i)
			{
				Int_t newIndex = p[i];
				Point3 aVertex = fMesh.Verts()[newIndex].Pos();
				Int_t newClassification = compute_classification(plane.SignedDistance(aVertex),onEpsilon);
				if ((newClassification != lastClassification) && newClassification && lastClassification)
				{
					Int_t newVertexIndex = fMesh.Verts().size();
               typedef typename TMesh::Vertex VERTEX_t;
					fMesh.Verts().push_back(VERTEX_t());
               Vector3 v = aVertex - lastVertex;
					Double_t sideA = plane.SignedDistance(lastVertex);
					Double_t epsilon = -sideA / plane.Normal().Dot(v);
					fMesh.Verts().back().Pos() = lastVertex + (v * epsilon);
					typename TMesh::Polygon::TVProp splitProp(newVertexIndex,p.VertexProps(j),p.VertexProps(i),epsilon);
					inP.Verts().push_back(  splitProp );
					outP.Verts().push_back(	splitProp );
					fFunctionBinder.InsertVertexAlongEdge(lastIndex,newIndex,splitProp);
				}
				Classify(inP.Verts(),outP.Verts(),newClassification, p.VertexProps(i));
				lastClassification = newClassification;
				totalClassification |= newClassification;
				lastVertex = aVertex;
				lastIndex = newIndex;
			}
			if (totalClassification == 3)
			{
				inPiece = p1Index;
				outPiece = fMesh.Polys().size();
				fMesh.Polys()[p1Index] = inP;
				fMesh.Polys().push_back( outP );
				fFunctionBinder.ConnectPolygon(inPiece);
				fFunctionBinder.ConnectPolygon(outPiece);
			} else {
				fFunctionBinder.ConnectPolygon(p1Index);
				if (totalClassification == 1)
				{
					inPiece = p1Index;
					outPiece = -1;
				} else {
					outPiece = p1Index;
					inPiece = -1;
				}
			}
		}
		void Classify(typename TMesh::Polygon::TVPropList &inGroup,
						  typename TMesh::Polygon::TVPropList &outGroup,
						  Int_t classification,
						  typename TMesh::Polygon::TVProp prop)
		{
			switch (classification)
			{
				case 0 :
					inGroup.push_back(prop);
					outGroup.push_back(prop);
					break;
				case 1 :
					inGroup.push_back(prop);
					break;
				case 2 :
					outGroup.push_back(prop);
					break;
				default :
					break;
			}
		}
	};

	template <typename PROP>
	class DefaultSplitFunctionBinder {
	public :
		void DisconnectPolygon(Int_t){}
		void ConnectPolygon(Int_t){}
		void InsertVertexAlongEdge(Int_t,Int_t,const PROP& ){}
	};

	template <typename TMesh>
	class MeshWrapper {
	private :
		TMesh &fMesh;
	public:
		typedef typename TMesh::Polygon Polygon;
		typedef typename TMesh::Vertex Vertex;
		typedef typename TMesh::VLIST VLIST;
		typedef typename TMesh::PLIST PLIST;
		typedef PolygonGeometry<MeshWrapper> TGBinder;
		typedef MeshWrapper<TMesh> MyType;
	public:
		VLIST &Verts(){return fMesh.Verts();}
		const VLIST &Verts()const{return fMesh.Verts();}
		PLIST &Polys(){return fMesh.Polys();}
		const PLIST& Polys() const {return fMesh.Polys();}
		MeshWrapper(TMesh &mesh):fMesh(mesh){}
		void ComputePlanes();
		BBox ComputeBBox()const;
		//void Triangulate();
		void SplitPolygon(Int_t p1Index,	const Plane3 &plane,
								Int_t &inPiece, Int_t &outPiece,	Double_t onEpsilon);
	};

	template <typename TMesh>
	void MeshWrapper<TMesh>::ComputePlanes()
	{PLIST& polyList = Polys();
	 UInt_t i;
	 for (i=0;i < polyList.size(); i++) {
		TGBinder binder(fMesh,i);
		polyList[i].SetPlane(compute_plane(binder));
	 }}
	template <typename TMesh>
	BBox MeshWrapper<TMesh>::ComputeBBox()const
	{const VLIST &vertexList = Verts();
	 BBox bbox;
	 bbox.SetEmpty();
	 Int_t i;
	 for (i=0;i<vertexList.size(); i++) bbox.Include(vertexList[i].Pos());
	 return bbox;}

	template<typename TMesh>
	void MeshWrapper<TMesh>::SplitPolygon(Int_t p1Index, const Plane3 &plane,
													  Int_t &inPiece,	Int_t &outPiece,
													  Double_t onEpsilon)
	{DefaultSplitFunctionBinder<typename TMesh::Polygon::TVProp> defaultSplitFunction;
	 SplitFunction<MyType,DefaultSplitFunctionBinder<typename TMesh::Polygon::TVProp> >
			splitFunction(*this,defaultSplitFunction);
	 splitFunction.SplitPolygon(p1Index,plane,inPiece,outPiece,onEpsilon);}

	template <typename AVProp, typename AFProp>
	class PolygonBase {
	public:
		typedef AVProp TVProp;
		typedef AFProp TFProp;
		typedef std::vector<TVProp> TVPropList;
		typedef typename TVPropList::iterator TVPropIt;
	private :
		TVPropList fVerts;
		Plane3 fPlane;
		TFProp fFaceProp;
		Int_t fClassification;
	public:
		const TVPropList &Verts()const{return fVerts;}
		TVPropList &Verts(){return fVerts;}
		Int_t Size()const{return Int_t(fVerts.size());}
		Int_t operator[](Int_t i) const {return fVerts[i];}
		const TVProp &VertexProps(Int_t i)const{return fVerts[i];}
		TVProp &VertexProps(Int_t i){return fVerts[i];}
		void SetPlane(const Plane3 &plane){fPlane = plane;}
		const Plane3 &Plane()const{return fPlane;}
		Vector3 Normal()const{return fPlane.Normal();}
		Int_t &Classification(){ return fClassification;}
		const Int_t &Classification()const{return fClassification;}
		void Reverse()
		{std::reverse(fVerts.begin(),fVerts.end());fPlane.Invert();}
		TFProp &FProp(){return fFaceProp;}
		const TFProp &FProp()const{return fFaceProp;}
      void AddProp(const TVProp &prop){fVerts.push_back(prop);}
	};

	typedef std::vector<Int_t> PIndexList;
	typedef PIndexList::iterator PIndexIt;
	typedef PIndexList::const_iterator const_PIndexIt;
	typedef std::vector<Int_t> VIndexList;
	typedef VIndexList::iterator VIndexIt;
	typedef VIndexList::const_iterator const_VIndexIt;
	typedef std::vector< PIndexList > OverlapTable;

	template <typename TMesh>
	class TreeIntersector {
	private:
		OverlapTable *fAoverlapsB;
		OverlapTable *fBoverlapsA;
		const TMesh *fMeshA;
		const TMesh *fMeshB;
	public :
		TreeIntersector(const BBoxTree &a, const BBoxTree &b,
							 OverlapTable *aOverlapsB,	OverlapTable *bOverlapsA,
							 const TMesh *meshA,	const TMesh *meshB)
		{fAoverlapsB = aOverlapsB;
		 fBoverlapsA = bOverlapsA;
		 fMeshA = meshA;
		 fMeshB = meshB;
		 MarkIntersectingPolygons(a.RootNode(),b.RootNode());}
	private :
		void MarkIntersectingPolygons(const BBoxNode*a,const BBoxNode *b)
		{
			if (!intersect(a->fBBox, b->fBBox)) return;
			if (a->fTag == BBoxNode::kLeaf && b->fTag == BBoxNode::kLeaf)
			{
				const BBoxLeaf *la = (const BBoxLeaf *)a;
				const BBoxLeaf *lb = (const BBoxLeaf *)b;

				PolygonGeometry<TMesh> pg1(*fMeshA,la->fPolyIndex);
				PolygonGeometry<TMesh> pg2(*fMeshB,lb->fPolyIndex);

				if (intersect_polygons(pg1, pg2,	fMeshA->Polys()[la->fPolyIndex].Plane(),
					 fMeshB->Polys()[lb->fPolyIndex].Plane()))
				{
					(*fAoverlapsB)[lb->fPolyIndex].push_back(la->fPolyIndex);
					(*fBoverlapsA)[la->fPolyIndex].push_back(lb->fPolyIndex);
				}
			} else if ( a->fTag == BBoxNode::kLeaf || (b->fTag != BBoxNode::kLeaf && a->fBBox.Size() < b->fBBox.Size()))
			{
				MarkIntersectingPolygons(a, ((const BBoxInternal *)b)->fLeftSon);
				MarkIntersectingPolygons(a, ((const BBoxInternal *)b)->fRightSon);
			} else {
				MarkIntersectingPolygons(((const BBoxInternal *)a)->fLeftSon, b);
				MarkIntersectingPolygons(((const BBoxInternal *)a)->fRightSon, b);
			}
		}
	};

	template<typename TMesh>
	class RayTreeIntersector {
	private:
		const TMesh *fMeshA;
		Double_t fLastIntersectValue;
		Int_t fPolyIndex;
	public:
		RayTreeIntersector(const BBoxTree &a, const TMesh *meshA, const Line3 &xRay, Int_t &polyIndex)
				:fMeshA(meshA), fLastIntersectValue(infinity), fPolyIndex(-1)
		{FindIntersectingPolygons(a.RootNode(),xRay);
		 polyIndex = fPolyIndex;}
	private :
		void FindIntersectingPolygons(const BBoxNode*a,const Line3& xRay)
		{
			if ((xRay.Origin().X() + fLastIntersectValue < a->fBBox.Lower(0)) ||!a->fBBox.IntersectXRay(xRay.Origin()))
				return;
			if (a->fTag == BBoxNode::kLeaf) {
				const BBoxLeaf *la = (const BBoxLeaf *)a;
				Double_t testParameter(0.);
				PolygonGeometry<TMesh> pg(*fMeshA, la->fPolyIndex);
				if (instersect_poly_with_line_3d(xRay,pg,fMeshA->Polys()[la->fPolyIndex].Plane(),testParameter))
				{
					if (testParameter < fLastIntersectValue) {
						fLastIntersectValue = testParameter;
						fPolyIndex = la->fPolyIndex;
					}
				}
			} else {
				FindIntersectingPolygons(((const BBoxInternal*)a)->fLeftSon, xRay);
				FindIntersectingPolygons(((const BBoxInternal*)a)->fRightSon, xRay);
			}
		}
	};

	class VertexBase {
	protected:
		Int_t fVertexMap;
		Point3 fPos;
	public:
		VertexBase(Double_t x, Double_t y, Double_t z) : fVertexMap(-1), fPos(x, y, z){}
		VertexBase():fVertexMap(-1) {}
		const Point3 &Pos()const{return fPos;}
		Point3 &Pos(){return fPos;}
		Int_t &VertexMap(){return fVertexMap;}
		const Int_t &VertexMap()const{return fVertexMap;}
		Double_t operator[](Int_t ind)const{return fPos[ind];}
		const Double_t * GetValue()const{return fPos.GetValue();}
	};

	class CVertex : public VertexBase {
	private:
		PIndexList fPolygons;
	public:
		CVertex() : VertexBase(){}
		CVertex(const VertexBase& vertex) : VertexBase(vertex){}
		CVertex &operator = (const VertexBase &other)
		{fPos= other.Pos(); return *this;}
		const PIndexList &Polys()const{return fPolygons;}
		PIndexList &Polys(){return fPolygons;}
		Int_t &operator[] (Int_t i) { return fPolygons[i];}
		const Int_t &operator[] (Int_t i)const{return fPolygons[i];}
		void AddPoly(Int_t polyIndex){fPolygons.push_back(polyIndex);}
		void RemovePolygon(Int_t polyIndex)
		{PIndexIt foundIt = std::find(fPolygons.begin(), fPolygons.end(), polyIndex);
		 if (foundIt != fPolygons.end()) {
			 std::swap(fPolygons.back(), *foundIt);
			 fPolygons.pop_back();
		 }
		}
	};

	template <typename TMesh>
	class ConnectedMeshWrapper {
	private:
		TMesh &fMesh;
		UInt_t fUniqueEdgeTestId;
	public:
		typedef typename TMesh::Polygon Polygon;
		typedef typename TMesh::Vertex Vertex;
		typedef typename TMesh::Polygon::TVProp VProp;
		typedef typename TMesh::VLIST VLIST;
		typedef typename TMesh::PLIST PLIST;
		typedef PolygonGeometry<ConnectedMeshWrapper> TGBinder;
		typedef ConnectedMeshWrapper<TMesh> MyType;
		ConnectedMeshWrapper(TMesh &mesh) : fMesh(mesh), fUniqueEdgeTestId(0){}
		VLIST &Verts(){return fMesh.Verts();}
		const VLIST &Verts()const{return fMesh.Verts();}
		PLIST &Polys() {return fMesh.Polys();}
		const PLIST &Polys() const {return fMesh.Polys();}
		void BuildVertexPolyLists();
		void DisconnectPolygon(Int_t polyIndex);
		void ConnectPolygon(Int_t polyIndex);
		//return the polygons neibouring the given edge.
		void EdgePolygons(Int_t v1, Int_t v2, PIndexList &polys);
		void InsertVertexAlongEdge(Int_t v1,Int_t v2, const VProp &prop);
		void SplitPolygon(Int_t p1Index,	const Plane3 &plane,	Int_t &inPiece,
								Int_t &outPiece, Double_t onEpsilon);
	};

	template <class CMesh> class SplitFunctionBinder {
	private:
		CMesh &fMesh;
	public:
		SplitFunctionBinder(CMesh &mesh):fMesh(mesh){}
		void DisconnectPolygon(Int_t polyIndex){fMesh.DisconnectPolygon(polyIndex);}
		void ConnectPolygon(Int_t polygonIndex){fMesh.ConnectPolygon(polygonIndex);}
		void InsertVertexAlongEdge(Int_t lastIndex, Int_t newIndex, const typename CMesh::VProp &prop)
		{fMesh.InsertVertexAlongEdge(lastIndex, newIndex,prop);}
	};

	template <typename TMesh>
	void ConnectedMeshWrapper<TMesh>::BuildVertexPolyLists()
	{UInt_t i;
	for (i=0; i < Polys().size(); i++){ConnectPolygon(i);}}

	template <typename TMesh>
	void ConnectedMeshWrapper<TMesh>::DisconnectPolygon(Int_t polyIndex)
	{const Polygon &poly = Polys()[polyIndex];
	 UInt_t j;
		for (j=0;j<poly.Verts().size(); j++){Verts()[poly[j]].RemovePolygon(polyIndex);}}

	template <typename TMesh>
	void ConnectedMeshWrapper<TMesh>::ConnectPolygon(Int_t polyIndex)
	{const Polygon &poly = Polys()[polyIndex];
	 UInt_t j;
	 for (j=0;j<poly.Verts().size(); j++) {Verts()[poly[j]].AddPoly(polyIndex);}}

	template <typename TMesh>
	void ConnectedMeshWrapper<TMesh>::EdgePolygons(Int_t v1, Int_t v2, PIndexList &polys)
	{++fUniqueEdgeTestId;
	 Vertex &vb1 = Verts()[v1];
	 UInt_t i;
	 for (i=0;i < vb1.Polys().size(); ++i){Polys()[vb1[i]].Classification() = fUniqueEdgeTestId;}
	 Vertex &vb2 = Verts()[v2];
	 UInt_t j;
	 for (j=0;j < vb2.Polys().size(); ++j)
	 {
		 if ((UInt_t)Polys()[vb2[j]].Classification() == fUniqueEdgeTestId) {polys.push_back(vb2[j]);}
	 }
	}

	template <typename TMesh>
	void ConnectedMeshWrapper<TMesh>::InsertVertexAlongEdge(Int_t v1,	Int_t v2, const VProp &prop)
	{
		PIndexList npolys;
		EdgePolygons(v1,v2,npolys);
		Int_t newVertex = Int_t(prop);
		UInt_t i;
		for (i=0;i < npolys.size(); i++)
		{
			typename Polygon::TVPropList& polyVerts = Polys()[npolys[i]].Verts();
			typename Polygon::TVPropIt v1pos = std::find(polyVerts.begin(),polyVerts.end(),v1);
			if (v1pos != polyVerts.end()) {
				typename Polygon::TVPropIt prevPos = (v1pos == polyVerts.begin()) ? polyVerts.end()-1 : v1pos-1;
				typename Polygon::TVPropIt nextPos = (v1pos == polyVerts.end()-1) ? polyVerts.begin() : v1pos+1;
				if (*prevPos == v2) {
					polyVerts.insert(v1pos,prop);
				} else
				if (*nextPos == v2) {
					polyVerts.insert(nextPos,prop);
				} else {
					//assert(kFALSE);
				}
				Verts()[newVertex].AddPoly(npolys[i]);
			} else {
				//assert(kFALSE);
			}
		}
	}


	template <typename TMesh>
	void ConnectedMeshWrapper<TMesh>::SplitPolygon(Int_t p1Index, const Plane3 &plane,
																  Int_t &inPiece,	Int_t &outPiece,
																  Double_t onEpsilon)
	{
		SplitFunctionBinder<MyType> functionBindor(*this);
		SplitFunction<MyType,SplitFunctionBinder<MyType> > splitFunction(*this,functionBindor);
		splitFunction.SplitPolygon(p1Index, plane, inPiece, outPiece, onEpsilon);
	}

	struct NullType{};
	//Original TestPolygon has two parameters, the second is face property

	typedef PolygonBase<BlenderVProp, NullType> TestPolygon;
	typedef Mesh<TestPolygon,VertexBase> AMesh;
	typedef Mesh<TestPolygon,CVertex > AConnectedMesh;
	typedef MeshWrapper<AMesh> AMeshWrapper;
	typedef ConnectedMeshWrapper<AConnectedMesh> AConnectedMeshWrapper;

	template <class TMesh>
	void build_split_group(const TMesh &meshA, const TMesh &meshB,
								  const BBoxTree &treeA, const BBoxTree &treeB,
								  OverlapTable &aOverlapsB, OverlapTable &bOverlapsA)
	{
		aOverlapsB = OverlapTable(meshB.Polys().size());
		bOverlapsA = OverlapTable(meshA.Polys().size());
		TreeIntersector<TMesh>(treeA, treeB, &aOverlapsB, &bOverlapsA, &meshA, &meshB);
	}

	template <class CMesh, class TMesh>
	void partition_mesh(CMesh &mesh, const TMesh &mesh2, const OverlapTable &table)
	{
		UInt_t i;
		Double_t onEpsilon(1e-4);
		for (i = 0; i < table.size(); i++) {
			if (table[i].size()) {
				PIndexList fragments;
				fragments.push_back(i);
				UInt_t j;
				for (j =0 ; j <table[i].size(); ++j) {
					PIndexList newFragments;
					Plane3 splitPlane = mesh2.Polys()[table[i][j]].Plane();
					UInt_t k;
					for (k = 0; k < fragments.size(); ++k) {
						Int_t newInFragment;
						Int_t newOutFragment;
						typename CMesh::TGBinder pg1(mesh,fragments[k]);
						typename TMesh::TGBinder pg2(mesh2,table[i][j]);
						const Plane3 &fragPlane = mesh.Polys()[fragments[k]].Plane();

						if (intersect_polygons(pg1,pg2,fragPlane,splitPlane)) {
							mesh.SplitPolygon(fragments[k], splitPlane, newInFragment, newOutFragment, onEpsilon);
							if (-1 != newInFragment) newFragments.push_back(newInFragment);
							if (-1 != newOutFragment) newFragments.push_back(newOutFragment);
						} else {
							newFragments.push_back(fragments[k]);
						}
					}
					fragments = newFragments;
				}
			}
		}
	}

	template <typename CMesh, typename TMesh>
	void classify_mesh(const TMesh &meshA,	const BBoxTree &aTree, CMesh &meshB)
	{
		UInt_t i;
		for (i = 0; i < meshB.Polys().size(); i++) {
			typename CMesh::TGBinder pg(meshB,i);
			Line3 midPointRay = polygon_mid_point_ray(pg,meshB.Polys()[i].Plane());
			Line3 midPointXRay(midPointRay.Origin(),Vector3(1,0,0));
			Int_t aPolyIndex(-1);
			RayTreeIntersector<TMesh>(aTree,&meshA,midPointXRay,aPolyIndex);
			if (-1 != aPolyIndex) {
				if (meshA.Polys()[aPolyIndex].Plane().SignedDistance(midPointXRay.Origin()) < 0) {
					meshB.Polys()[i].Classification()= 1;
				} else {
					meshB.Polys()[i].Classification()= 2;
				}
			} else {
				meshB.Polys()[i].Classification()= 2;
			}
		}
	}

	template <typename CMesh, typename TMesh>
	void extract_classification(CMesh &meshA,	TMesh &newMesh, Int_t classification, Bool_t reverse)
	{
		UInt_t i;
		for (i = 0; i < meshA.Polys().size(); ++i) {
			typename CMesh::Polygon &meshAPolygon = meshA.Polys()[i];
			if (meshAPolygon.Classification() == classification) {
				newMesh.Polys().push_back(meshAPolygon);
				typename TMesh::Polygon &newPolygon = newMesh.Polys().back();
				if (reverse) newPolygon.Reverse();
				Int_t j;
				for (j=0; j< newPolygon.Size(); j++) {
					if (meshA.Verts()[newPolygon[j]].VertexMap() == -1) {
						newMesh.Verts().push_back(meshA.Verts()[newPolygon[j]]);
						meshA.Verts()[newPolygon[j]].VertexMap() = newMesh.Verts().size() -1;
					}
					newPolygon.VertexProps(j) = meshA.Verts()[newPolygon[j]].VertexMap();
				}
			}
		}
	}

	template <typename MeshA, typename MeshB>
	void copy_mesh(const MeshA &source, MeshB &output)
	{
		Int_t vertexNum = source.Verts().size();
		Int_t polyNum = source.Polys().size();

      typedef typename MeshB::VLIST VLIST_t;
      typedef typename MeshB::PLIST PLIST_t;

		output.Verts() = VLIST_t(vertexNum);
		output.Polys() = PLIST_t(polyNum);

		std::copy(source.Verts().begin(), source.Verts().end(), output.Verts().begin());
		std::copy(source.Polys().begin(), source.Polys().end(), output.Polys().begin());
	}

	void build_tree(const AMesh &mesh, BBoxTree &tree)
	{
		Int_t numLeaves = mesh.Polys().size();
		BBoxLeaf *aLeaves = new BBoxLeaf[numLeaves];
		UInt_t i;
		for (i=0;i< mesh.Polys().size(); i++) {
			PolygonGeometry<AMesh> pg(mesh,i);
			aLeaves[i] = BBoxLeaf(i, fit_bbox(pg));
		}
		tree.BuildTree(aLeaves,numLeaves);
	}

	void extract_classification_preserve(const AMesh &meshA,
													 const AMesh &meshB,
													 const BBoxTree &aTree,
													 const BBoxTree &bTree,
													 const OverlapTable &aOverlapsB,
													 const OverlapTable &bOverlapsA,
													 Int_t aClassification,
													 Int_t bClassification,
													 Bool_t reverseA,
													 Bool_t reverseB,
													 AMesh &output)
	{
		AConnectedMesh meshAPartitioned;
		AConnectedMesh meshBPartitioned;
		copy_mesh(meshA,meshAPartitioned);
		copy_mesh(meshB,meshBPartitioned);
		AConnectedMeshWrapper meshAWrapper(meshAPartitioned);
		AConnectedMeshWrapper meshBWrapper(meshBPartitioned);
		meshAWrapper.BuildVertexPolyLists();
		meshBWrapper.BuildVertexPolyLists();
		partition_mesh(meshAWrapper, meshB, bOverlapsA);
		partition_mesh(meshBWrapper, meshA, aOverlapsB);
		classify_mesh(meshB, bTree, meshAPartitioned);
		classify_mesh(meshA, aTree, meshBPartitioned);
		extract_classification(meshAPartitioned, output, aClassification, reverseA);
		extract_classification(meshBPartitioned, output, bClassification, reverseB);
	}

	void extract_classification(const AMesh &meshA,
										 const AMesh &meshB,
										 const BBoxTree &aTree,
										 const BBoxTree &bTree,
										 const OverlapTable &aOverlapsB,
										 const OverlapTable &bOverlapsA,
										 Int_t aClassification,
										 Int_t bClassification,
										 Bool_t reverseA,
										 Bool_t reverseB,
										 AMesh &output)
	{
		AMesh meshAPartitioned(meshA);
		AMesh meshBPartitioned(meshB);
		AMeshWrapper meshAWrapper(meshAPartitioned);
		AMeshWrapper meshBWrapper(meshBPartitioned);
		partition_mesh(meshAWrapper, meshB, bOverlapsA);
		partition_mesh(meshBWrapper, meshA, aOverlapsB);
		classify_mesh(meshB, bTree, meshAPartitioned);
		classify_mesh(meshA, aTree, meshBPartitioned);
		extract_classification(meshAPartitioned, output, aClassification, reverseA);
		extract_classification(meshBPartitioned, output, bClassification, reverseB);
	}

	AMesh *build_intersection(const AMesh &meshA, const AMesh &meshB,	Bool_t preserve)
	{
		BBoxTree aTree, bTree;
		build_tree(meshA, aTree);
		build_tree(meshB, bTree);
		OverlapTable bOverlapsA(meshA.Polys().size());
		OverlapTable aOverlapsB(meshB.Polys().size());
		build_split_group(meshA, meshB, aTree, bTree, aOverlapsB, bOverlapsA);
		AMesh *output = new AMesh;
		if (preserve) {
			extract_classification_preserve(
													  meshA, meshB, aTree, bTree,
													  aOverlapsB, bOverlapsA,
													  1, 1, kFALSE, kFALSE, *output
													 );
		} else {
			extract_classification(
										  meshA, meshB, aTree, bTree,
										  aOverlapsB, bOverlapsA,
										  1, 1, kFALSE, kFALSE, *output
										 );
		}
		return output;
	}

	AMesh *build_union(const AMesh &meshA, const AMesh &meshB,	Bool_t preserve)
	{
		BBoxTree aTree, bTree;
		build_tree(meshA, aTree);
		build_tree(meshB, bTree);
		OverlapTable bOverlapsA(meshA.Polys().size());
		OverlapTable aOverlapsB(meshB.Polys().size());
		build_split_group(meshA, meshB, aTree, bTree, aOverlapsB, bOverlapsA);
		AMesh *output = new AMesh;
		if (preserve) {
			extract_classification_preserve(
					  								  meshA, meshB, aTree, bTree,
													  aOverlapsB, bOverlapsA,
													  2, 2, kFALSE, kFALSE, *output
												    );
		} else {
			extract_classification(
										  meshA, meshB, aTree, bTree,
										  aOverlapsB, bOverlapsA,
										  2, 2, kFALSE, kFALSE, *output
										 );
		}
		return output;
	}

	AMesh *build_difference(const AMesh &meshA, const AMesh &meshB, Bool_t preserve)
	{
		BBoxTree aTree,bTree;
		build_tree(meshA,aTree);
		build_tree(meshB,bTree);
		OverlapTable bOverlapsA(meshA.Polys().size());
		OverlapTable aOverlapsB(meshB.Polys().size());
		build_split_group(meshA, meshB, aTree, bTree, aOverlapsB, bOverlapsA);
		AMesh *output = new AMesh;
		if (preserve) {
			extract_classification_preserve(
													  meshA, meshB, aTree, bTree,
													  aOverlapsB, bOverlapsA,
													  2, 1, kFALSE, kTRUE, *output
													 );
		} else {
			extract_classification(
										  meshA, meshB, aTree, bTree,
										  aOverlapsB, bOverlapsA,
										  2, 1, kFALSE, kTRUE, *output
										 );
		}
		return output;
	}

   BaseMesh *ConvertToMesh(const TBuffer3D &buff)
   {
      AMesh *newMesh = new AMesh;
      const Double_t *v = buff.fPnts;

      newMesh->Verts().resize(buff.NbPnts());

      for (UInt_t i = 0; i < buff.NbPnts(); ++i)
         newMesh->Verts()[i] = VertexBase(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);

      const Int_t *segs = buff.fSegs;
      const Int_t *pols = buff.fPols;
      Int_t shiftInd = buff.fReflection ? 1 : -1;

      newMesh->Polys().resize(buff.NbPols());

      for (UInt_t numPol = 0, j = 1; numPol < buff.NbPols(); ++numPol) {
         TestPolygon &currPoly = newMesh->Polys()[numPol];
         Int_t segmentInd = shiftInd < 0 ? pols[j] + j : j + 1;
         Int_t segmentCol = pols[j];
         Int_t s1 = pols[segmentInd];
         segmentInd += shiftInd;
         Int_t s2 = pols[segmentInd];
         segmentInd += shiftInd;
         Int_t segEnds[] = {segs[s1 * 3 + 1], segs[s1 * 3 + 2],
                            segs[s2 * 3 + 1], segs[s2 * 3 + 2]};
         Int_t numPnts[3] = {0};

         if (segEnds[0] == segEnds[2]) {
            numPnts[0] = segEnds[1], numPnts[1] = segEnds[0], numPnts[2] = segEnds[3];
         } else if (segEnds[0] == segEnds[3]) {
            numPnts[0] = segEnds[1], numPnts[1] = segEnds[0], numPnts[2] = segEnds[2];
         } else if (segEnds[1] == segEnds[2]) {
            numPnts[0] = segEnds[0], numPnts[1] = segEnds[1], numPnts[2] = segEnds[3];
         } else {
            numPnts[0] = segEnds[0], numPnts[1] = segEnds[1], numPnts[2] = segEnds[2];
         }

         currPoly.AddProp(BlenderVProp(numPnts[0]));
         currPoly.AddProp(BlenderVProp(numPnts[1]));
         currPoly.AddProp(BlenderVProp(numPnts[2]));

         Int_t lastAdded = numPnts[2];

         Int_t end = shiftInd < 0 ? j + 1 : j + segmentCol;
         for (; segmentInd != end; segmentInd += shiftInd) {
            segEnds[0] = segs[pols[segmentInd] * 3 + 1];
            segEnds[1] = segs[pols[segmentInd] * 3 + 2];
            if (segEnds[0] == lastAdded) {
               currPoly.AddProp(BlenderVProp(segEnds[1]));
               lastAdded = segEnds[1];
            } else {
               currPoly.AddProp(BlenderVProp(segEnds[0]));
               lastAdded = segEnds[0];
            }
         }
         j += segmentCol + 2;
      }

      AMeshWrapper wrap(*newMesh);

      wrap.ComputePlanes();

      return newMesh;
   }

   BaseMesh *BuildUnion(const BaseMesh *l, const BaseMesh *r)
   {
      return build_union(*static_cast<const AMesh *>(l), *static_cast<const AMesh *>(r), kFALSE);
   }

   BaseMesh *BuildIntersection(const BaseMesh *l, const BaseMesh *r)
   {
      return build_intersection(*static_cast<const AMesh *>(l), *static_cast<const AMesh *>(r), kFALSE);
   }

   BaseMesh *BuildDifference(const BaseMesh *l, const BaseMesh *r)
   {
      return build_difference(*static_cast<const AMesh *>(l), *static_cast<const AMesh *>(r), kFALSE);
   }

}
