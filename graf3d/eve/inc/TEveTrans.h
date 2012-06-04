// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrans
#define ROOT_TEveTrans

#include "TEveVector.h"
#include "TVector3.h"

class TGeoMatrix;
class TGeoHMatrix;
class TBuffer3D;

/******************************************************************************/
// TEveTrans -- 3D transformation in generalised coordinates
/******************************************************************************/

class TEveTrans : public TObject
{
   friend class TEveTransSubEditor;
   friend class TEveTransEditor;

protected:
   Double32_t            fM[16];

   mutable Float_t       fA1;   //!
   mutable Float_t       fA2;   //!
   mutable Float_t       fA3;   //!
   mutable Bool_t        fAsOK; //!

   // TEveUtil
   Bool_t                fUseTrans;       // use transformation matrix
   Bool_t                fEditTrans;      // edit transformation in TGedFrame
   Bool_t                fEditRotation;   // edit rotation
   Bool_t                fEditScale;      // edit scale

   Double_t Norm3Column(Int_t col);
   Double_t Orto3Column(Int_t col, Int_t ref);

public:
   TEveTrans();
   TEveTrans(const TEveTrans& t);
   TEveTrans(const Double_t arr[16]);
   TEveTrans(const Float_t  arr[16]);
   virtual ~TEveTrans() {}

   // General operations

   void     UnitTrans();
   void     ZeroTrans(Double_t w=1.0);
   void     UnitRot();
   void     SetTrans(const TEveTrans& t, Bool_t copyAngles=kTRUE);
   void     SetFromArray(const Double_t arr[16]);
   void     SetFromArray(const Float_t  arr[16]);
   TEveTrans&  operator=(const TEveTrans& t) { SetTrans(t); return *this; }
   void     SetupRotation(Int_t i, Int_t j, Double_t f);
   void     SetupFromToVec(const TEveVector& from, const TEveVector& to);

   void     OrtoNorm3();
   Double_t Invert();

   void MultLeft(const TEveTrans& t);
   void MultRight(const TEveTrans& t);
   void operator*=(const TEveTrans& t) { MultRight(t); }

   void TransposeRotationPart();

   TEveTrans operator*(const TEveTrans& t);

   // Move & Rotate

   void MoveLF(Int_t ai, Double_t amount);
   void Move3LF(Double_t x, Double_t y, Double_t z);
   void RotateLF(Int_t i1, Int_t i2, Double_t amount);

   void MovePF(Int_t ai, Double_t amount);
   void Move3PF(Double_t x, Double_t y, Double_t z);
   void RotatePF(Int_t i1, Int_t i2, Double_t amount);

   void Move(const TEveTrans& a, Int_t ai, Double_t amount);
   void Move3(const TEveTrans& a, Double_t x, Double_t y, Double_t z);
   void Rotate(const TEveTrans& a, Int_t i1, Int_t i2, Double_t amount);

   // Element access

   Double_t* Array() { return fM; }      const Double_t* Array() const { return fM; }
   Double_t* ArrX()  { return fM; }      const Double_t* ArrX()  const { return fM; }
   Double_t* ArrY()  { return fM +  4; } const Double_t* ArrY()  const { return fM +  4; }
   Double_t* ArrZ()  { return fM +  8; } const Double_t* ArrZ()  const { return fM +  8; }
   Double_t* ArrT()  { return fM + 12; } const Double_t* ArrT()  const { return fM + 12; }

   Double_t  operator[](Int_t i) const { return fM[i]; }
   Double_t& operator[](Int_t i)       { return fM[i]; }

   Double_t  CM(Int_t i, Int_t j) const { return fM[4*j + i]; }
   Double_t& CM(Int_t i, Int_t j)       { return fM[4*j + i]; }

   Double_t  operator()(Int_t i, Int_t j) const { return fM[4*j + i - 5]; }
   Double_t& operator()(Int_t i, Int_t j)       { return fM[4*j + i - 5]; }

   // Base-vector interface

   void SetBaseVec(Int_t b, Double_t x, Double_t y, Double_t z);
   void SetBaseVec(Int_t b, const TVector3& v);

   TVector3 GetBaseVec(Int_t b) const;
   void     GetBaseVec(Int_t b, TVector3& v) const;

   // Position interface

   void SetPos(Double_t x, Double_t y, Double_t z);
   void SetPos(Double_t* x);
   void SetPos(Float_t * x);
   void SetPos(const TEveTrans& t);

   void GetPos(Double_t& x, Double_t& y, Double_t& z) const;
   void GetPos(Double_t* x) const;
   void GetPos(Float_t * x) const;
   void GetPos(TVector3& v) const;
   TVector3 GetPos() const;

   // Cardan angle interface

   void SetRotByAngles(Float_t a1, Float_t a2, Float_t a3);
   void SetRotByAnyAngles(Float_t a1, Float_t a2, Float_t a3, const char* pat);
   void GetRotAngles(Float_t* x) const;

   // Scaling

   void     Scale(Double_t sx, Double_t sy, Double_t sz);
   Double_t Unscale();
   void     Unscale(Double_t& sx, Double_t& sy, Double_t& sz);
   void     GetScale(Double_t& sx, Double_t& sy, Double_t& sz) const;
   void     SetScale(Double_t  sx, Double_t  sy, Double_t  sz);
   void     SetScaleX(Double_t sx);
   void     SetScaleY(Double_t sy);
   void     SetScaleZ(Double_t sz);

   // Operations on vectors

   void     MultiplyIP(TVector3& v, Double_t w=1) const;
   void     MultiplyIP(Double_t* v, Double_t w=1) const;
   void     MultiplyIP(Float_t*  v, Double_t w=1) const;
   TVector3 Multiply(const TVector3& v, Double_t w=1) const;
   void     Multiply(const Double_t *vin, Double_t* vout, Double_t w=1) const;
   void     RotateIP(TVector3& v) const;
   void     RotateIP(Double_t* v) const;
   void     RotateIP(Float_t*  v) const;
   TVector3 Rotate(const TVector3& v) const;

   virtual void Print(Option_t* option = "") const;

   // TEveUtil stuff

   void SetFrom(Double_t* carr);
   void SetFrom(const TGeoMatrix& mat);
   void SetGeoHMatrix(TGeoHMatrix& mat);
   void SetBuffer3D(TBuffer3D& buff);

   Bool_t GetUseTrans()  const { return fUseTrans; }
   void SetUseTrans(Bool_t v)  { fUseTrans = v;    }

   void SetEditRotation(Bool_t x){ fEditRotation = x; }
   void SetEditScale(Bool_t x)   { fEditScale = x; }
   Bool_t GetEditRotation()      { return fEditRotation; }
   Bool_t GetEditScale()         { return fEditScale; }

   Bool_t GetEditTrans() const { return fEditTrans; }
   void SetEditTrans(Bool_t v) { fEditTrans = v;    }

   Bool_t IsScale(Double_t low=0.9, Double_t high=1.1) const;

   ClassDef(TEveTrans, 1); // Column-major 4x4 transforamtion matrix for homogeneous coordinates.
};

std::ostream& operator<<(std::ostream& s, const TEveTrans& t);

#endif
