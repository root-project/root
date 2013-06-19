// @(#)root/geom:$Id$
// Author: Andrei Gheata   25/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoMatrix
#define ROOT_TGeoMatrix

/*************************************************************************
 * Geometrical transformations. TGeoMatrix - base class, TGeoTranslation *
 * TGeoRotation, TGeoScale, TGeoCombiTrans, TGeoGenTrans .               *
 *                                                                       *
 *************************************************************************/

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

//--- globals 
const Double_t kNullVector[3]       =       {0.0,  0.0,  0.0};

const Double_t kIdentityMatrix[3*3] =       {1.0,  0.0,  0.0,
                                             0.0,  1.0,  0.0,
                                             0.0,  0.0,  1.0};

const Double_t kUnitScale[3]        =       {1.0,  1.0,  1.0};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoMatrix - base class for geometrical transformations.               //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoMatrix : public TNamed
{
public:
enum EGeoTransfTypes {
   kGeoIdentity  = 0,
   kGeoShared       = BIT(14),
   kGeoTranslation  = BIT(17),
   kGeoRotation     = BIT(18),
   kGeoScale        = BIT(19),
   kGeoReflection   = BIT(20),
   kGeoRegistered   = BIT(21),
   kGeoSavePrimitive = BIT(22),
   kGeoMatrixOwned   = BIT(23),
   kGeoCombiTrans   = kGeoTranslation | kGeoRotation,
   kGeoGenTrans     = kGeoTranslation | kGeoRotation | kGeoScale
};

protected:
   TGeoMatrix(const TGeoMatrix &other);

public :
   TGeoMatrix();
   TGeoMatrix(const char *name);
   virtual ~TGeoMatrix();

   TGeoMatrix& operator=(const TGeoMatrix &matrix);
// Preventing warnings with -Weffc++ in GCC since the behaviour of operator * was chosen so by design.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
   TGeoMatrix& operator*(const TGeoMatrix &right) const;
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif
   Bool_t      operator ==(const TGeoMatrix &other) const;
   
   Bool_t               IsIdentity()    const {return !TestBit(kGeoGenTrans);}
   Bool_t               IsTranslation() const {return TestBit(kGeoTranslation);}
   Bool_t               IsRotation()    const {return TestBit(kGeoRotation);}
   Bool_t               IsReflection()  const {return TestBit(kGeoReflection);}
   Bool_t               IsScale()       const {return TestBit(kGeoScale);}
   Bool_t               IsShared()      const {return TestBit(kGeoShared);}
   Bool_t               IsCombi()       const {return (TestBit(kGeoTranslation) 
                                               && TestBit(kGeoRotation));}
   Bool_t               IsGeneral()     const {return (TestBit(kGeoTranslation) 
                            && TestBit(kGeoRotation) && TestBit(kGeoScale));}
   Bool_t               IsRegistered()  const {return TestBit(kGeoRegistered);}
   Bool_t               IsRotAboutZ()   const;
   void                 GetHomogenousMatrix(Double_t *hmat) const;
   char                *GetPointerName() const;

   virtual Int_t              GetByteCount() const;
   virtual const Double_t    *GetTranslation()    const = 0;
   virtual const Double_t    *GetRotationMatrix() const = 0;
   virtual const Double_t    *GetScale()          const = 0;
   virtual TGeoMatrix&  Inverse()                 const = 0;
   virtual void         LocalToMaster(const Double_t *local, Double_t *master) const;
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master) const;
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master) const;
   virtual TGeoMatrix  *MakeClone() const = 0;
   virtual void         MasterToLocal(const Double_t *master, Double_t *local) const;
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local) const;
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local) const;
   static void          Normalize(Double_t *vect);
   void                 Print(Option_t *option="") const; // *MENU*
   virtual void         RotateX(Double_t) {}
   virtual void         RotateY(Double_t) {}
   virtual void         RotateZ(Double_t) {}
   virtual void         ReflectX(Bool_t leftside,Bool_t rotonly=kFALSE);
   virtual void         ReflectY(Bool_t leftside,Bool_t rotonly=kFALSE);
   virtual void         ReflectZ(Bool_t leftside,Bool_t rotonly=kFALSE);
   virtual void         RegisterYourself();
   void                 SetDefaultName();
   virtual void         SetDx(Double_t) {}
   virtual void         SetDy(Double_t) {}
   virtual void         SetDz(Double_t) {}
   void                 SetShared(Bool_t flag=kTRUE) {SetBit(kGeoShared, flag);}
   
   ClassDef(TGeoMatrix, 1)                 // base geometrical transformation class
};



////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoTranslation - class describing translations. A translation is      //
//    basicaly an array of 3 doubles matching the positions 12, 13        //
//    and 14 in the homogenous matrix description.                        //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoTranslation : public TGeoMatrix
{
protected:
   Double_t             fTranslation[3];  // translation vector
public :
   TGeoTranslation();
   TGeoTranslation(const TGeoTranslation &other);
   TGeoTranslation(const TGeoMatrix &other);
   TGeoTranslation(Double_t dx, Double_t dy, Double_t dz);
   TGeoTranslation(const char *name, Double_t dx, Double_t dy, Double_t dz);
   virtual ~TGeoTranslation() {}
   
   TGeoTranslation& operator=(const TGeoMatrix &matrix);
   TGeoTranslation& operator=(const TGeoTranslation &other) {return operator=((const TGeoMatrix&)other);};

   void                 Add(const TGeoTranslation *other);
   virtual TGeoMatrix&  Inverse() const;
   virtual void         LocalToMaster(const Double_t *local, Double_t *master) const;
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master) const;
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master) const;
   virtual TGeoMatrix  *MakeClone() const;
   virtual void         MasterToLocal(const Double_t *master, Double_t *local) const;
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local) const;
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local) const;
   virtual void         RotateX(Double_t angle);
   virtual void         RotateY(Double_t angle);
   virtual void         RotateZ(Double_t angle);
   virtual void         SavePrimitive(std::ostream &out, Option_t *option = "");
   void                 Subtract(const TGeoTranslation *other);
   void                 SetTranslation(Double_t dx, Double_t dy, Double_t dz);
   void                 SetTranslation(const TGeoMatrix &other);
   virtual void         SetDx(Double_t dx) {SetTranslation(dx, fTranslation[1], fTranslation[2]);}
   virtual void         SetDy(Double_t dy) {SetTranslation(fTranslation[0], dy, fTranslation[2]);}
   virtual void         SetDz(Double_t dz) {SetTranslation(fTranslation[0], fTranslation[1], dz);}
   
   virtual const Double_t    *GetTranslation() const {return &fTranslation[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &kIdentityMatrix[0];}
   virtual const Double_t    *GetScale()       const {return &kUnitScale[0];}

   ClassDef(TGeoTranslation, 1)                 // translation class
};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoRotation - class describing rotations. A rotation is a 3*3 array   //
//    Column vectors has to be orthogonal unit vectors.                   //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoRotation : public TGeoMatrix
{
protected:
   Double_t             fRotationMatrix[3*3];   // rotation matrix

   void                 CheckMatrix();
public :
   TGeoRotation();
   TGeoRotation(const TGeoRotation &other);
   TGeoRotation(const TGeoMatrix &other);
   TGeoRotation(const char *name);
//   TGeoRotation(const char *name, Double_t *matrix) ;
   TGeoRotation(const char *name, Double_t phi, Double_t theta, Double_t psi);
   TGeoRotation(const char *name, Double_t theta1, Double_t phi1, Double_t theta2, Double_t phi2,
                Double_t theta3, Double_t phi3);
   virtual ~TGeoRotation() {}
   
   TGeoRotation& operator=(const TGeoMatrix &matrix);
   TGeoRotation& operator=(const TGeoRotation &other) {return operator=((const TGeoMatrix&)other);};
   
   Bool_t               IsValid() const;
   virtual TGeoMatrix&  Inverse() const;
   void                 Clear(Option_t *option ="");
   Double_t             Determinant() const;
   void                 FastRotZ(const Double_t *sincos);
   void                 GetAngles(Double_t &theta1, Double_t &phi1, Double_t &theta2, Double_t &phi2,
                                  Double_t &theta3, Double_t &phi3) const;
   void                 GetAngles(Double_t &phi, Double_t &theta, Double_t &psi) const;
   Double_t             GetPhiRotation(Bool_t fixX=kFALSE) const;
   virtual void         LocalToMaster(const Double_t *local, Double_t *master) const;
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master) const {TGeoRotation::LocalToMaster(local, master);}
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master) const {TGeoRotation::LocalToMaster(local, master);}
   virtual TGeoMatrix  *MakeClone() const;
   virtual void         MasterToLocal(const Double_t *master, Double_t *local) const;
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local) const {TGeoRotation::MasterToLocal(master, local);}
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local) const {TGeoRotation::MasterToLocal(master, local);}
   void                 MultiplyBy(TGeoRotation *rot, Bool_t after=kTRUE);
   virtual void         RotateX(Double_t angle);
   virtual void         RotateY(Double_t angle);
   virtual void         RotateZ(Double_t angle);
   virtual void         SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void         ReflectX(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         ReflectY(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         ReflectZ(Bool_t leftside, Bool_t rotonly=kFALSE);
   void                 SetAngles(Double_t phi, Double_t theta, Double_t psi);
   void                 SetAngles(Double_t theta1, Double_t phi1, Double_t theta2, Double_t phi2,
                                  Double_t theta3, Double_t phi3);
   void                 SetMatrix(const Double_t *rot) {memcpy(&fRotationMatrix[0], rot, 9*sizeof(Double_t));CheckMatrix();}
   void                 SetRotation(const TGeoMatrix &other);
   void                 GetInverse(Double_t *invmat) const;
   
   virtual const Double_t    *GetTranslation()    const {return &kNullVector[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &fRotationMatrix[0];}
   virtual const Double_t    *GetScale()          const {return &kUnitScale[0];}

   ClassDef(TGeoRotation, 1)               // rotation class
};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoScale - class describing scale transformations. A scale is an      //
//    array of 3 doubles (sx, sy, sz) multiplying elements 0, 5 and 10    //
//    of the homogenous matrix. A scale is normalized : sx*sy*sz = 1      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoScale : public TGeoMatrix
{
protected:
   Double_t             fScale[3];        // scale (x, y, z)
public :
   TGeoScale();
   TGeoScale(const TGeoScale &other);
   TGeoScale(Double_t sx, Double_t sy, Double_t sz);
   TGeoScale(const char *name, Double_t sx, Double_t sy, Double_t sz);
   virtual ~TGeoScale();
   
   virtual TGeoMatrix&  Inverse() const;
   void                 SetScale(Double_t sx, Double_t sy, Double_t sz);
   virtual void         LocalToMaster(const Double_t *local, Double_t *master) const;
   Double_t             LocalToMaster(Double_t dist, const Double_t *dir=0) const;
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master) const {TGeoScale::LocalToMaster(local, master);}
   virtual TGeoMatrix  *MakeClone() const;
   virtual void         MasterToLocal(const Double_t *master, Double_t *local) const;
   Double_t             MasterToLocal(Double_t dist, const Double_t *dir=0) const;
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local) const {TGeoScale::MasterToLocal(master, local);}
   virtual void         ReflectX(Bool_t, Bool_t) {fScale[0]=-fScale[0]; SetBit(kGeoReflection, !IsReflection());}
   virtual void         ReflectY(Bool_t, Bool_t) {fScale[1]=-fScale[1]; SetBit(kGeoReflection, !IsReflection());}
   virtual void         ReflectZ(Bool_t, Bool_t) {fScale[2]=-fScale[2]; SetBit(kGeoReflection, !IsReflection());}
   
   virtual const Double_t    *GetTranslation()    const {return &kNullVector[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &kIdentityMatrix[0];}
   virtual const Double_t    *GetScale()          const {return &fScale[0];}

   ClassDef(TGeoScale, 1)                 // scaling class
};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoCombiTrans - class describing rotation + translation. Most         //
//    frequently used in the description of TGeoNode 's                   //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoCombiTrans : public TGeoMatrix
{
protected:
   Double_t             fTranslation[3]; // translation vector
   TGeoRotation        *fRotation;       // rotation matrix
public :
   TGeoCombiTrans();
   TGeoCombiTrans(const TGeoCombiTrans &other);
   TGeoCombiTrans(const TGeoMatrix &other);
   TGeoCombiTrans(const TGeoTranslation &tr, const TGeoRotation &rot);
   TGeoCombiTrans(const char *name);
   TGeoCombiTrans(Double_t dx, Double_t dy, Double_t dz, TGeoRotation *rot);
   TGeoCombiTrans(const char *name, Double_t dx, Double_t dy, Double_t dz, TGeoRotation *rot);

   TGeoCombiTrans& operator=(const TGeoMatrix &matrix);
   TGeoCombiTrans& operator=(const TGeoCombiTrans &other) {return operator=((const TGeoMatrix&)other);};

   virtual ~TGeoCombiTrans();
   
   void                 Clear(Option_t *option ="");
   virtual TGeoMatrix&  Inverse() const;
   virtual TGeoMatrix  *MakeClone() const;
   virtual void         RegisterYourself();
   virtual void         RotateX(Double_t angle);
   virtual void         RotateY(Double_t angle);
   virtual void         RotateZ(Double_t angle);
   virtual void         ReflectX(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         ReflectY(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         ReflectZ(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void         SetDx(Double_t dx) {SetTranslation(dx, fTranslation[1], fTranslation[2]);}
   virtual void         SetDy(Double_t dy) {SetTranslation(fTranslation[0], dy, fTranslation[2]);}
   virtual void         SetDz(Double_t dz) {SetTranslation(fTranslation[0], fTranslation[1], dz);}
   void                 SetTranslation(const TGeoTranslation &tr);
   void                 SetTranslation(Double_t dx, Double_t dy, Double_t dz);
   void                 SetTranslation(Double_t *vect);
   void                 SetRotation(const TGeoRotation &other);
   void                 SetRotation(const TGeoRotation *rot);

   TGeoRotation              *GetRotation() const    {return fRotation;}

   virtual const Double_t    *GetTranslation()    const {return &fTranslation[0];}
   virtual const Double_t    *GetRotationMatrix() const;
   virtual const Double_t    *GetScale()          const {return &kUnitScale[0];}

   ClassDef(TGeoCombiTrans, 1)            // rotation + translation
};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoGenTrans - most general transformation, holding a translation,     //
//    a rotation and a scale                                              //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoGenTrans : public TGeoCombiTrans
{
protected:
   Double_t             fScale[3];       // scale (x, y, z)
public :
   TGeoGenTrans();
   TGeoGenTrans(const char *name);
   TGeoGenTrans(Double_t dx, Double_t dy, Double_t dz,
                Double_t sx, Double_t sy, Double_t sz, TGeoRotation *rot);
   TGeoGenTrans(const char *name, Double_t dx, Double_t dy, Double_t dz,
                Double_t sx, Double_t sy, Double_t sz, TGeoRotation *rot);
   virtual ~TGeoGenTrans();
   
   void                 Clear(Option_t *option ="");
   virtual TGeoMatrix&  Inverse() const;
   void                 SetScale(Double_t sx, Double_t sy, Double_t sz);
   void                 SetScale(Double_t *scale) {memcpy(&fScale[0], scale, 3*sizeof(Double_t));}
   virtual TGeoMatrix  *MakeClone() const {return NULL;}
   Bool_t               Normalize();

   virtual const Double_t    *GetScale()     const {return &fScale[0];}

   ClassDef(TGeoGenTrans, 1)            // rotation + translation + scale
};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoIdentity - an identity transformation. It holds no data member     //
//    and returns pointers to static null translation and identity        //
//    transformations for rotation and scale                              //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoIdentity : public TGeoMatrix
{
private:
   // no data members
public :
   TGeoIdentity();
   TGeoIdentity(const char *name);
   virtual ~TGeoIdentity() {}
   
   virtual TGeoMatrix&  Inverse() const;
   virtual void         LocalToMaster(const Double_t *local, Double_t *master) const {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master) const {memcpy(master, local, 3*sizeof(Double_t));}
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master) const {TGeoIdentity::LocalToMaster(local, master);}
   virtual TGeoMatrix  *MakeClone() const {return NULL;}
   virtual void         MasterToLocal(const Double_t *master, Double_t *local) const {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local) const {memcpy(local, master, 3*sizeof(Double_t));}
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local) const {TGeoIdentity::MasterToLocal(master, local);}

   virtual const Double_t    *GetTranslation() const {return &kNullVector[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &kIdentityMatrix[0];}
   virtual const Double_t    *GetScale()       const {return &kUnitScale[0];}
   virtual void         SavePrimitive(std::ostream &, Option_t * = "") {;}

   ClassDef(TGeoIdentity, 1)                 // identity transformation class
};



////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoHMatrix - Matrix class used for computing global transformations   //
//     Should NOT be used for node definition. An instance of this class  //
//     is generally used to pile-up local transformations starting from   //
//     the top level physical node, down to the current node.             //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoHMatrix : public TGeoMatrix
{
private:
   Double_t              fTranslation[3];    // translation component
   Double_t              fRotationMatrix[9]; // rotation matrix
   Double_t              fScale[3];          // scale component
   
public :
   TGeoHMatrix();
   TGeoHMatrix(const TGeoMatrix &matrix);
   TGeoHMatrix(const char *name);
   virtual ~TGeoHMatrix();
   
   TGeoHMatrix& operator=(const TGeoMatrix *matrix);
   TGeoHMatrix& operator=(const TGeoMatrix &matrix);
   TGeoHMatrix& operator=(const TGeoHMatrix &other) {return operator=((const TGeoMatrix&)other);};
   
   TGeoHMatrix& operator*=(const TGeoMatrix &matrix) {Multiply(&matrix);return(*this);}

   void                 Clear(Option_t *option ="");
   void                 CopyFrom(const TGeoMatrix *other);
   Double_t             Determinant() const;
   void                 FastRotZ(const Double_t *sincos);
   virtual TGeoMatrix&  Inverse() const;
   virtual TGeoMatrix  *MakeClone() const;
   void                 Multiply(const TGeoMatrix *right);
   void                 MultiplyLeft(const TGeoMatrix *left);

   virtual void         RotateX(Double_t angle);
   virtual void         RotateY(Double_t angle);
   virtual void         RotateZ(Double_t angle);
   virtual void         ReflectX(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         ReflectY(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         ReflectZ(Bool_t leftside, Bool_t rotonly=kFALSE);
   virtual void         SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void         SetDx(Double_t dx) {fTranslation[0] = dx; SetBit(kGeoTranslation);}
   virtual void         SetDy(Double_t dy) {fTranslation[1] = dy; SetBit(kGeoTranslation);}
   virtual void         SetDz(Double_t dz) {fTranslation[2] = dz; SetBit(kGeoTranslation);}
   void                 SetTranslation(const Double_t *vect) {SetBit(kGeoTranslation); memcpy(&fTranslation[0], vect, 3*sizeof(Double_t));}
   void                 SetRotation(const Double_t *matrix) {SetBit(kGeoRotation); memcpy(&fRotationMatrix[0], matrix, 9*sizeof(Double_t));}
   void                 SetScale(const Double_t *scale) {SetBit(kGeoScale); memcpy(&fScale[0], scale, 3*sizeof(Double_t));}


   virtual const Double_t    *GetTranslation() const {return &fTranslation[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &fRotationMatrix[0];}
   virtual const Double_t    *GetScale() const {return &fScale[0];}

   virtual Double_t    *GetTranslation() {return &fTranslation[0];}
   virtual Double_t    *GetRotationMatrix() {return &fRotationMatrix[0];}
   virtual Double_t    *GetScale() {return &fScale[0];}
   ClassDef(TGeoHMatrix, 1)                 // global matrix class
};


R__EXTERN TGeoIdentity *gGeoIdentity;

#endif

