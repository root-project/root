/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 09:06:36 AM CEST

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

enum EGeoTransfTypes {
   kGeoIdentity  = 0,
   kGeoTranslation  = BIT(17),
   kGeoRotation     = BIT(18),
   kGeoScale        = BIT(19),
   kGeoReflection   = BIT(20),
   kGeoRegistered   = BIT(21),
   kGeoCombiTrans   = kGeoTranslation | kGeoRotation,
   kGeoGenTrans     = kGeoTranslation | kGeoRotation | kGeoScale
};

//--- globals 
const Double_t kNullVector[3]       =       {0.0,  0.0,  0.0};

const Double_t kIdentityMatrix[3*3] =       {1.0,  0.0,  0.0,
                                             0.0,  1.0,  0.0,
                                             0.0,  0.0,  1.0};

const Double_t kUnitScale[3]        =       {1.0,  1.0,  1.0};

/*************************************************************************
 * TGeoMatrix - base class for geometrical transformations.              *
 * 
 * 
 *************************************************************************/

class TGeoMatrix : public TNamed
{
protected:
   void                 SetDefaultName();
public :
   TGeoMatrix();
   TGeoMatrix(const char *name);
   virtual ~TGeoMatrix() {}
   
   Bool_t               IsIdentity()    const {return ((!TestBit(kGeoTranslation))
                            && (!TestBit(kGeoRotation)) && (!TestBit(kGeoScale)));}
   Bool_t               IsTranslation() const {return TestBit(kGeoTranslation);}
   Bool_t               IsRotation()    const {return TestBit(kGeoRotation);}
   Bool_t               IsScale()       const {return TestBit(kGeoScale);}
   Bool_t               IsCombi()       const {return (TestBit(kGeoTranslation) 
                                               && TestBit(kGeoRotation));}
   Bool_t               IsGeneral()     const {return (TestBit(kGeoTranslation) 
                            && TestBit(kGeoRotation) && TestBit(kGeoScale));}
   Bool_t               IsRegistered()  const {return TestBit(kGeoRegistered);}
   void                 GetHomogenousMatrix(Double_t *hmat);

   virtual Int_t              GetByteCount();
   virtual const Double_t    *GetTranslation()    const = 0;
   virtual const Double_t    *GetRotationMatrix() const = 0;
   virtual const Double_t    *GetScale()          const = 0;
   virtual void         LocalToMaster(const Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master);
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master);
   virtual void         MasterToLocal(const Double_t *master, Double_t *local);
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local);
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local);
   void                 Print();
   
  ClassDef(TGeoMatrix, 0)                 // base geometrical transformation class
};



/*************************************************************************
 * TGeoTranslation - class describing translations. A translation is     *
 *    basicaly an array of 3 doubles matching the positions 12, 13       *
 *    and 14 in the homogenous matrix description.                       *
 *                                                                       *
 *************************************************************************/

class TGeoTranslation : public TGeoMatrix
{
protected:
   Double_t             fTranslation[3];  // translation vector
public :
   TGeoTranslation();
   TGeoTranslation(Double_t dx, Double_t dy, Double_t dz);
   virtual ~TGeoTranslation() {}
   
   void                 Add(TGeoTranslation *other);
   virtual void         LocalToMaster(const Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master);
   virtual void         MasterToLocal(const Double_t *master, Double_t *local);
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local);
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master);
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local);
   void                 Subtract(TGeoTranslation *other);
   void                 SetTranslation(Double_t dx, Double_t dy, Double_t dz);
   void                 SetDx(Double_t dx) {fTranslation[0]=dx;}
   void                 SetDy(Double_t dy) {fTranslation[1]=dy;}
   void                 SetDz(Double_t dz) {fTranslation[2]=dz;}
   
   virtual const Double_t    *GetTranslation() const {return &fTranslation[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &kIdentityMatrix[0];}
   virtual const Double_t    *GetScale()       const {return &kUnitScale[0];}

  ClassDef(TGeoTranslation, 1)                 // translation class
};

/*************************************************************************
 * TGeoRotation - class describing rotations. A rotation is a 3*3 array  *
 *    Column vectors has to be orthogonal unit vectors. 
 * 
 *************************************************************************/

class TGeoRotation : public TGeoMatrix
{
protected:
   Double_t             fRotationMatrix[3*3];   // rotation matrix

   void                 CheckMatrix();
public :
   TGeoRotation();
   TGeoRotation(const char *name);
//   TGeoRotation(const char *name, Double_t *matrix) ;
   TGeoRotation(const char *name, Double_t alpha, Double_t beta, Double_t gamma);
   TGeoRotation(const char *name, Double_t theta1, Double_t phi1, Double_t theta2, Double_t phi2,
                Double_t theta3, Double_t phi3);
   virtual ~TGeoRotation() {}
   
   Bool_t               IsReflection()  {return TestBit(kGeoReflection);}
   void                 Clear();
   Double_t             Determinant();
   void                 FastRotZ(Double_t *sincos);
   virtual void         LocalToMaster(const Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master)
                          {TGeoRotation::LocalToMaster(local, master);}
   virtual void         MasterToLocal(const Double_t *master, Double_t *local);
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local)
                          {TGeoRotation::MasterToLocal(master, local);}
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master)
                          {TGeoRotation::LocalToMaster(local, master);}
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local)
                          {TGeoRotation::MasterToLocal(master, local);}
   void                 MultiplyBy(TGeoRotation *rot, Bool_t after=kTRUE);
   void                 SetAngles(Double_t alpha, Double_t beta, Double_t gamma);
   void                 SetAngles(Double_t theta1, Double_t phi1, Double_t theta2, Double_t phi2,
                                  Double_t theta3, Double_t phi3);
   void                 SetMatrix(Double_t *rot) 
                           {memcpy(&fRotationMatrix[0], rot, 9*sizeof(Double_t));}
   void                 GetInverse(Double_t *invmat);
   
   virtual const Double_t    *GetTranslation()    const {return &kNullVector[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &fRotationMatrix[0];}
   virtual const Double_t    *GetScale()          const {return &kUnitScale[0];}

  ClassDef(TGeoRotation, 1)               // rotation class
};

/*************************************************************************
 * TGeoScale - class describing scale transformations. A scale is an     *
 *    array of 3 doubles (sx, sy, sz) multiplying elements 0, 5 and 10
 *    of the homogenous matrix. A scale is normalized : sx*sy*sz = 1
 *************************************************************************/

class TGeoScale : public TGeoMatrix
{
protected:
   Double_t             fScale[3];        // scale (x, y, z)
public :
   TGeoScale();
   TGeoScale(Double_t sx, Double_t sy, Double_t sz);
   virtual ~TGeoScale();
   
   void                       SetScale(Double_t sx, Double_t sy, Double_t sz);
   Bool_t                     Normalize();
   
   virtual const Double_t    *GetTranslation()    const {return &kNullVector[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &kIdentityMatrix[0];}
   virtual const Double_t    *GetScale()          const {return &fScale[0];}

  ClassDef(TGeoScale, 1)                 // scaling class
};

/*************************************************************************
 * TGeoCombiTrans - class describing rotation + translation. Most        *
 *    frequently used in the description of TGeoNode 's
 * 
 *************************************************************************/

class TGeoCombiTrans : public TGeoMatrix
{
protected:
   Double_t             fTranslation[3]; // translation vector
   TGeoRotation        *fRotation;       // rotation matrix
public :
   TGeoCombiTrans();
   TGeoCombiTrans(const char *name);
   TGeoCombiTrans(Double_t dx, Double_t dy, Double_t dz, TGeoRotation *rot);

   virtual ~TGeoCombiTrans();
   
   void                 SetTranslation(Double_t dx, Double_t dy, Double_t dz);
   void                 SetTranslation(Double_t *vect);
   void                 SetRotation(TGeoRotation *rot) {fRotation = rot;}

   TGeoRotation              *GetRotation()    {return fRotation;}

   virtual const Double_t    *GetTranslation()    const {return &fTranslation[0];}
   virtual const Double_t    *GetRotationMatrix() const;
   virtual const Double_t    *GetScale()          const {return &kUnitScale[0];}

  ClassDef(TGeoCombiTrans, 1)            // rotation + translation
};

/*************************************************************************
 * TGeoGenTrans - most general transformation, holding a translation,    *
 *    a rotation and a scale
 * 
 *************************************************************************/

class TGeoGenTrans : public TGeoCombiTrans
{
protected:
   Double_t             fScale[3];       // scale (x, y, z)
public :
   TGeoGenTrans();
   TGeoGenTrans(const char *name);
   TGeoGenTrans(Double_t dx, Double_t dy, Double_t dz,
                  Double_t sx, Double_t sy, Double_t sz, TGeoRotation *rot);
   virtual ~TGeoGenTrans();
   
   void                 Clear();
   void                 SetScale(Double_t sx, Double_t sy, Double_t sz);
   void                 SetScale(Double_t *scale)
                           {memcpy(&fScale[0], scale, 3*sizeof(Double_t));}
   Bool_t               Normalize();

   virtual const Double_t    *GetScale()     const {return &fScale[0];}

  ClassDef(TGeoGenTrans, 1)            // rotation + translation + scale
};

/*************************************************************************
 * TGeoIdentity - an identity transformation. It holds no data member    *
 *    and returns pointers to static null translation and identity       *
 *    transformations for rotation and scale                             *
 *                                                                       *
 *************************************************************************/

class TGeoIdentity : public TGeoMatrix
{
private:
   // no data members
public :
   TGeoIdentity();
   TGeoIdentity(const char *name);
   virtual ~TGeoIdentity() {}
   
   virtual void         LocalToMaster(const Double_t *local, Double_t *master);
   virtual void         LocalToMasterVect(const Double_t *local, Double_t *master)
                           {TGeoIdentity::LocalToMaster(local, master);}
   virtual void         MasterToLocal(const Double_t *master, Double_t *local);
   virtual void         MasterToLocalVect(const Double_t *master, Double_t *local)
                           {TGeoIdentity::MasterToLocal(master, local);}
   virtual void         LocalToMasterBomb(const Double_t *local, Double_t *master)
                           {TGeoIdentity::LocalToMaster(local, master);}
   virtual void         MasterToLocalBomb(const Double_t *master, Double_t *local)
                           {TGeoIdentity::MasterToLocal(master, local);}

   virtual const Double_t    *GetTranslation() const {return &kNullVector[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &kIdentityMatrix[0];}
   virtual const Double_t    *GetScale()       const {return &kUnitScale[0];}

  ClassDef(TGeoIdentity, 1)                 // identity transformation class
};



/*************************************************************************
 * TGeoHMatrix - Matrix class used for computing global transformations  *
 *     Should NOT be used for node definition. An instance of this class *
 *     is generally used to pile-up local transformations starting from  *
 *     the top level physical node, down to the current node.            *
 *************************************************************************/

class TGeoHMatrix : public TGeoMatrix
{
private:
   Double_t              fTranslation[3];    // translation component
   Double_t              fRotationMatrix[9]; // rotation matrix
   Double_t              fScale[3];          // scale component
   
public :
   TGeoHMatrix();
   TGeoHMatrix(const char *name);
   virtual ~TGeoHMatrix();
   
   TGeoHMatrix& operator=(const TGeoMatrix *matrix);

   void                       Clear();
   void                       Multiply(TGeoMatrix *right);

   void                       SetTranslation(const Double_t *vect) 
                                 {memcpy(&fTranslation[0], vect, 3*sizeof(Double_t));}
   void                       SetRotation(const Double_t *matrix)
                                 {memcpy(&fRotationMatrix[0], matrix, 9*sizeof(Double_t));}
   void                       SetScale(const Double_t *scale) 
                                 {memcpy(&fScale[0], scale, 3*sizeof(Double_t));}


   virtual const Double_t    *GetTranslation() const {return &fTranslation[0];}
   virtual const Double_t    *GetRotationMatrix() const {return &fRotationMatrix[0];}
   virtual const Double_t    *GetScale()       const {return &fScale[0];}

   virtual Double_t    *GetTranslation()  {return &fTranslation[0];}
   virtual Double_t    *GetRotationMatrix() {return &fRotationMatrix[0];}
   virtual Double_t    *GetScale()       {return &fScale[0];}

  ClassDef(TGeoHMatrix, 0)                 // global matrix class
};


R__EXTERN TGeoIdentity *gGeoIdentity;

#endif

