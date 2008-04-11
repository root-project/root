// @(#)root/g3d:$Id$
// Author: Rene Brun   22/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeometry
#define ROOT_TGeometry


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGeometry                                                            //
//                                                                      //
// Structure for Matrices, Shapes and Nodes.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_THashList
#include "THashList.h"
#endif

const Int_t kMAXLEVELS = 20;
const Int_t kVectorSize = 3;
const Int_t kMatrixSize = kVectorSize*kVectorSize;

class TNode;
class TBrowser;
class TMaterial;
class TRotMatrix;
class TShape;
class TObjArray;


class TGeometry : public TNamed {

private:
   THashList        *fMaterials;          //->Collection of materials
   THashList        *fMatrices;           //->Collection of rotation matrices
   THashList        *fShapes;             //->Collection of shapes
   TList            *fNodes;              //->Collection of nodes
   TRotMatrix       *fMatrix;             //!Pointers to current rotation matrices
   TNode            *fCurrentNode;        //!Pointer to current node
   TMaterial       **fMaterialPointer;    //!Pointers to materials
   TRotMatrix      **fMatrixPointer;      //!Pointers to rotation matrices
   TShape          **fShapePointer;       //!Pointers to shapes
   Float_t          fBomb;                //Bomb factor for exploded geometry
   Int_t            fGeomLevel;           //!
   Double_t         fX;                   //!
   Double_t         fY;                   //! The global translation of the current node
   Double_t         fZ;                   //!
   Double_t         fTranslation[kMAXLEVELS][kVectorSize];//!
   Double_t         fRotMatrix[kMAXLEVELS][kMatrixSize];  //!
   Bool_t           fIsReflection[kMAXLEVELS];            //!

protected:
   TGeometry(const TGeometry&);
   TGeometry& operator=(const TGeometry&);

public:
   TGeometry();
   TGeometry(const char *name, const char *title);
   virtual           ~TGeometry();
   virtual void      Browse(TBrowser *b);
   virtual void      cd(const char *path=0);
   virtual void      Draw(Option_t *option="");
   virtual TObject  *FindObject(const char *name) const;
   virtual TObject  *FindObject(const TObject *obj) const;
   Float_t           GetBomb() const {return fBomb;}
   Int_t             GeomLevel() const {return fGeomLevel;}
   THashList        *GetListOfShapes() const  {return fShapes;}
   TList            *GetListOfNodes()  const   {return fNodes;}
   THashList        *GetListOfMaterials() const {return fMaterials;}
   THashList        *GetListOfMatrices() const {return fMatrices;}
   TNode            *GetCurrentNode()  const {return fCurrentNode;}
   TMaterial        *GetMaterial(const char *name) const;
   TMaterial        *GetMaterialByNumber(Int_t number) const;
   TNode            *GetNode(const char *name) const;
   TShape           *GetShape(const char *name) const;
   TShape           *GetShapeByNumber(Int_t number) const;
   TRotMatrix       *GetRotMatrix(const char *name) const;
   TRotMatrix       *GetRotMatrixByNumber(Int_t number) const;
   TRotMatrix       *GetCurrentMatrix() const;
   TRotMatrix       *GetCurrentPosition(Double_t *x,Double_t *y,Double_t *z) const;
   TRotMatrix       *GetCurrentPosition(Float_t *x,Float_t *y,Float_t *z) const;
   Bool_t            GetCurrentReflection() const;
   Bool_t            IsFolder() const {return kTRUE;}
   virtual void      Local2Master(Double_t *local, Double_t *master);
   virtual void      Local2Master(Float_t *local, Float_t *master);
   virtual void      ls(Option_t *option="rsn2") const;
   virtual void      Master2Local(Double_t *master, Double_t *local);
   virtual void      Master2Local(Float_t *master, Float_t *local);
   virtual void      Node(const char *name, const char *title, const char *shapename, Double_t x=0, Double_t y=0, Double_t z=0
                        , const char *matrixname="", Option_t *option="");
   virtual Int_t     PushLevel(){return fGeomLevel++;}
   virtual Int_t     PopLevel(){return fGeomLevel>0?fGeomLevel--:0;}
   virtual void      RecursiveRemove(TObject *obj);
   virtual void      SetBomb(Float_t bomb=1.4) {fBomb = bomb;}
   virtual void      SetCurrentNode(TNode *node) {fCurrentNode = node;}
   virtual void      SetGeomLevel(Int_t level=0){fGeomLevel=level;}
   virtual void      SetMatrix(TRotMatrix *matrix=0){fMatrix = matrix;}
   virtual void      SetPosition(TRotMatrix *matrix, Double_t x=0,Double_t y=0,Double_t z=0);
   virtual void      SetPosition(TRotMatrix *matrix, Float_t x,Float_t y,Float_t z);
   virtual void      SetPosition(Double_t x,Double_t y,Double_t z);
   virtual void      SetPosition(Float_t x,Float_t y,Float_t z);
   virtual void      UpdateMatrix(TNode *node);
   virtual void      UpdateTempMatrix(Double_t x=0, Double_t y=0, Double_t z=0, TRotMatrix *matrix=0);
   virtual void      UpdateTempMatrix(Double_t x, Double_t y, Double_t z, Double_t *matrix,Bool_t isReflection=kFALSE);

   static TObjArray *Get(const char *name);
   static void       UpdateTempMatrix(Double_t *dx1,Double_t *rmat1,
                                      Double_t x, Double_t y, Double_t z, Double_t *matrix,
                                      Double_t *dxnew, Double_t *rmatnew);

   ClassDef(TGeometry,2)  //Structure for Matrices, Shapes and Nodes
};


inline TRotMatrix *TGeometry::GetCurrentMatrix() const
{
   return fMatrix;
}
inline TRotMatrix *TGeometry::GetCurrentPosition(Double_t *x,Double_t *y,Double_t *z) const
{
   *x = fX; *y = fY; *z = fZ; return GetCurrentMatrix();
}
inline TRotMatrix *TGeometry::GetCurrentPosition(Float_t *x,Float_t *y,Float_t *z) const
{
   *x = Float_t(fX); *y = Float_t(fY); *z = Float_t(fZ); return GetCurrentMatrix();
}
inline Bool_t TGeometry::GetCurrentReflection() const
{
   return fIsReflection[fGeomLevel];
}
inline void TGeometry::SetPosition(Double_t x,Double_t y,Double_t z)
{
   fX = x; fY = y; fZ = z;
}
inline void TGeometry::SetPosition(Float_t x,Float_t y,Float_t z)
{
   fX = x; fY = y; fZ = z;
}
inline void TGeometry::SetPosition(TRotMatrix *matrix, Double_t x,Double_t y,Double_t z)
{
   SetMatrix(matrix);
   SetPosition(x,y,z);
}
inline void TGeometry::SetPosition(TRotMatrix *matrix, Float_t x,Float_t y,Float_t z)
{
   SetMatrix(matrix);
   SetPosition(x,y,z);
}

R__EXTERN TGeometry *gGeometry;

#endif
