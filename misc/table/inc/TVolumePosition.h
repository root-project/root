// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVolumePosition                                                      //
//                                                                      //
// Description of parameters to position a 3-D geometry object          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVolumePosition
#define ROOT_TVolumePosition

#include "TVolume.h"

class TBrowser;
class TRotMatrix;

class TVolumePosition  : public TObject {
protected:
   Double_t        fX[3];        //X offset with respect to parent object
   TRotMatrix     *fMatrix;      //Pointer to rotation matrix
   TVolume        *fNode;        //Refs pointer to the node defined
   UInt_t          fId;          // Unique ID of this position

protected:
   void DeleteOwnMatrix();

public:
   enum EPositionBits {
       kIsOwn      = BIT(23)   // if the TVolumePoistion doesn't own the TRotMatrix object
   };
   TVolumePosition(TVolume *node=0,Double_t x=0, Double_t y=0, Double_t z=0, TRotMatrix *matrix=0);
   TVolumePosition(TVolume *node,Double_t x, Double_t y, Double_t z, const char *matrixname);
   TVolumePosition(const TVolumePosition* oldPosition, const TVolumePosition* curPosition);
   TVolumePosition(const TVolumePosition&pos);
   virtual ~TVolumePosition();
   virtual void        Browse(TBrowser *b);
   virtual Float_t    *Errmx2Local (const Float_t *masterError, Float_t *localError  ) const;
   virtual Double_t   *Errmx2Local (const Double_t *masterError, Double_t *localError) const;
   virtual Float_t    *Errmx2Master(const Float_t *localError, Float_t *masterError  ) const;
   virtual Double_t   *Errmx2Master(const Double_t *localError, Double_t *masterError) const;
   virtual Double_t   *Cormx2Local (const Double_t *masterCorr, Double_t *localCorr  ) const;
   virtual Float_t    *Cormx2Local (const Float_t *masterCorr, Float_t *localCorr    ) const;
   virtual Double_t   *Cormx2Master(const Double_t *localCorr, Double_t *masterCorr  ) const;
   virtual Float_t    *Cormx2Master(const Float_t *localCorr, Float_t *masterCorr    ) const;
   virtual Double_t   *Master2Local(const Double_t *master, Double_t *local,Int_t nPoints=1) const;
   virtual Float_t    *Master2Local(const Float_t *master, Float_t *local,Int_t nPoints=1) const;

   virtual Int_t       DistancetoPrimitive(Int_t px, Int_t py);
   virtual TDataSet *DefineSet();
   virtual void        Draw(Option_t *depth="3"); // *MENU*
   virtual void        ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual TVolume     *GetNode() const {return fNode;}
   virtual char      *GetObjectInfo(Int_t px, Int_t py) const;
   const   Option_t    *GetOption() const { return GetNode()?GetNode()->GetOption():0;}
   virtual const Char_t *GetName() const;
   const TRotMatrix    *GetMatrix() const;
   TRotMatrix          *GetMatrix();

   Int_t               GetVisibility() const {return GetNode()?GetNode()->GetVisibility():0;}
   virtual Double_t    GetX(Int_t indx=0) const {return fX[indx];}
   virtual const Double_t *GetXYZ() const {return fX;}
   virtual Double_t    GetY() const {return fX[1];}
   virtual Double_t    GetZ() const {return fX[2];}
   virtual UInt_t      GetId() const {return fId;}
   Bool_t              IsMatrixOwner() const;
   Bool_t              SetMatrixOwner(Bool_t ownerShips=kTRUE);
   Bool_t              IsFolder() const {return GetNode()?kTRUE:kFALSE;}
   virtual Bool_t      Is3D() const {return kTRUE;}
   virtual Double_t   *Local2Master(const Double_t *local, Double_t *master,Int_t nPoints=1) const;
   virtual Float_t    *Local2Master(const Float_t *local, Float_t *master,Int_t nPoints=1) const;
   virtual TVolumePosition &Mult(const TVolumePosition &position);
   virtual void        Paint(Option_t *option="");
   virtual void        Print(Option_t *option="") const;
   virtual void        UpdatePosition(Option_t *option="");
   virtual TVolumePosition *Reset(TVolume *node=0,Double_t x=0, Double_t y=0, Double_t z=0, TRotMatrix *matrix=0);
   virtual void        SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void        SetLineAttributes(); // *MENU*
   virtual void        SetMatrix(TRotMatrix *matrix=0);
   virtual void        SetNode(TVolume *node){ fNode = node;}
   virtual void        SetPosition( Double_t x=0, Double_t y=0, Double_t z=0) {fX[0]=x; fX[1]=y; fX[2]=z;}
   virtual void        SetVisibility(Int_t vis=1); // *MENU*
   virtual void        SetX(Double_t x){ fX[0]  =  x;}
   virtual void        SetY(Double_t y){ fX[1]  =  y;}
   virtual void        SetZ(Double_t z){ fX[2]  =  z;}
   virtual void        SetXYZ(Double_t *xyz = 0);
   virtual void        SetId(UInt_t id){fId  = id;}
   TVolumePosition    &operator=(const TVolumePosition &rhs);
   ClassDef(TVolumePosition,2)  //Description of parameters to position a 3-D geometry object
};

//______________________________________________________________________________
inline TDataSet *TVolumePosition::DefineSet(){ return GetNode(); }
//______________________________________________________________________________
inline void TVolumePosition::DeleteOwnMatrix()
{
   if (IsMatrixOwner()) {
      TRotMatrix *erasing = fMatrix;
      fMatrix = 0;
      delete erasing;
   }
}
//______________________________________________________________________________
inline TRotMatrix *TVolumePosition::GetMatrix()
{   return fMatrix;                           }
//______________________________________________________________________________
inline const TRotMatrix *TVolumePosition::GetMatrix() const
{   return fMatrix;                                }
//______________________________________________________________________________
inline Bool_t TVolumePosition::SetMatrixOwner(Bool_t ownerShips)
{
   Bool_t currentOwner = IsMatrixOwner();
   SetBit(kIsOwn,ownerShips);
   return currentOwner;
}
//______________________________________________________________________________
inline Bool_t TVolumePosition::IsMatrixOwner() const
{
  // Check whether this object owns the TRotMatrix (to be deleted for example)
  // Note: This method is to be caleed from dtor.
  //       It is dangerous to make it virtual
   return TestBit(kIsOwn);
}
//______________________________________________________________________________
inline  TVolumePosition    &TVolumePosition::operator=(const TVolumePosition &rhs) {
   if (this != &rhs) {
      for (int i = 0; i < 3; i++) fX[i] = rhs.fX[i];
      fMatrix = rhs.fMatrix;
      fNode   = rhs.fNode;
      fId     = rhs.fId;
   }
   return *this;
}
//______________________________________________________________________________
std::ostream& operator<<(std::ostream& s,const TVolumePosition &target);
#endif
