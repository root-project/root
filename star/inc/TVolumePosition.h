// @(#)root/star:$Name:  $:$Id: TVolumePosition.h,v 1.1.1.1 2000/05/16 17:00:49 rdm Exp $
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

#include "TClass.h"
#include "TVolume.h"

class TBrowser;
class TRotMatrix;

class TVolumePosition  : public TObject {
 protected:
   Double_t        fX[3];        //X offset with respect to parent object
   TRotMatrix     *fMatrix;      //Pointer to rotation matrix
   TVolume        *fNode;        //Refs pointer to the node defined
   UInt_t          fId;          // Unique ID of this position

 public:
        TVolumePosition(TVolume *node=0,Double_t x=0, Double_t y=0, Double_t z=0, TRotMatrix *matrix=0);
        TVolumePosition(TVolume *node,Double_t x, Double_t y, Double_t z, const Text_t *matrixname);
        TVolumePosition(const TVolumePosition&pos): fMatrix(pos.GetMatrix()),fNode(pos.GetNode()),fId(pos.GetId())
                                                    {for (int i=0;i<3;i++) fX[i] = pos.GetX(i);}
        virtual ~TVolumePosition(){;}
        virtual void        Browse(TBrowser *b);
        virtual Float_t    *Errmx2Master(const Float_t *localError, Float_t *masterError);
        virtual Double_t   *Errmx2Master(const Double_t *localError, Double_t *masterError);
        virtual Double_t   *Cormx2Master(const Double_t *localCorr, Double_t *masterCorr);
        virtual Float_t    *Cormx2Master(const Float_t *localCorr, Float_t *masterCorr);
        virtual Int_t       DistancetoPrimitive(Int_t px, Int_t py);
        virtual TDataSet *DefineSet();
        virtual void        Draw(Option_t *depth="3"); // *MENU*
        virtual void        ExecuteEvent(Int_t event, Int_t px, Int_t py);
        virtual TRotMatrix  *GetMatrix() const {return fMatrix;}
        virtual TVolume     *GetNode() const {return fNode;}
        virtual Text_t      *GetObjectInfo(Int_t px, Int_t py);
        const   Option_t    *GetOption() const { return GetNode()?GetNode()->GetOption():0;}
        virtual const Char_t *GetName() const { return GetNode() ? GetNode()->GetName():IsA()->GetName();}
        Int_t               GetVisibility() const {return GetNode()?GetNode()->GetVisibility():0;}
        virtual Double_t    GetX(Int_t indx=0) const {return fX[indx];}
        virtual Double_t    GetY() const {return fX[1];}
        virtual Double_t    GetZ() const {return fX[2];}
        virtual UInt_t      GetId() const {return fId;}
        Bool_t              IsFolder() const {return GetNode()?kTRUE:kFALSE;}
        virtual Bool_t      Is3D()  {return kTRUE;}
        virtual Double_t   *Local2Master(const Double_t *local, Double_t *master,Int_t nPoints=1);
        virtual Float_t    *Local2Master(const Float_t *local, Float_t *master,Int_t nPoints=1);
        virtual void        Paint(Option_t *option="");
        virtual void        Print(Option_t *option="");
        virtual void        UpdatePosition(Option_t *option="");
        virtual TVolumePosition *Reset(TVolume *node=0,Double_t x=0, Double_t y=0, Double_t z=0, TRotMatrix *matrix=0);
        virtual void        SavePrimitive(ofstream &out, Option_t *option);
        virtual void        SetLineAttributes(); // *MENU*
        virtual void        SetMatrix(TRotMatrix *matrix=0) {fMatrix = matrix;}
        virtual void        SetNode(TVolume *node){ fNode = node;}
        virtual void        SetPosition( Double_t x=0, Double_t y=0, Double_t z=0) {fX[0]=x; fX[1]=y; fX[2]=z;}
        virtual void        SetVisibility(Int_t vis=1); // *MENU*
        virtual void        SetX(Double_t x){ fX[0]  =  x;}
        virtual void        SetY(Double_t y){ fX[1]  =  y;}
        virtual void        SetZ(Double_t z){ fX[2]  =  z;}
        virtual void        SetId(UInt_t id){fId  = id;}

        ClassDef(TVolumePosition,1)  //Description of parameters to position a 3-D geometry object
};

inline TDataSet *TVolumePosition::DefineSet(){ return GetNode(); }

#endif
