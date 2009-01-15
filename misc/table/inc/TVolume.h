// @(#)root/table:$Id$
// Author: Valery Fine   10/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVolume                                                              //
//                                                                      //
// Description of parameters to position a 3-D geometry object          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVolume
#define ROOT_TVolume

#include "TObjectSet.h"

#include "TNode.h"

#ifndef ROOT_TShape
#include "TShape.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

class TBrowser;
class TVolumePosition;
class TRotMatrix;
class TList;

class TVolume  : public TObjectSet, public TAttLine, public TAttFill, public TAtt3D {
public:
   enum ENodeSEEN {kBothVisible  = 00,                 //'00'
      kSonUnvisible =  1,                              //'01'
      kThisUnvisible=  2,                              //'10'
      kNoneVisible  = kThisUnvisible | kSonUnvisible}; //'11'
protected:
   TShape         *fShape;         //Pointer to the "master" shape definition
   TList          *fListOfShapes;  //Pointer to the list of the shape definitions
   TString         fOption;        //List of options if any
   ENodeSEEN       fVisibility;    //Visibility flag  00 - everything visible,
   //                 10 - this unvisible, but sons are visible
   //                 01 - this visible but sons
   //                 11 - neither this nor its sons are visible

   virtual void             Add(TDataSet *dataset);
   virtual void             Add(TVolumePosition *position);
   virtual TVolumePosition *Add(TVolume *node, TVolumePosition *nodePosition);
   virtual Int_t            DistancetoNodePrimitive(Int_t px, Int_t py,TVolumePosition *position=0);
   void             SetPositionsList(TList *list=0){AddObject((TObject *)list);}
   virtual void             PaintNodePosition(Option_t *option="",TVolumePosition *postion=0);
   friend class TPolyLineShape;
public:
   TVolume();
   TVolume(const char *name, const char *title, const char *shapename, Option_t *option="");
   TVolume(const char *name, const char *title, TShape *shape, Option_t *option="");
   TVolume(TNode &node);
   virtual ~TVolume();
   virtual TVolumePosition *Add(TVolume *node, Double_t x=0, Double_t y=0, Double_t z=0, TRotMatrix *matrix=0, UInt_t id=0, Option_t *option="");
   virtual TVolumePosition *Add(TVolume *node, Double_t x, Double_t y, Double_t z,  const char *matrixname,  UInt_t id=0, Option_t *option="");
   static  Int_t       MapStNode2GEANTVis(ENodeSEEN  vis);
   static  Int_t       MapGEANT2StNodeVis(Int_t vis);
   virtual void        Add(TShape *shape, Bool_t IsMaster=kFALSE);
   virtual void        Browse(TBrowser *b);
   virtual TNode      *CreateTNode(const TVolumePosition *position=0);
   virtual void        DeletePosition(TVolumePosition *position);
   virtual Int_t       DistancetoPrimitive(Int_t px, Int_t py);
   virtual void        Draw(Option_t *depth="3"); // *MENU*
   virtual void        DrawOnly(Option_t *option="");
   virtual void        ExecuteEvent(Int_t event, Int_t px, Int_t py);
   static  TRotMatrix *GetIdentity();
   virtual char     *GetObjectInfo(Int_t px, Int_t py) const;
   const   Option_t   *GetOption() const { return fOption.Data();}
   TShape     *GetShape()  const {return fShape;}
   TList      *GetListOfShapes()  const {return fListOfShapes;}
   virtual void        GetLocalRange(Float_t *min, Float_t *max);
   virtual ENodeSEEN   GetVisibility() const {return fVisibility;}
   virtual TList      *GetListOfPositions()  { return (TList *)(GetObject());}
   virtual ULong_t     Hash() const { return TObject::Hash();}
   virtual void        ImportShapeAttributes();
   virtual Bool_t      IsMarked() const;
   virtual Bool_t      Is3D() const {return kTRUE;}
   virtual TList      *Nodes() const { return GetList(); }
   virtual void        Paint(Option_t *option="");
   virtual void        PaintShape(Option_t *option="");
   virtual void        SetVisibility(ENodeSEEN vis=TVolume::kBothVisible); // *MENU*
   virtual void        Sizeof3D() const;

   ClassDef(TVolume,1)  //Description of parameters to position a 3-D geometry object
};

inline void   TVolume::Add(TDataSet *dataset){ TDataSet::Add(dataset);}
inline Bool_t TVolume::IsMarked() const { return TestBit(kMark); }

#endif
