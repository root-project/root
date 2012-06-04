// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVolumeView
#define ROOT_TVolumeView

#include <assert.h>

#include "TVolume.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVolumeView                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// typedef TVolumeView TNodeView;

class TVolumeView : public TObjectSet, public TAtt3D  {
protected:
   TList          *fListOfShapes;     //Pointer to the list of the "extra" shape definitions

   virtual void    PaintShape(Option_t *option);
   TVolumeView(TVolumeView &viewNode);


public:
   TVolumeView():TObjectSet(),fListOfShapes(0) {;}
   TVolumeView(TVolumeView *viewNode,TVolumePosition *nodePosition=0);
   TVolumeView(TVolumeView *viewNode,const Char_t *NodeName1,const Char_t *NodeName2=0);
   TVolumeView(TVolumeView *viewNode,TVolumeView *topNode);
   TVolumeView(TVolumeView *viewNode,const TVolumeView *node1,const TVolumeView *node2);
   TVolumeView(TVolume &pattern,Int_t maxDepLevel=0,const TVolumePosition *nodePosition=0,EDataSetPass iopt=kMarked,TVolumeView *root=0);
   TVolumeView(Double_t *translate, Double_t *rotate, UInt_t positionId, TVolume *thisNode,
               const Char_t *thisNodePath, const Char_t *matrixName=0, Int_t matrixType=0);
   TVolumeView(TVolume *thisNode,TVolumePosition *nodePosition);
   virtual ~TVolumeView();
   virtual TVolume *AddNode(TVolume *node);
   virtual void     Add(TDataSet *dataset);
   virtual void     Add(TVolumeView *node);
   virtual void     Add(TShape *shape, Bool_t IsMaster=kFALSE);
   virtual void     Browse(TBrowser *b);
   virtual void     Draw(Option_t *depth="3"); // *MENU*
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual TVolumePosition *GetPosition() const { return (TVolumePosition *)GetObject();}
   virtual TVolume *GetNode() const ;
   virtual Int_t    GetGlobalRange(const TVolumeView *rootNode,Float_t *min, Float_t *max);
   virtual TList   *GetListOfShapes() const;
   virtual void     GetLocalRange(Float_t *min, Float_t *max);
   virtual char  *GetObjectInfo(Int_t px, Int_t py) const;
   virtual TShape  *GetShape()  const;
   virtual Int_t    GetVisibility() const;
   virtual Bool_t   IsMarked() const;
   virtual Bool_t   Is3D() const {return kTRUE;}
   virtual TVolumePosition  *Local2Master(const TVolumeView *localNode,const TVolumeView *masterNode=0);
   virtual TVolumePosition  *Local2Master(const Char_t *localName, const Char_t *masterName=0);
   virtual Float_t *Local2Master(const Float_t *local, Float_t *master,
                                 const Char_t *localName, const Char_t *masterName=0, Int_t nVector=1);
   virtual Float_t   *Local2Master(const Float_t *local, Float_t *master,
                                   const TVolumeView *localNode,
                                   const TVolumeView *masterNode=0, Int_t nVector=1);
   virtual TList   *Nodes(){ return GetList();}
   virtual void     Paint(Option_t *option="");
   virtual TString  PathP() const;
   virtual void     SetLineAttributes(); // *MENU*
   virtual void     SavePrimitive(std::ostream &out, Option_t *option="");
   virtual void     SetVisibility(Int_t vis=1); // *MENU*
   virtual void     Sizeof3D() const;
   ClassDef(TVolumeView,1)  // Special kind of TDataSet
};

inline void    TVolumeView::Add(TDataSet * /*dataset*/){ assert(0);}
inline void    TVolumeView::Add(TVolumeView *node){ TDataSet::Add(node);}
inline Bool_t  TVolumeView::IsMarked() const { return TestBit(kMark); }
inline TList  *TVolumeView::GetListOfShapes() const {return fListOfShapes;}
inline TShape *TVolumeView::GetShape()  const
       {return fListOfShapes ? (TShape *)fListOfShapes->First():0;}
inline Int_t   TVolumeView::GetVisibility() const {return GetNode() ? GetNode()->GetVisibility():0;}

#endif

