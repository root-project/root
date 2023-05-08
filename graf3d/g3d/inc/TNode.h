// @(#)root/g3d:$Id$
// Author: Rene Brun   14/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNode                                                                //
//                                                                      //
// Description of parameters to position a 3-D geometry object          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNode
#define ROOT_TNode

#include "TShape.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAtt3D.h"

#include "TRotMatrix.h"

class TBrowser;

class TNode : public TNamed , public TAttLine, public TAttFill, public TAtt3D {

protected:
   enum { kSonsInvisible = BIT(17) };

   Double_t        fX;          //X offset with respect to parent object
   Double_t        fY;          //Y offset with respect to parent object
   Double_t        fZ;          //Z offset with respect to parent object
   TRotMatrix     *fMatrix;     //Pointer to rotation matrix
   TShape         *fShape;      //Pointer to shape definition
   TNode          *fParent;     //Pointer to parent positioned volume
   TList          *fNodes;      //List of son nodes (if any)
   TString         fOption;     //List of options if any
   Int_t           fVisibility; //Visibility flag

   TNode(const TNode&);
   TNode& operator=(const TNode&);

public:
   TNode();
   TNode(const char *name, const char *title, const char *shapename, Double_t x=0, Double_t y=0, Double_t z=0,
         const char *matrixname="", Option_t *option="");
   TNode(const char *name, const char *title, TShape *shape, Double_t x=0, Double_t y=0, Double_t z=0,
         TRotMatrix *matrix=nullptr, Option_t *option="");
   virtual ~TNode();
   virtual void        Browse(TBrowser *b);
   virtual void        BuildListOfNodes();
   virtual void        cd(const char *path=nullptr); // *MENU*
   virtual Int_t       DistancetoPrimitive(Int_t px, Int_t py);
   virtual void        Draw(Option_t *option=""); // *MENU*
   virtual void        DrawOnly(Option_t *option="");
   virtual void        ExecuteEvent(Int_t event, Int_t px, Int_t py);
   TList              *GetListOfNodes() const {return fNodes;}
   virtual TRotMatrix *GetMatrix() const {return fMatrix;}
   virtual TNode      *GetNode(const char *name) const;
   virtual char       *GetObjectInfo(Int_t px, Int_t py) const;
   const   Option_t   *GetOption() const { return fOption.Data();}
   virtual TNode      *GetParent() const {return fParent;}
   TShape             *GetShape() const {return fShape;}
   Int_t               GetVisibility() const {return fVisibility;}
   virtual Double_t    GetX() const {return fX;}
   virtual Double_t    GetY() const {return fY;}
   virtual Double_t    GetZ() const {return fZ;}
   virtual void        ImportShapeAttributes();
   Bool_t              IsFolder() const;
   virtual void        Local2Master(const Double_t *local, Double_t *master);
   virtual void        Local2Master(const Float_t *local, Float_t *master);
   virtual void        ls(Option_t *option="2") const; // *MENU*
   virtual void        Master2Local(const Double_t *master, Double_t *local);
   virtual void        Master2Local(const Float_t *master, Float_t *local);
   virtual void        Paint(Option_t *option="");
   virtual void        RecursiveRemove(TObject *obj);
   virtual void        SetMatrix(TRotMatrix *matrix=nullptr) {fMatrix = matrix;}
   virtual void        SetName(const char *name);
   virtual void        SetParent(TNode *parent);
   virtual void        SetNameTitle(const char *name, const char *title);
   virtual void        SetPosition( Double_t x=0, Double_t y=0, Double_t z=0) {fX=x; fY=y; fZ=z;}
   virtual void        SetVisibility(Int_t vis=1); // *MENU*
   virtual void        Sizeof3D() const;
   virtual void        UpdateMatrix();
   virtual void        UpdateTempMatrix(const Double_t *dx1,const Double_t *rmat1,
                              Double_t x, Double_t y, Double_t z, Double_t *matrix,
                              Double_t *dxnew, Double_t *rmatnew);

   ClassDef(TNode,3)  //Description of parameters to position a 3-D geometry object
};

#endif
