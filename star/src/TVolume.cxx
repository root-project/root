// @(#)root/star:$Name:  $:$Id: TVolume.cxx,v 1.5 2002/01/23 17:52:51 rdm Exp $
// Author: Valery Fine   10/12/98
//
/*************************************************************************
 * Copyright(c) 1998, FineSoft, All rights reserved. Valery Fine (Faine) *
 *************************************************************************/

#include <stdlib.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TClass.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TGeometry.h"
#include "TRotMatrix.h"
#include "TShape.h"
#include "TVolume.h"
#include "TBrowser.h"
#include "X3DBuffer.h"

#include "TPadView3D.h"
#include "TCanvas.h"

#include "TRotMatrix.h"
#include "TVolumePosition.h"

//const Int_t kMAXLEVELS = 20;
const Int_t kSonsInvisible = BIT(17);

#if 0
const Int_t kVectorSize = 3;
const Int_t kMatrixSize = kVectorSize*kVectorSize;

static Double_t gTranslation[kMAXLEVELS][kVectorSize];
static Double_t gRotMatrix[kMAXLEVELS][kMatrixSize];
static Int_t gGeomLevel = 0;

TVolume *gNode;
#endif
R__EXTERN  Size3D gSize3D;
static TRotMatrix *gIdentity = 0;

ClassImp(TVolume)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-* T V O L U M E  description *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                     ==========================
//*-*
//*-*    A TVolume object is used to build the geometry hierarchy.
//*-*    Since TVolume is derived from TDataSet class it may contain other volumes.
//*-*
//*-*    A geometry volume has attributes:
//*-*      - name and title
//*-*      - pointer to the referenced shape (see TShape).
//*-*      - list of TVolumePosition object defining the position of the nested volumes
//*-*        with respect to the mother node.
//*-*
//*-*
//*-*    A volume can be drawn.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


//______________________________________________________________________________
TVolume::TVolume()
{
//*-*-*-*-*-*-*-*-*-*-*Volume default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   fShape        = 0;
   fListOfShapes = 0;
   fVisibility   = kBothVisible;
   if (!gGeometry) new TGeometry;
}

//______________________________________________________________________________
TVolume::TVolume(const Text_t *name, const Text_t *title, const Text_t *shapename, Option_t *option)
       :TObjectSet(name),TAttLine(), TAttFill(),fShape(0),fListOfShapes(0)
{
//*-*-*-*-*-*-*-*-*-*-*Volume normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*
//*-*    name    is the name of the node
//*-*    title   is title
//*-*    shapename is the name of the referenced shape
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#ifdef WIN32
//*-* The color "1" - default produces a very bad 3D image with OpenGL
   Color_t lcolor = 16;
   SetLineColor(lcolor);
#endif
   static Int_t counter = 0;
   counter++;
   SetTitle(title);
   if(!(counter%1000))cout<<"TVolume count="<<counter<<" name="<<name<<endl;
   if (!gGeometry) new TGeometry;
   Add(gGeometry->GetShape(shapename),kTRUE);
//   fParent = gGeometry->GetCurrenTVolume();
   fOption = option;
   fVisibility = kBothVisible;

   if(!fShape) {Printf("Illegal referenced shape"); return;}
   ImportShapeAttributes();
}


//______________________________________________________________________________
TVolume::TVolume(const Text_t *name, const Text_t *title, TShape *shape, Option_t *option)
                :TObjectSet(name),TAttLine(),TAttFill(),fShape(0),fListOfShapes(0)
{
//*-*-*-*-*-*-*-*-*-*-*Volume normal constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================================
//*-*
//*-*    name    is the name of the node
//*-*    title   is title
//*-*    shape   is the pointer to the shape definition
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#ifdef WIN32
//*-* The color "1" - default produces a very bad 3D image with OpenGL
   Color_t lcolor = 16;
   SetLineColor(lcolor);
#endif

   if (!gGeometry) new TGeometry;
   Add(shape,kTRUE);
   fOption = option;
   fVisibility = kBothVisible;
   SetTitle(title);
   if(!shape) {Printf("Illegal referenced shape"); return;}
   ImportShapeAttributes();
}

//______________________________________________________________________________
Int_t TVolume::MapStNode2GEANTVis(ENodeSEEN  vis)
{
// ENodeSEEN Visibility flag  00 - everything visible,
//                            10 - this unvisible, but sons are visible
//                            01 - this visible but sons
//                            11 - neither this nor its sons are visible
// Maps the value of the visibility flag to begin_html <a href="http://wwwinfo.cern.ch/asdoc/geant_html3/node128.html#SECTION056000000000000000000000">GEANT 3.21 "volume attributes"</a>end_html
  const Int_t mapVis[4] = {1, -2, 0, -1 };
  return mapVis[vis];
}

//______________________________________________________________________________
//ENodeSEEN TVolume::MapGEANT2StNodeVis(Int_t vis)
Int_t TVolume::MapGEANT2StNodeVis(Int_t vis)
{
// Maps the value of begin_html <a href="http://wwwinfo.cern.ch/asdoc/geant_html3/node128.html#SECTION056000000000000000000000">GEANT 3.21 "volume attributes"</a>end_html to the visibility flag
  const Int_t mapVis[4] = {1, -2, 0, -1 };
  Int_t i;
//  for (i =0; i<3;i++) if (mapVis[i] == vis) return (ENodeSEEN)i;
  for (i =0; i<3;i++) if (mapVis[i] == vis) return i;
  return kBothVisible;
}

//______________________________________________________________________________
TVolume::TVolume(TNode &rootNode):fShape(0),fListOfShapes(0)
{
  // Convert a TNode object into a TVolume

  SetName(rootNode.GetName());
  SetTitle(rootNode.GetTitle());
  fVisibility = ENodeSEEN(MapGEANT2StNodeVis(rootNode.GetVisibility()));
  fOption     = rootNode.GetOption();
  Add(rootNode.GetShape(),kTRUE);

  SetLineColor(rootNode.GetLineColor());
  SetLineStyle(rootNode.GetLineStyle());
  SetLineWidth(rootNode.GetLineWidth());
  SetFillColor(rootNode.GetFillColor());
  SetFillStyle(rootNode.GetFillStyle());

  TList *nodes = rootNode.GetListOfNodes();
  if (nodes) {
    TIter next(nodes);
    TNode *node = 0;
    while ( (node = (TNode *) next()) ){
      TVolume *nextNode = new TVolume(*node);
      Add(nextNode,node->GetX(),node->GetY(),node->GetZ(),node->GetMatrix());
    }
  }
}

//______________________________________________________________________________
void TVolume::Add(TShape *shape, Bool_t IsMaster)
{
  if (!shape) return;
  if (!fListOfShapes) fListOfShapes = new TList;
  fListOfShapes->Add(shape);
  if (IsMaster) fShape = shape;
}

//______________________________________________________________________________
TNode *TVolume::CreateTNode(const TVolumePosition *position)
{
  // Convert a TVolume object into a TNode

  Double_t x=0;
  Double_t y=0;
  Double_t z=0;
  TRotMatrix* matrix = 0;
  if (position) {
     x=position->GetX();
     y=position->GetY();
     z=position->GetZ();
     matrix = position->GetMatrix();
  }
//  const Char_t  *path = Path();
//  printf("%s: %s/%s, shape=%s/%s\n",path,GetName(),GetTitle(),GetShape()->GetName(),GetShape()->ClassName());
  TNode *newNode  = new TNode(GetName(),GetTitle(),GetShape(),x,y,z,matrix,GetOption());
  newNode->SetVisibility(MapStNode2GEANTVis(GetVisibility()));

  newNode->SetLineColor(GetLineColor());
  newNode->SetLineStyle(GetLineStyle());
  newNode->SetLineWidth(GetLineWidth());
  newNode->SetFillColor(GetFillColor());
  newNode->SetFillStyle(GetFillStyle());

  TList *positions = GetListOfPositions();
  if (positions) {
    TIter next(positions);
    TVolumePosition *pos = 0;
    while ( (pos = (TVolumePosition *) next()) ){
      TVolume *node = pos->GetNode();
      if (node) {
          newNode->cd();
          node->CreateTNode(pos);
      }
    }
  }
  newNode->ImportShapeAttributes();
  return newNode;
}

//______________________________________________________________________________
TVolume::~TVolume()
{
//*-*-*-*-*-*-*-*-*-*-*Volume default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

   // Hmm, here we are in the troubles, in theory we have to find all
   // place where this node is sitting but we don't (yet :-()

   if (GetListOfPositions()) {
     GetListOfPositions()->Delete();
     SetPositionsList();
   }
   SafeDelete(fListOfShapes);
}

//______________________________________________________________________________
void TVolume::Add(TVolumePosition *position)
{
  if (!GetListOfPositions()) SetPositionsList(new TList);
  if ( GetListOfPositions()) GetListOfPositions()->Add(position);
  else Error("Add","Can not create list of positions for the current node <%s>:<%s>",GetName(),GetTitle());
}

//______________________________________________________________________________
TVolumePosition *TVolume::Add(TVolume *node, TVolumePosition *nodePosition)
{
  TVolumePosition *position = nodePosition;
  if (!node) return 0;
  if (!position) position = new TVolumePosition(node);  // Create default position
  TDataSet::Add(node);
  Add(position);
  return position;
}

//______________________________________________________________________________
TVolumePosition *TVolume::Add(TVolume *volume, Double_t x, Double_t y, Double_t z,
                              TRotMatrix *matrix,  UInt_t id, Option_t *)
{
//*-*
//*-*    volume  the pointer to the volume to be placed
//*-*    x,y,z   are the offsets of the volume with respect to his mother
//*-*    matrix  is the pointer to the rotation matrix
//*-*     id     is a unique position id
//*-*
 if (!volume) return 0;
 TRotMatrix *rotation = matrix;
 if(!rotation) rotation = GetIdentity();
 TVolumePosition *position = new TVolumePosition(volume,x,y,z,rotation);
 position->SetId(id);
 return Add(volume,position);
}

//______________________________________________________________________________
TVolumePosition *TVolume::Add(TVolume *volume, Double_t x, Double_t y, Double_t z,
                              const Text_t *matrixname,  UInt_t id, Option_t *)
{
//*-*
//*-*    volume      the pointer to the volume to be placed
//*-*    x,y,z       are the offsets of the volume with respect to his mother
//*-*    matrixname  is the name of the rotation matrix
//*-*     id         is a unique position id
//*-*
 if (!volume) return 0;
 TRotMatrix *rotation = 0;
 if (matrixname && strlen(matrixname)) rotation = gGeometry->GetRotMatrix(matrixname);
 if (!rotation)                        rotation = GetIdentity();
 TVolumePosition *position = new TVolumePosition(volume,x,y,z,rotation);
 position->SetId(id);
 return Add(volume,position);
}

//______________________________________________________________________________
void TVolume::Browse(TBrowser *b)
{
   if (GetListOfPositions()){
       TVolumePosition *nodePosition = 0;
       TIter next(GetListOfPositions());
       Int_t posNumber = 0;
       while ( (nodePosition = (TVolumePosition *)next()) ) {
         posNumber       = nodePosition->GetId();
         TString posName = "*";
         posName += nodePosition->GetNode()->GetTitle();
         char num[10];
         posName += ";";
         sprintf(num,"%d",posNumber);
         posName += num;
         b->Add(nodePosition,posName.Data());
       }
   }
   else {
#ifndef WIN32
      Inspect();
#endif
    }
}

//______________________________________________________________________________
Int_t TVolume::DistancetoPrimitive(Int_t px, Int_t py)
{
  return DistancetoNodePrimitive(px,py);
}

//______________________________________________________________________________
Int_t TVolume::DistancetoNodePrimitive(Int_t px, Int_t py,TVolumePosition *pos)
{
//*-*-*-*-*-*-*-*-*Compute distance from point px,py to a TVolumeView*-*-*-*-*-*
//*-*                  ===========================================
//*-*  Compute the closest distance of approach from point px,py to the position of
//*-*  this volume.
//*-*  The distance is computed in pixels units.
//*-*
//*-*  It is restricted by 2 levels of TVolumes
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   const Int_t big = 9999;
   if ( GetVisibility() == kNoneVisible )  return big;

   const Int_t inaxis = 7;
   const Int_t maxdist = 5;

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());

//*-*- return if point is not in the user area
   if (px < puxmin - inaxis) return big;
   if (py > puymin + inaxis) return big;
   if (px > puxmax + inaxis) return big;
   if (py < puymax - inaxis) return big;

   TView *view =gPad->GetView();
   if (!view) return big;

   static TVolumePosition nullPosition;
   TVolumePosition *position = pos;
   if (!position) position = &nullPosition;
   if (pos) position->UpdatePosition();
   Int_t dist = big;
   if ( !(GetVisibility() & kThisUnvisible ) ) {
     TShape  *shape = 0;
     TIter nextShape(fListOfShapes);
     while ((shape = (TShape *)nextShape())) {
      //*-*- Distnance to the next referenced shape  if visible
      if (shape->GetVisibility()) {
        Int_t dshape = shape->DistancetoPrimitive(px,py);
        if (dshape < maxdist) {
           gPad->SetSelected(this);
           return 0;
        }
        if (dshape < dist) dist = dshape;
      }
    }
  }

  if ( (GetVisibility() & kSonUnvisible) ) return dist;

//*-*- Loop on all sons
   TList *posList = GetListOfPositions();
   Int_t dnode = dist;
   if (posList && posList->GetSize()) {
      gGeometry->PushLevel();
      TVolumePosition *thisPosition;
      TObject *obj;
      TIter  next(posList);
      while ((obj = next())) {
         thisPosition = (TVolumePosition*)obj;
         TVolume *node = thisPosition->GetNode();
         dnode = node->DistancetoNodePrimitive(px,py,thisPosition);
         if (dnode <= 0)  break;
         if (dnode < dist) dist = dnode;
         if (gGeometry->GeomLevel() > 2) break;
      }
      gGeometry->PopLevel();
   }

   if (gGeometry->GeomLevel()==0 && dnode > maxdist) {
      gPad->SetSelected(view);
      return 0;
   } else
      return dnode;
}

//______________________________________________________________________________
void TVolume::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw Referenced node with current parameters*-*-*-*
//*-*                   =============================================

   TString opt = option;
   opt.ToLower();
//*-*- Clear pad if option "same" not given
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }
   if (!opt.Contains("same")) gPad->Clear();

    // Check geometry level

    Int_t iopt = atoi(option);
    TDataSet *parent = 0;
    char buffer[10];
    if (iopt < 0) {
      // set the "positive option"
      sprintf(buffer,"%d",-iopt);
      option = buffer;
      // select parent to draw
      parent = this;
      do parent = parent->GetParent();
      while (parent && ++iopt);
    }
    if (parent) parent->AppendPad(option);
    else        AppendPad(option);

//*-*- Create a 3-D View
    TView *view = gPad->GetView();
    if (!view) {
       view = new TView(1);
       view->SetAutoRange(kTRUE);
       Paint("range");
       view->SetAutoRange(kFALSE);
    }
}


//______________________________________________________________________________
void TVolume::DrawOnly(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Draw only Sons of this node*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ===========================

   SetVisibility(kThisUnvisible);
   Draw(option);
}


//______________________________________________________________________________
void TVolume::ExecuteEvent(Int_t, Int_t, Int_t)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//*-*  This member function must be implemented to realize the action
//*-*  corresponding to the mouse click on the object in the window
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//   if (gPad->GetView())
//             gPad->GetView()->ExecuteRotateView(event, px, py);

//   if (!gPad->GetListOfPrimitives()->FindObject(this)) gPad->SetCursor(kCross);
   gPad->SetCursor(kHand);
}

//______________________________________________________________________________
TRotMatrix *TVolume::GetIdentity()
{
 Double_t *IdentityMatrix = 0;
 // Return a pointer the "identity" matrix
 if (!gIdentity) {
      gIdentity = gGeometry->GetRotMatrix("Identity");
      if (!gIdentity) {
         gIdentity  =  new TRotMatrix();
         gIdentity->SetName("Identity");
         gIdentity->SetTitle("Identity matrix");
         gIdentity->SetMatrix(IdentityMatrix);
         gIdentity->SetMatrix(IdentityMatrix);
         IdentityMatrix = gIdentity->GetMatrix();
                                *IdentityMatrix = 1;
         IdentityMatrix += 4;   *IdentityMatrix = 1;
         IdentityMatrix += 4;   *IdentityMatrix = 1;
         gGeometry->GetListOfMatrices()->AddFirst(gIdentity);
      }
 }
 return gIdentity;
}

//______________________________________________________________________________
Text_t *TVolume::GetObjectInfo(Int_t px, Int_t py) const
{
   if (!gPad) return 0;
   static char info[512];
   sprintf(info,"%s/%s",GetName(),GetTitle());
   Double_t x[3];
   ((TPad *)gPad)->AbsPixeltoXY(px,py,x[0],x[1]);
   TView *view =gPad->GetView();
   if (view) view->NDCtoWC(x, x);

   TIter nextShape(fListOfShapes);
   TShape *shape = 0;
   while( (shape = (TShape *)nextShape()) )
      sprintf(&info[strlen(info)]," %6.2f/%6.2f: shape=%s/%s",x[0],x[1],shape->GetName(),shape->ClassName());

   return info;
}

//______________________________________________________________________________
void TVolume::ImportShapeAttributes()
{
//*-*-*-*-*-*-*Copy shape attributes as node attributes*-*-*-*-*--*-*-*-*-*-*
//*-*          ========================================

   if (fShape) {
     SetLineColor(fShape->GetLineColor());
     SetLineStyle(fShape->GetLineStyle());
     SetLineWidth(fShape->GetLineWidth());
     SetFillColor(fShape->GetFillColor());
     SetFillStyle(fShape->GetFillStyle());
   }

   if (!GetCollection()) return;
   TVolume *volume;
   TIter  next(GetCollection());
   while ( (volume = (TVolume *)next()) )
     volume->ImportShapeAttributes();
}

//______________________________________________________________________________
void TVolume::Paint(Option_t *opt)
{
//*-*- Draw Referenced node
  gGeometry->SetGeomLevel();
  gGeometry->UpdateTempMatrix();
  PaintNodePosition(opt);
  return;
}

//______________________________________________________________________________
void TVolume::PaintNodePosition(Option_t *option,TVolumePosition *pos)
{
//*-*-*-*-*-*-*-*-*-*-*-*Paint Referenced volume with current parameters*-*-*-*
//*-*                   ==============================================
//*-*
//*-*  vis = 1  (default) shape is drawn
//*-*  vis = 0  shape is not drawn but its sons may be not drawn
//*-*  vis = -1 shape is not drawn. Its sons are not drawn
//*-*  vis = -2 shape is drawn. Its sons are not drawn
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  if ( GetVisibility() == kNoneVisible )  return;

  static TVolumePosition nullPosition;

// restrict the levels for "range" option
  Int_t level = gGeometry->GeomLevel();
//  if (option && option[0]=='r' && level > 3 && strcmp(option,"range") == 0) return;
  if ((!(GetVisibility() & kThisUnvisible)) && option && option[0]=='r' && level > 3 ) return;
  Int_t iopt = atoi(option);
  if ( (0 < iopt) && (iopt <= level) )  return;

  TPadView3D *view3D = (TPadView3D*)gPad->GetView3D();

  TVolumePosition *position = pos;
  if (!position)   position   = &nullPosition;

  // PaintPosition does change the current matrix and it MUST be callled FIRST !!!

  position->UpdatePosition(option);

  if (!(GetVisibility() & kThisUnvisible))  PaintShape(option);

  if (GetVisibility() & kSonUnvisible) return;

//*-*- Paint all sons
  TList *posList = GetListOfPositions();
  if (posList && posList->GetSize()) {
    gGeometry->PushLevel();
    TVolumePosition *thisPosition;
    TIter  next(posList);
    while ((thisPosition = (TVolumePosition *)next())) {
       if (view3D)  view3D->PushMatrix();

       TVolume *volume = thisPosition->GetNode();
       if (volume) volume->PaintNodePosition(option,thisPosition);

       if (view3D) view3D->PopMatrix();
    }
    gGeometry->PopLevel();
  }
}

//______________________________________________________________________________
void TVolume::PaintShape(Option_t *option)
{
  // Paint shape of the volume
  // To be called from the TObject::Paint method only
  Bool_t rangeView = option && option[0]=='r';
  if (!rangeView) {
    TAttLine::Modify();
    TAttFill::Modify();
  }

  if ( (GetVisibility() & kThisUnvisible) ) return;

  TIter nextShape(fListOfShapes);
  TShape *shape = 0;
  while( (shape = (TShape *)nextShape()) ) {
    if (!rangeView) {
      shape->SetLineColor(GetLineColor());
      shape->SetLineStyle(GetLineStyle());
      shape->SetLineWidth(GetLineWidth());
      shape->SetFillColor(GetFillColor());
      shape->SetFillStyle(GetFillStyle());
      TPadView3D *view3D = (TPadView3D*)gPad->GetView3D();
      if (view3D)
         view3D->SetLineAttr(GetLineColor(),GetLineWidth(),option);
    }
    shape->Paint(option);
  }
}

//______________________________________________________________________________
void TVolume::DeletePosition(TVolumePosition *position)
{
  // DeletePosition deletes the position of the TVolume *node from this TVolume
  // and removes that volume from the list of the nodes of this TVolume

  if (!position) return;

  if (GetListOfPositions()) {
    TObjLink *lnk = GetListOfPositions()->FirstLink();
    while (lnk) {
       TVolumePosition *nextPosition = (TVolumePosition *)(lnk->GetObject());
       if (nextPosition && nextPosition == position) {
          TVolume *node = nextPosition->GetNode();
          GetListOfPositions()->Remove(lnk);
          delete nextPosition;
          Remove(node);
          break;
       }
       lnk = lnk->Next();
    }
  }
}

//______________________________________________________________________________
void TVolume::GetLocalRange(Float_t *min, Float_t *max)
{
  //  GetRange
  //
  //  Calculates the size of 3 box the volume occupies,
  //  Return:
  //    two floating point arrays with the bound of box
  //     surroundind all shapes of this TVolumeView
  //

  TVirtualPad *savePad = gPad;
  //  Create a dummy TPad;
  TCanvas dummyPad("--Dumm--","dum",1,1);
  // Assing 3D TView
  TView view(1);

  gGeometry->SetGeomLevel();
  gGeometry->UpdateTempMatrix();
  view.SetAutoRange(kTRUE);
  Paint("range");
  view.GetRange(&min[0],&max[0]);
  // restore "current pad"
   if (savePad) savePad->cd();
}

//______________________________________________________________________________
void TVolume::SetVisibility(ENodeSEEN vis)
{
//*-*-*-*-*-*-*Set visibility for this volume and its sons*-*-*-*-*--*-*-*-*-*-*
//*-*          =========================================
// ENodeSEEN Visibility flag  00 - everything visible,
//                            10 - this unvisible, but sons are visible
//                            01 - this visible but sons
//                            11 - neither this nor its sons are visible
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  fVisibility = vis;
}

//______________________________________________________________________________
void TVolume::Sizeof3D() const
{
//*-*-*-*-*-*-*Return total size of this 3-D volume with its attributes*-*-*
//*-*          ==========================================================

   if (!(GetVisibility() & kThisUnvisible) ) {
     TIter nextShape(fListOfShapes);
     TShape *shape = 0;
     while( (shape = (TShape *)nextShape()) ) {
        if (shape->GetVisibility())  shape->Sizeof3D();
     }
   }

   if ( GetVisibility() & kSonUnvisible ) return;

   if (!Nodes()) return;
   TVolume *node;
   TObject *obj;
   TIter  next(Nodes());
   while ((obj = next())) {
      node = (TVolume*)obj;
      node->Sizeof3D();
   }
}
