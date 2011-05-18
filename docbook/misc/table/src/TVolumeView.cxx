// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "Riostream.h"
#include <assert.h>
#include <stdlib.h>

#include "TCanvas.h"
#include "TPad.h"
#include "TCernLib.h"
#include "TBrowser.h"
#include "TVolumeView.h"
#include "TVolumeViewIter.h"
#include "TVolumePosition.h"
#include "TROOT.h"
#include "TView.h"
#include "TTablePadView3D.h"
#include "TGeometry.h"
#include "TVirtualPad.h"
#include "TObjArray.h"
#include "TVirtualViewer3D.h"
#include "TBuffer3D.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVolumeView                                                          //
//                                                                      //
// TVolumeView class is a special kind of TDataSet with one extra       //
// pointer to wrap any TObject onto TDataSet object                     //
//                                                                      //
//  BE CAREFUL !!!                                                      //
//  One has to use it carefully no control over that extra object       //
//  is performed. This means: the object m_Obj data-member points to can//
//  be destroyed with no this kbject notifying.                         //
//  There is no tool /protection to check whether m_Obj is till alive.  //
//  It is one's  code responsilitiy                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TVolumeView)

//_____________________________________________________________________________
TVolumeView::TVolumeView(TVolumeView *viewNode,TVolumePosition *nodePosition)
            : TObjectSet(viewNode->GetName(),(TObject *)nodePosition),fListOfShapes(0)
            //             ,fListOfAttributes(0)
{
  //
  // This ctor creates a TVolumeView structure from the "marked" nodes
  // of the "viewNode" input structure
  // It re-calculates all positions according of the new topology
  // All new TVolume became UNMARKED though
  //
   if (!gGeometry) new TGeometry;
   if (viewNode) {
      SetTitle(viewNode->GetTitle());
      EDataSetPass mode = kContinue;
      TVolumeViewIter next(viewNode,0);
      TVolumeView *nextView = 0;
      while ( (nextView = (TVolumeView *)next(mode)) ){
         mode = kContinue;
         if (nextView->IsMarked()) {
            TVolumePosition *position =next[0];
            if (!position->GetNode()) {
               Error("TVolumeView ctor","%s %s ",GetName(),nextView->GetName());
            }
            Add(new TVolumeView(nextView,position));
            mode = kPrune;
         }
      }
   }
}

//_____________________________________________________________________________
TVolumeView::TVolumeView(TVolumeView *viewNode,TVolumeView *topNode)
            : TObjectSet(viewNode->GetName(),(TObject *)0),fListOfShapes(0)
            //             ,fListOfAttributes(0)
{
  //
  // This ctor creates a TVolumeView structure containing:
  //
  //   - viewNode on the top
  //   - skip ALL node from the original viewNode untill topNode found
  //   - include all "marked" node below "topNode" if any
  //     topNode is always included
  //
  // It re-calculates all positions according of the new topology
  //
   if (!gGeometry) new TGeometry;
   if (viewNode && topNode) {
      SetTitle(viewNode->GetTitle());
      // define the depth of the "top" Node
      EDataSetPass mode = kContinue;
      TVolumeViewIter next(viewNode,0);
      TVolumeView *nextView = 0;
      while ( (nextView = (TVolumeView *)next(mode)) ){
         mode = kContinue;
         // Skip till  "top Node" found
         if (topNode != nextView) continue;
         TVolumePosition *position = next[0];
         if (!position->GetNode()) {
            Error("TVolumeView ctor","%s %s ",GetName(),nextView->GetName());
         }
         Add(new TVolumeView(nextView,position));
         break;
      }
   }
}

//_____________________________________________________________________________
TVolumeView::TVolumeView(TVolumeView *viewNode,const Char_t *nodeName1,const Char_t *nodeName2)
            : TObjectSet(viewNode->GetName(),(TObject *)0),fListOfShapes(0)
            //             ,fListOfAttributes(0)
{
  //
  // This ctor creates a TVolumeView structure containing:
  //
  //   - viewNode on the top
  //   - skip ALL node from the original viewNode untill topNodeName found
  //   - include all "marked" node below "topNodename" if any
  //     topNodeName is always included
  //
  // It re-calculates all positions according of the new topology
  //
   const Char_t *foundName[2] = {nodeName1, nodeName2};
   Bool_t found = kFALSE;
   if (!gGeometry) new TGeometry;
   if (viewNode && nodeName1 && nodeName1[0]) {
      SetTitle(viewNode->GetTitle());
      // define the depth of the "top" Node
      EDataSetPass mode = kContinue;
      TVolumeViewIter next(viewNode,0);
      TVolumeView *nextView = 0;
      while ( (nextView = (TVolumeView *)next(mode)) ){
         mode = kContinue;
         // Skip till  "top Node" found
         Int_t i = 0;
         found = kFALSE;
         for (i=0;i<2;i++) {
            if (foundName[i]) {
               if (strcmp(nextView->GetName(),foundName[i])) continue;
               foundName[i] = 0;
               found = kTRUE;
               break;
            }
         }
         if (!found) continue;
         TVolumePosition *position = next[0];
         if (!position->GetNode()) {
            Error("TVolumeView ctor","%s %s ",GetName(),nextView->GetName());
         }
         Add(new TVolumeView(nextView,position));
         mode = kPrune;
      }
   }
}

//_____________________________________________________________________________
TVolumeView::TVolumeView(TVolumeView *viewNode,const TVolumeView *node1,const TVolumeView *node2)
            : TObjectSet(viewNode->GetName(),(TObject *)0),fListOfShapes(0)
            //             ,fListOfAttributes(0)
{
  //
  // This ctor creates a TVolumeView structure containing:
  //
  //   - viewNode on the top
  //   - skip ALL node from the original viewNode untill topNodeName found
  //   - include all "marked" node below "topNodename" if any
  //     topNodeName is always included
  //
  // It re-calculates all positions according of the new topology
  //
   const TVolumeView *foundView[2] = {node1, node2};
   const Int_t nViews = sizeof(foundView)/sizeof(const TVolumeView *);
   Bool_t found = kFALSE;
   if (!gGeometry) new TGeometry;
   if (viewNode) {
      SetTitle(viewNode->GetTitle());
      // define the depth of the "top" Node
      EDataSetPass mode = kContinue;
      TVolumeViewIter next(viewNode,0);
      TVolumeView *nextView = 0;
      while ( (nextView = (TVolumeView *)next(mode)) ){
         mode = kContinue;
         // Skip till  "top Node" found
         Int_t i = 0;
         found = kFALSE;
         for (i=0;i<nViews;i++) {
            if (foundView[i]) {
               if (nextView != foundView[i]) continue;
               foundView[i] = 0;
               found = kTRUE;
               break;
            }
         }
         if (!found) continue;
         TVolumePosition *position = next[0];
         if (!position->GetNode()) {
            Error("TVolumeView ctor","%s %s ",GetName(),nextView->GetName());
         }
         Add(new TVolumeView(nextView,position));
         mode = kPrune;
      }
   }
}

//_____________________________________________________________________________
TVolumeView::TVolumeView(TVolume &pattern,Int_t maxDepLevel,
             const TVolumePosition *nodePosition,EDataSetPass iopt, TVolumeView *rootVolume)
            : TObjectSet(pattern.GetName(),(TObject *)nodePosition),fListOfShapes(0)
{
  //
  // Creates TVolumeView (view) with a topology similar with TVolume *pattern
  //
  //  Parameters:
  //  -----------
  //  pattern        - the pattern dataset
  //  iopt = kStruct - clone only my structural links
  //         kAll    - clone all links
  //         kRefs   - clone only refs
  //         kMarked - clone marked (not implemented yet) only
  //
  //   All new-created sets become the structural ones anyway.
  //
  //  cout << "ctor for " << GetName() << " - " << GetTitle() << endl;
   if (!gGeometry) new TGeometry;
   if (!nodePosition) {
      // Create the trivial position if any
      nodePosition = new TVolumePosition(&pattern);
      SetObject((TObject*)nodePosition);
   }
   if (!rootVolume) {
      rootVolume   = this;
      nodePosition = 0;
   }
   SetTitle(pattern.GetTitle());
   if ( pattern.IsMarked() ) Mark();
   TVolumePosition *position = 0;
   const TList *list = pattern.GetListOfPositions();
   if (!list || maxDepLevel == 1 || maxDepLevel < 0) return;

   TIter nextPosition(list);
   Bool_t optSel    = (iopt == kStruct);
//  Bool_t optAll    = (iopt == kAll);
   Bool_t optMarked = (iopt == kMarked);

   const TRotMatrix *thisMatrix = 0;
   Double_t thisTranslation[3] = {0,0,0};
   if (nodePosition ) {
      thisMatrix = nodePosition->GetMatrix();
      for (int i =0; i< 3; i++) thisTranslation[i]= nodePosition->GetX(i);
   }
   while ( (position = (TVolumePosition *)nextPosition()) ) {
      // define the the related TVolume
      TVolume *node     = position->GetNode();
      Double_t *positionMatrix = ((TRotMatrix *)position->GetMatrix())->GetMatrix();
      if (node) {
         UInt_t positionId = position->GetId();
         Double_t newTranslation[3] = {position->GetX(),position->GetY(),position->GetZ()};
         Double_t newMatrix[9];
         TRotMatrix currentMatrix;

         if (nodePosition) {
            if (positionMatrix) {
               TGeometry::UpdateTempMatrix(thisTranslation,thisMatrix?((TRotMatrix *)thisMatrix)->GetMatrix():0
                         ,position->GetX(),position->GetY(),position->GetZ(),positionMatrix
                         ,newTranslation,newMatrix);
               currentMatrix.SetMatrix(newMatrix);
            } else {
               TCL::vadd(thisTranslation, newTranslation,newTranslation,3);
               currentMatrix.SetMatrix(((TRotMatrix *)thisMatrix)->GetMatrix());
            }
         } else {
            if (positionMatrix)
               currentMatrix.SetMatrix(positionMatrix);
            else {
               TCL::ucopy(thisTranslation,newTranslation,3);
               currentMatrix.SetMatrix(TVolume::GetIdentity()->GetMatrix());
            }
         }
         TVolumePosition nextPos(node,newTranslation[0],newTranslation[1],
                                      newTranslation[2], &currentMatrix);
         nextPos.SetId(positionId);
         if (optMarked && !node->IsMarked()) {
            TVolumeView fakeView(*node,maxDepLevel,&nextPos,iopt,rootVolume);
            fakeView.DoOwner(kFALSE);
            continue;
         }

         if (optSel) {
            TDataSet *parent = node->GetParent();
            if ( parent && (parent != (TDataSet *)&pattern) ) continue;
         }
         TRotMatrix *newRotation = new TRotMatrix();
         newRotation->SetMatrix(currentMatrix.GetMatrix());
         TVolumePosition *nP = new TVolumePosition(node,newTranslation[0],newTranslation[1],
                                     newTranslation[2], newRotation);
         nP->SetId(positionId);
         rootVolume->Add(new TVolumeView(*node,maxDepLevel?maxDepLevel-1:0,nP,iopt));
      } else
         Error("TVolumeView ctor","Position with NO node attached has been supplied");

   }
}
//_____________________________________________________________________________
TVolumeView::TVolumeView(TVolumeView &viewNode):
             TObjectSet(viewNode.GetName(),(TObject *)viewNode.GetPosition())
            ,TAtt3D()
            ,fListOfShapes(viewNode.GetListOfShapes())
{
   //to be documented
   if (viewNode.IsOwner()) {
      viewNode.DoOwner(kFALSE); DoOwner();
   }
}

//_____________________________________________________________________________
TVolumeView::TVolumeView(Double_t *translate, Double_t *rotate, UInt_t positionId, TVolume *topNode,
                         const Char_t *thisNodePath, const Char_t *matrixName, Int_t matrixType)
            // : fListOfAttributes(0)
{
   // Special ctor to back TVolumeView::SavePrimitive() method
   if (!gGeometry) new TGeometry;
   fListOfShapes     = 0;
   TVolume *thisNode = 0;
   Double_t thisX  = translate[0];
   Double_t thisY  = translate[1];
   Double_t thisZ  = translate[2];

   // Find TVolume by path;
   if (topNode) {
      thisNode =  (TVolume *)topNode->Find(thisNodePath);
      if (!thisNode->InheritsFrom(TVolume::Class())) {
         Error("TVolumeView","wrong node <%s> on path: \"%s\"",thisNode->GetName(),thisNodePath);
         thisNode = 0;
      }
   }

   TRotMatrix *thisRotMatrix =  0;
   if (matrixName && strlen(matrixName)) thisRotMatrix = gGeometry->GetRotMatrix(matrixName);
   TVolumePosition *thisPosition = 0;
   if (thisRotMatrix)
      thisPosition = new TVolumePosition(thisNode,thisX, thisY, thisZ, matrixName);
   else if (matrixType==2)
      thisPosition = new TVolumePosition(thisNode,thisX, thisY, thisZ);
   else if (rotate) {
      const Char_t *title = "rotation";
      thisRotMatrix = new TRotMatrix((char *)matrixName,(char *)title,rotate);
      thisPosition  = new TVolumePosition(thisNode,thisX, thisY, thisZ, thisRotMatrix);
   } else
      Error("TVolumeView"," No rotation matrix is defined");
   if (thisPosition) thisPosition->SetId(positionId);
   SetObject(thisPosition);
   if (thisNode) {
      SetName(thisNode->GetName());
      SetTitle(thisNode->GetTitle());
   }
}

//_____________________________________________________________________________
TVolumeView::TVolumeView(TVolume *thisNode,TVolumePosition *nodePosition)
            : TObjectSet(thisNode?thisNode->GetName():"",(TObject *)nodePosition),fListOfShapes(0)
{
   //to be documented
   if (!gGeometry) new TGeometry;
   SafeDelete(fListOfShapes);
   if (thisNode)
      SetTitle(thisNode->GetTitle());
}

//______________________________________________________________________________
TVolumeView::~TVolumeView()
{
// default dtor (empty for this class)
}

//_____________________________________________________________________________
TVolume *TVolumeView::AddNode(TVolume *node)
{
  // Add the TVolume in the Tnode data-structure refered
  // by this TVolumeView object
  // Return TVolume * the input TVolume * was attached to

   TVolume *closedNode = 0;
   TVolumePosition *pos ;
   if ( node && (pos = GetPosition() )  && (closedNode = pos->GetNode()) )
      closedNode->Add(node);
   return closedNode;
}

//______________________________________________________________________________
void TVolumeView::Add(TShape *shape, Bool_t IsMaster)
{
   //to be documented
   if (!shape) return;
   if (!fListOfShapes) fListOfShapes = new TList;
   if (IsMaster)
      fListOfShapes->AddFirst(shape);
   else
      fListOfShapes->Add(shape);
}

//_____________________________________________________________________________
void TVolumeView::Browse(TBrowser *b)
{
   //to be documented
   TObjectSet::Browse(b);
//  TVolumePosition *pos = GetPosition();
//  if (pos) pos->Browse(b);
//    b->Add(pos);
}

//______________________________________________________________________________
Int_t TVolumeView::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*Compute distance from point px,py to a TVolumeView*-*-*-*-*-*
//*-*                  ===========================================
//*-*  Compute the closest distance of approach from point px,py to the position of
//*-*  this node.
//*-*  The distance is computed in pixels units.
//*-*
//*-*  It is restricted by 2 levels of TVolumes
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   const Int_t big = 9999;
   const Int_t inaxis = 7;
   const Int_t maxdist = 5;

   Int_t dist = big;

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

   TVolumePosition *position = GetPosition();
   TVolume *thisNode  = 0;
   TShape  *thisShape = 0;
   if (position) {
      thisNode = position->GetNode();
      position->UpdatePosition();
      if (thisNode) {
         thisShape    = thisNode->GetShape();
         if (!(thisNode->GetVisibility() & TVolume::kThisUnvisible) &&
            thisShape && thisShape->GetVisibility()) {
            dist = thisShape->DistancetoPrimitive(px,py);
            if (dist < maxdist) {
               gPad->SetSelected(this);
               return 0;
            }
         }
      }
   }

//   if ( TestBit(kSonsInvisible) ) return dist;

//*-*- Loop on all sons
   TSeqCollection *fNodes =  GetCollection();
   Int_t nsons = fNodes?fNodes->GetSize():0;
   Int_t dnode = dist;
   if (nsons) {
      gGeometry->PushLevel();
      TVolume *node;
      TIter  next(fNodes);
      while ((node  = (TVolume *)next())) {
         dnode = node->DistancetoPrimitive(px,py);
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
void TVolumeView::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw Referenced node with current parameters*-*-*-*
//*-*                   =============================================

   TString opt = option;
   opt.ToLower();
//*-*- Clear pad if option "same" not given
   if (!gPad) {
      gROOT->MakeDefCanvas();
   }
   if (!opt.Contains("same")) gPad->Clear();

//*-*- Draw Referenced node
   gGeometry->SetGeomLevel();
   gGeometry->UpdateTempMatrix();

   // Check geometry level

   Int_t iopt = atoi(option);
   TDataSet *parent = 0;
   char buffer[10];
   if (iopt < 0) {
      snprintf(buffer,10,"%d",-iopt);
      option = buffer;
      // select parent to draw
      parent = this;
      do parent = parent->GetParent();
      while (parent && ++iopt);
   }
   if (parent) parent->AppendPad(option);
   else        AppendPad(option);

#if ROOT_VERSION_CODE >= ROOT_VERSION(4,03,05)
   // the new (4.03/05) way to active 3D viewer
   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = TView::CreateView(1,0,0);
      // Set the view to perform a first autorange (frame) draw.
      // TViewer3DPad will revert view to normal painting after this
      view->SetAutoRange(kTRUE);
   }

   // Create a 3D viewer to draw us
   gPad->GetViewer3D(option);
#else
    Paint(option);
#endif
}

//_____________________________________________________________________________
TVolume *TVolumeView::GetNode() const
{
   //to be documented
   TVolumePosition *pos = GetPosition();
   if (pos)
      return pos->GetNode();
   return 0;
}

//_____________________________________________________________________________
Int_t TVolumeView::GetGlobalRange(const TVolumeView *rootNode,Float_t *globalMin,Float_t *globalMax)
{
   //
   // Calculate the position of the vertrex of the outlined cube in repect
   // of the given TVolumeView object
   //
   if (rootNode) {
      SetTitle(rootNode->GetTitle());
      EDataSetPass mode = kContinue;
      TVolumeViewIter next((TVolumeView *)rootNode,0);
      TVolumeView *nextView = 0;
      // Find itself.
      while ( (nextView = (TVolumeView *)next(mode)) && nextView != this ){}
      if (nextView == this) {
         TVolumePosition *position = next[0];
         if (!position->GetNode()) {
            Error("TVolumeView ctor","%s %s ",GetName(),nextView->GetName());
         }
         // Calculate the range of the outlined cube verteces.
         GetLocalRange(globalMin,globalMax);
         Float_t offSet[3] = {position->GetX(),position->GetY(),position->GetZ()};
         for (Int_t i=0;i<3;i++) {
            globalMin[i] += offSet[i];
            globalMax[i] += offSet[i];
         }
      }
      return next.GetDepth();
   }
   else return -1;
}

//______________________________________________________________________________
void TVolumeView::GetLocalRange(Float_t *min, Float_t *max)
{
  //  GetRange
  //
  //  Calculates the size of 3 box the node occupies.
  //  Return:
  //    two floating point arrays with the bound of box
  //     surroundind all shapes of this TModeView
  //

   TVirtualPad *savePad = gPad;
   //  Create a dummy TPad;
   TCanvas dummyPad("--Dumm--","dum",1,1);
   // Assing 3D TView
   TView *view = TView::CreateView(1,0,0);

   gGeometry->SetGeomLevel();
   gGeometry->UpdateTempMatrix();
   view->SetAutoRange(kTRUE);
   Paint("range");
   view->GetRange(&min[0],&max[0]);
   delete view;
   // restore "current pad"
   if (savePad) savePad->cd();
}

//______________________________________________________________________________
char *TVolumeView::GetObjectInfo(Int_t px, Int_t py) const
{
   //to be documented
   if (!gPad) return 0;
   static char info[512];
   Double_t x[3] = {0,0,0.5};
   ((TPad *)gPad)->AbsPixeltoXY(px,py,x[0],x[1]);
   TView *view =gPad->GetView();
   if (view) {
      Double_t min[3], max[3];
      view->GetRange(min,max);
      for (int i =0; i<3;i++) min[i] = (max[i]+min[i])/2;
      view->WCtoNDC(min,max);
      min[0] = x[0]; min[1] = x[1];
      min[2] = max[2];
      view->NDCtoWC(min, x);
   }
   TShape *shape = GetShape();
   if (shape)
      snprintf(info,512,"%6.2f/%6.2f/%6.2f: %s/%s, shape=%s/%s",x[0],x[1],x[2],GetName(),GetTitle(),shape->GetName(),shape->ClassName());
   else
      snprintf(info,512,"%6.2f/%6.2f/%6.2f: %s/%s",x[0],x[1],x[2],GetName(),GetTitle());
   return info;
}

//______________________________________________________________________________
TVolumePosition  *TVolumeView::Local2Master(const Char_t *localName, const Char_t *masterName)
{
   //to be documented
   TVolumeView *masterNode = this;
   TVolumePosition *position = 0;
   if (masterName && masterName[0]) masterNode = (TVolumeView *)Find(masterName);
   if (masterNode) {
      TVolumeViewIter transform(masterNode,0);
      if (transform(localName)) position = transform[0];
   }
   return position;
}

//______________________________________________________________________________
TVolumePosition *TVolumeView::Local2Master(const TVolumeView *localNode,const TVolumeView *masterNode)
{
   //to be documented
   TVolumePosition *position = 0;
   if (!masterNode) masterNode = this;
   if (masterNode && localNode) {
      TVolumeViewIter transform((TVolumeView *)masterNode,0);
      TVolumeView *nextNode = 0;
      while ((nextNode = (TVolumeView *)transform()) && nextNode != localNode) { }
      if (nextNode) position = transform[0];
   }
   return position;
}

//______________________________________________________________________________
Float_t *TVolumeView::Local2Master(const Float_t *local, Float_t *master,
                                   const Char_t *localName, const Char_t *masterName, Int_t nVector)
{
  //
  // calculate  transformation  master = (M-local->master )*local + (T-local->master )
  //  where
  //     M-local->master - rotation matrix 3 x 3 from the master node to the local node
  //     T-local->master - trasport vector 3 from the master node to the local node
  //
  // returns a "master" pointer if transformation has been found
  //        otherwise 0;
  //
   Float_t *trans = 0;
   TVolumePosition *position = 0;
   TVolumeView *masterNode = this;
   if (masterName && masterName[0]) masterNode = (TVolumeView *)Find(masterName);
   if (masterNode) {
      TVolumeViewIter transform(masterNode,0);
      if (transform(localName) && (position = (TVolumePosition *) transform.GetPosition()) )
         trans = position->Local2Master(local,master,nVector);
   }
   return trans;
}

//______________________________________________________________________________
Float_t *TVolumeView::Local2Master(const Float_t *local, Float_t *master,
                                   const TVolumeView *localNode,
                                   const TVolumeView *masterNode, Int_t nVector)
{
  //
  // calculate  transformation  master = (M-local->master )*local + (T-local->master )
  //  where
  //     M-local->master - rotation matrix 3 x 3 from the master node to the local node
  //     T-local->master - trasport vector 3 from the master node to the local node
  //
  // returns a "master" pointer if transformation has been found
  //        otherwise 0;
  //
   Float_t *trans = 0;
   TVolumePosition *position = 0;
   if (!masterNode) masterNode = this;
   if (masterNode && localNode) {
      TVolumeViewIter transform((TVolumeView *)masterNode,0);
      TVolumeView *nextNode = 0;
      while ((nextNode = (TVolumeView *)transform()) && nextNode != localNode) { }
      if (nextNode && (position = (TVolumePosition *) transform.GetPosition()) )
         trans = position->Local2Master(local,master,nVector);
   }
   return trans;
}

//______________________________________________________________________________
void TVolumeView::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Paint Referenced node with current parameters*-*-*-*
//*-*                   ==============================================
//*-*
//*-*  vis = 1  (default) shape is drawn
//*-*  vis = 0  shape is not drawn but its sons may be not drawn
//*-*  vis = -1 shape is not drawn. Its sons are not drawn
//*-*  vis = -2 shape is drawn. Its sons are not drawn
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//
// It draw the TVolumeView layers from the iFirst one (form the zero) till
// iLast one reached.
//
// restrict the levels for "range" option
   Int_t level = gGeometry->GeomLevel();
   if (!option) return;
   if (option[0]=='r' && level > 3 ) return;

   Int_t iFirst =  atoi(option);
   Int_t iLast = 0;
   const char *delim = strpbrk( option,":-,");
   if (delim)  iLast = atoi(delim+1);
   if (iLast < iFirst) {
      iLast = iFirst-1;
      iFirst = 0;
   }

   if ( (0 < iLast) && (iLast < level) )  return;

   TTablePadView3D *view3D = (TTablePadView3D*)gPad->GetView3D();

   TVolume *thisNode  = 0;
   TVolumePosition *position = GetPosition();

   // UpdatePosition does change the current matrix and it MUST be called FIRST !!!
   if (position) {
      thisNode  = position->GetNode();
      position->UpdatePosition(option);
   }

   // if (option[0] !='r' ) printf(" Level %d first = %d  iLast %d \n",level, iFirst, iLast);
   if (level >= iFirst) {
      PaintShape(option);
      if (thisNode)  thisNode->PaintShape(option);
   }
////---   if ( thisNode->TestBit(kSonsInvisible) ) return;

//*-*- Paint all sons
   TSeqCollection *nodes =  GetCollection();
   Int_t nsons = nodes?nodes->GetSize():0;

   if(!nsons) return;

   gGeometry->PushLevel();
   TVolumeView *node;
   TIter  next(nodes);
   while ((node = (TVolumeView *)next())) {
      if (view3D)  view3D->PushMatrix();

      node->Paint(option);

      if (view3D)  view3D->PopMatrix();
   }
   gGeometry->PopLevel();
}

//______________________________________________________________________________
void TVolumeView::PaintShape(Option_t *option)
{
   // Paint shape of the node
   // To be called from the TObject::Paint method only
   Bool_t rangeView = option && option[0]=='r';

   TIter nextShape(fListOfShapes);
   TShape *shape = 0;
   while( (shape = (TShape *)nextShape()) ) {
      if (!shape->GetVisibility())   continue;
      if (!rangeView) {
         TTablePadView3D *view3D = (TTablePadView3D*)gPad->GetView3D();
         if (view3D)
            view3D->SetLineAttr(shape->GetLineColor(),shape->GetLineWidth(),option);
      }
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,03,05)
      // It MUST be the TShape::PAint method:
      Bool_t viewerWantsSons = kTRUE;
      TVirtualViewer3D * viewer3D = gPad->GetViewer3D();
      if (viewer3D) {
         // We only provide master frame positions in these shapes
         // so don't ask viewer preference

         // Ask all shapes for kCore/kBoundingBox/kShapeSpecific
         // Not all will support the last two - which is fine
         const TBuffer3D & buffer =
            shape->GetBuffer3D(TBuffer3D::kCore|TBuffer3D::kBoundingBox|TBuffer3D::kShapeSpecific);

         // TShape sets buffer id based on TNode * gNode
         // As we not using TNode we need to override this
         const_cast<TBuffer3D &>(buffer).fID = this;

         Int_t reqSections = viewer3D->AddObject(buffer, &viewerWantsSons);
         if (reqSections != TBuffer3D::kNone) {
            shape->GetBuffer3D(reqSections);
            viewer3D->AddObject(buffer);
         }
      }
#else
      shape->Paint(option);
#endif
   }
}

//______________________________________________________________________________
TString TVolumeView::PathP() const
{
   // return the full path of this data set
   TString str;
   TVolumeView *parent = (TVolumeView *)GetParent();
   if (parent) {
      str = parent->PathP();
      str += "/";
   }
   str +=  GetName();
   UInt_t positionId = 0;
   TVolumePosition *p = GetPosition();
   if (p) {
      char buffer[10];
      positionId = p->GetId();
      snprintf(buffer,10,";%d",p->GetId());
      str +=  buffer;
   }
   return str;
}

//_______________________________________________________________________
void TVolumeView::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   //to be documented
   const Char_t *sceleton[] = {
      "TVolumeView *CreateNodeView(TVolume *topNode) {"
     ,"  TString     thisNodePath   = "
     ,"  UInt_t      thisPositionId = "
     ,"  Double_t thisTranslate[3]  = "
     ," "
     ,"  TString        matrixName  = "
     ,"  Int_t          matrixType  = "
     ,"  Double_t     thisMatrix[]  = {  "
     ,"                                  "
     ,"                                  "
     ,"                               };"
     ,"  return = new TVolumeView(thisTranslate, thisMatrix, thisPositionId, topNode,"
     ,"                          thisNodePath.Data(),matrixName.Data(), matrixType);"
     ,"}"
   };
//------------------- end of sceleton ---------------------
   Int_t sceletonSize = sizeof(sceleton)/sizeof(const Char_t*);
   TVolumePosition *thisPosition = GetPosition();
   TVolume *thisFullNode = GetNode();
   TString thisNodePath = thisFullNode ? thisFullNode->Path() : TString("");
   // Define position
   UInt_t thisPositionId = thisPosition ? thisPosition->GetId():0;
   Double_t thisX  = thisPosition ? thisPosition->GetX():0;
   Double_t thisY  = thisPosition ? thisPosition->GetY():0;
   Double_t thisZ  = thisPosition ? thisPosition->GetZ():0;

   const TRotMatrix *matrix = thisPosition ? thisPosition->GetMatrix():0;
   Int_t matrixType = 2;
   TString matrixName = " ";
   Double_t thisMatrix[] = { 0,0,0, 0,0,0, 0,0,0 };
   if (matrix) {
      matrixName = matrix->GetName();
      memcpy(thisMatrix,((TRotMatrix *)matrix)->GetMatrix(),9*sizeof(Double_t));
      matrixType = matrix->GetType();
   }
   Int_t im = 0;
   for (Int_t lineNumber =0; lineNumber < sceletonSize; lineNumber++) {
      out << sceleton[lineNumber];                             // cout << lineNumber << ". " << sceleton[lineNumber];
      switch (lineNumber) {
         case  1:  out  << "\"" << thisNodePath.Data() << "\";" ; // cout  << "\"" << thisNodePath.Data() << "\";" ;
            break;
         case  2:  out   << thisPositionId << ";" ; // cout  << "\"" << thisNodePath.Data() << "\";" ;
            break;
         case  3:  out << "{" << thisX << ", " << thisY << ", "<< thisZ << "};";  // cout << thisX << ";" ;
            break;
         case  5:  out << "\"" << matrixName << "\";" ;           // cout << "\"" << matrixName << "\";" ;
            break;
         case  6:  out <<  matrixType << ";" ;                    // cout <<  matrixType << ";" ;
            break;
         case  7:  out << thisMatrix[im++] << ", "; out << thisMatrix[im++] << ", "; out << thisMatrix[im++] << ", ";
            break;
         case  8:  out << thisMatrix[im++] << ", "; out << thisMatrix[im++] << ", "; out << thisMatrix[im++] << ", ";
            break;
         case  9:  out << thisMatrix[im++] << ", "; out << thisMatrix[im++] << ", "; out << thisMatrix[im++];
            break;
         default:
            break;
      };
//   cout << " " << endl;
      out << " " << endl;
   }
}

//______________________________________________________________________________
void  TVolumeView::SetLineAttributes()
{
   //to be documented
   TVolume *thisNode = GetNode();
   if (thisNode) thisNode->SetLineAttributes();
}

//______________________________________________________________________________
void TVolumeView::SetVisibility(Int_t vis)
{
   //to be documented
   TVolume *node = GetNode();
   if (node) node->SetVisibility(TVolume::ENodeSEEN(vis));
}

//______________________________________________________________________________
void TVolumeView::Sizeof3D() const
{
//*-*-*-*-*-*-*Return total size of this 3-D Node with its attributes*-*-*
//*-*          ==========================================================

   if (GetListOfShapes()) {
      TIter nextShape(GetListOfShapes());
      TShape *shape = 0;
      while( (shape = (TShape *)nextShape()) ) {
         if (shape->GetVisibility())  shape->Sizeof3D();
      }
   }

   TVolume *thisNode  = GetNode();
   if (thisNode && !(thisNode->GetVisibility()&TVolume::kThisUnvisible) ) {
      TIter nextShape(thisNode->GetListOfShapes());
      TShape *shape = 0;
      while( (shape = (TShape *)nextShape()) ) {
         if (shape->GetVisibility())  shape->Sizeof3D();
      }
   }

//   if ( TestBit(kSonsInvisible) ) return;

   TVolumeView *node;
   TDataSetIter next((TVolumeView *)this);
   while ((node = (TVolumeView *)next())) node->Sizeof3D();
}
