// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TStructViewerGUI.h"
#include <TRandom.h>
#include "TStructViewer.h"
#include "TStructNodeEditor.h"
#include "TStructNodeProperty.h"
#include "TStructNode.h"
#include <TCanvas.h>
#include <RQ_OBJECT.h>
#include <TGLLogicalShape.h>
#include <TGLPhysicalShape.h>
#include <TGLWidget.h>
#include <TGButtonGroup.h>
#include <TGSplitter.h>
#include <TList.h>
#include <TClass.h>
#include <TDataMember.h>
#include <TExMap.h>
#include <TPolyLine3D.h>
#include <TObjArray.h>
#include <TColor.h>
#include <TGTab.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TMath.h>
#include <TROOT.h>

ClassImp(TStructViewerGUI);

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
//
// TStructViewerGUI is main window of TStructViewer. It provides graphical
// interface. In the window we can find panel with tabs and frame with 
// GLViewer. Tab "Info" serves information about node and is used to naviagate 
// backward and forward. Second tab "Options" is used to set few options 
// such as links visibility, sorting method or setting a pointer.
// Last tab "Editor" is tab when the TStructNodeEditor is placed.
// 
//////////////////////////////////////////////////////////////////////////

//________________________________________________________________________
TStructViewerGUI::TStructViewerGUI(TStructViewer* parent, TStructNode* nodePtr, TList* colors, const TGWindow *p,UInt_t w,UInt_t h)
   : TGMainFrame(p, w, h, kHorizontalFrame)
{
   // Constructs window with "w" as width, "h" as height and given parent "p". Argument "parent" is a pointer to TStructViewer which contains this GUI.
   // This constructor build window with all controls, build map with colors, init OpenGL Viewer and create TGeoVolumes.

   fParent = parent;
   fNodePtr = nodePtr;
   
   fMaxSlices = 10;
   fMouseX = 0;
   fMouseY = 0;
   fSelectedObject = NULL;
   fMaxRatio = 0;
   fColors = colors;

   SetCleanup(kDeepCleanup);
   //////////////////////////////////////////////////////////////////////////
   // layout
   //////////////////////////////////////////////////////////////////////////
   TGTab* tabs = new TGTab(this, 200, 200, TGTab::GetDefaultGC()(), TGTab::GetDefaultFontStruct(), kFixedWidth);
   tabs->SetWidth(200);
   TGLayoutHints* expandX = new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5,5,5,5);
   //////////////////////////////////////////////////////////////////////////
   // INFO
   //////////////////////////////////////////////////////////////////////////
   TGCompositeFrame* infoFrame = tabs->AddTab("Info");
   TGGroupFrame* fInfoMenu = new TGGroupFrame(infoFrame, "Info");
   fNodeNameLabel = new TGLabel(fInfoMenu, "Name:");
   fInfoMenu->AddFrame(fNodeNameLabel, expandX);
   fNodeTypelabel = new TGLabel(fInfoMenu, "Type:");
   fInfoMenu->AddFrame(fNodeTypelabel, expandX);
   fMembersCountLabel = new TGLabel(fInfoMenu, "Members:");
   fInfoMenu->AddFrame(fMembersCountLabel, expandX);
   fAllMembersCountLabel = new TGLabel(fInfoMenu, "All members:");
   fInfoMenu->AddFrame(fAllMembersCountLabel, expandX);
   fLevelLabel = new TGLabel(fInfoMenu, "Level:");
   fInfoMenu->AddFrame(fLevelLabel, expandX);
   fSizeLabel = new TGLabel(fInfoMenu, "Size:");
   fInfoMenu->AddFrame(fSizeLabel, expandX);
   fTotalSizeLabel = new TGLabel(fInfoMenu, "Total size:");
   fInfoMenu->AddFrame(fTotalSizeLabel, expandX);
   infoFrame->AddFrame(fInfoMenu, expandX);

   fUndoButton = new TGTextButton(infoFrame, "Undo");
   fUndoButton->Connect("Clicked()", "TStructViewerGUI", this, "UndoButtonSlot()");
   fUndoButton->SetEnabled(false);
   infoFrame->AddFrame(fUndoButton, expandX);

   fRedoButton = new TGTextButton(infoFrame, "Redo");
   fRedoButton->Connect("Clicked()", "TStructViewerGUI", this, "RedoButtonSlot()");
   fRedoButton->SetEnabled(false);
   infoFrame->AddFrame(fRedoButton, expandX);

   TGTextButton* resetCameraButton = new TGTextButton(infoFrame, "Reset camera");
   infoFrame->AddFrame(resetCameraButton, expandX);
   resetCameraButton->Connect("Clicked()", "TStructViewerGUI", this, "ResetButtonSlot()");

   TGTextButton* updateButton = new TGTextButton(infoFrame, "Update");
   updateButton->Connect("Clicked()", "TStructViewerGUI", this, "UpdateButtonSlot()");
   infoFrame->AddFrame(updateButton, expandX);

   //////////////////////////////////////////////////////////////////////////
   // OPTIONS
   //////////////////////////////////////////////////////////////////////////
   TGCompositeFrame* options = tabs->AddTab("Options");
    
   fShowLinksCheckButton = new TGCheckButton(options, "Show links");
   fShowLinksCheckButton->Connect("Toggled(Bool_t)", "TStructViewerGUI", this, "ShowLinksToggled(Bool_t)");
   options->AddFrame(fShowLinksCheckButton);
   fShowLinksCheckButton->SetState(kButtonDown);
 
   TGVButtonGroup* fSortByGroup = new TGVButtonGroup(options, "Sort by");
   fSortBySizeButton = new TGRadioButton(fSortByGroup, "Size");
   fSortBySizeButton->Connect("Clicked()", "TStructViewerGUI", this, "Update()");
   fSortBySizeButton->SetOn();
   fSortByMembersButton = new TGRadioButton(fSortByGroup, "Members count");
   fSortByMembersButton->Connect("Clicked()", "TStructViewerGUI", this, "Update()");
   options->AddFrame(fSortByGroup, expandX);
  
     
   TGLabel* fPointerLabel = new TGLabel(options, "Pointer:");
   options->AddFrame(fPointerLabel, expandX);
   fPointerTextEntry = new TGTextEntry(options, "0x0000000");
   options->AddFrame(fPointerTextEntry, expandX);
   TGLabel* fPointerTypeLabel = new TGLabel(options, "Pointer Type:");
   options->AddFrame(fPointerTypeLabel, expandX);
   fPointerTypeTextEntry = new TGTextEntry(options, "TObject");
   options->AddFrame(fPointerTypeTextEntry, expandX);
   TGTextButton* fSetPointerButton = new TGTextButton(options, "Set pointer");
   fSetPointerButton->Connect("Clicked()", "TStructViewerGUI", this, "SetPointerButtonSlot()");
   options->AddFrame(fSetPointerButton, expandX);
  
   //////////////////////////////////////////////////////////////////////////
   // EDITOR
   //////////////////////////////////////////////////////////////////////////
   TGCompositeFrame* editTab = tabs->AddTab("Editor");
   fEditor = new TStructNodeEditor(fColors, editTab);
   fEditor->Connect("Update(Bool_t)", "TStructViewerGUI", this, "Update(Bool_t)");
   editTab->AddFrame(fEditor);

   AddFrame(tabs, new TGLayoutHints(kLHintsLeft, 1,1,1,1));

   TGVSplitter* splitter = new TGVSplitter(this);
   splitter->SetFrame(tabs, true);
   this->AddFrame(splitter, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));
 
    
   fTopVolume = gGeoManager->MakeBox("TOPVolume",TStructNode::GetMedium(),10000, 10000, 10000);
   gGeoManager->SetTopVolume(fTopVolume);
   gGeoManager->SetNsegments(40);
    
   fCanvas = new TCanvas("", "", 0, 0);
   // drawing after creating canvas to avoid drawing in default canvas
   fGLViewer = new TGLEmbeddedViewer(this, fCanvas);
   AddFrame(fGLViewer->GetFrame(), new TGLayoutHints(kLHintsExpandX| kLHintsExpandY, 10,10,10,10));
   fGLViewer->PadPaint(fCanvas);
   fGLViewer->Connect("MouseOver(TGLPhysicalShape*)", "TStructViewerGUI", this, "MouseOverSlot(TGLPhysicalShape*)");
   fGLViewer->GetGLWidget()->Connect("ProcessedEvent(Event_t*)", "TStructViewerGUI", this, "GLWidgetProcessedEventSlot(Event_t*))");
   fGLViewer->SetResetCameraOnDoubleClick(false);
   fGLViewer->Connect("DoubleClicked()", "TStructViewerGUI", this, "DoubleClickedSlot()");
   fGLViewer->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
   Update();
   fGLViewer->SetResetCamerasOnUpdate(false);
 
   SetWindowName("Struct Viewer");
   MapSubwindows();
   this->SetWMSizeHints(w, h, 2000, 2000, 0, 0);
   Resize(GetDefaultSize());
   MapWindow();

   fToolTip = new TGToolTip(0, 0, "ToolTip", 500);
}

//________________________________________________________________________
TStructViewerGUI::~TStructViewerGUI()
{
   // Destructor

   delete fCanvas;
}

//________________________________________________________________________
void TStructViewerGUI::CalculatePosistion(TStructNode* parent)
{
   // Recursive method to calculating nodes posistion in 3D space

   // choose sorting method
   if (fSortBySizeButton->GetState() == kButtonDown) {
      TStructNode::SetSortBy(kSize);
   } else if (fSortByMembersButton->GetState() == kButtonDown) {
      TStructNode::SetSortBy(kMembers);
   }
   Float_t ratio = (Float_t)((parent->GetLevel()+1.0) / parent->GetLevel());

   // changing the angle between parent object and daughters
   // if center of parent is 0 that is real piramid
   parent->SetWidth(1);
   parent->SetHeight(1);
   parent->SetX(-parent->GetWidth()/2);
   parent->SetY(-parent->GetHeight()/2);
   parent->SetZ((Float_t)(1.5 * parent->GetLevel()));

   fMaxRatio = parent->GetVolumeRatio();

   // sorting list of members by size or number of members
   parent->GetMembers()->Sort(kSortDescending);
   Divide(parent->GetMembers(), (parent->GetX()) *ratio, (parent->GetX() + parent->GetWidth())* ratio, (parent->GetY())* ratio, (parent->GetY() + parent->GetHeight())*ratio);

   // sclale all the objects
   Scale(parent);
}

//________________________________________________________________________
void TStructViewerGUI::CheckMaxObjects(TStructNode* parent)
{
   // Check if all of nodes can be displayed on scene. Hides redendant nodes.

   UInt_t object = 0;

   TList queue;
   queue.Add(parent);
   TStructNode* node;

   while ((node = (TStructNode*) queue.First() )) {
      object++;

      if (object > fNodePtr->GetMaxObjects() || node->GetLevel() - fNodePtr->GetLevel() >= fNodePtr->GetMaxLevel()) {
         break;
      }

      node->SetVisible(true);

      queue.AddAll(node->GetMembers());
      queue.RemoveFirst();

      fVisibleObjects.Add(node);
   }

   TIter it(&fVisibleObjects);
   TStructNode* member;
   while ((node = (TStructNode*) it() )) {
      if(node->GetLevel() - fNodePtr->GetLevel() == fNodePtr->GetMaxLevel()-1 && node->GetMembersCount() > 0) {
         node->SetCollapsed(true);
         continue;
      }

      TIter memIt(node->GetMembers());
      while ((member = (TStructNode*) memIt() )) {
         if(member->IsVisible() == false) {
            node->SetCollapsed(true);
            break;
         }
      }
   }
}

//________________________________________________________________________
void TStructViewerGUI::CloseWindow()
{
   // Delete window

   DeleteWindow();
}

//________________________________________________________________________
void TStructViewerGUI::Divide(TList* list, Float_t x1, Float_t x2, Float_t y1, Float_t y2)
{
   // Divides rectangle where the outlining box is placed.

   if (list->GetSize() > 1) { // spliting node into two lists
      ULong_t sum1 = 0, sum = 0;

      TStructNode* node;
      TList list1, list2;
      TIter it(list);

      while((node = (TStructNode*) it() )) {
         sum += node->GetVolume();
      }
      it.Reset();
      while((node = (TStructNode*) it() )) {
         if(sum1 >= sum/2.0) {
            list2.Add(node);
         } else {
            sum1 += node->GetVolume();
            list1.Add(node);
         }
      }

      Float_t ratio = (float)sum1/sum;

      Float_t width = x2 - x1;
      Float_t height = y2 - y1;
      if (width < height) { // vertical split
         Float_t split = y1 + ratio * height;
         Divide(&list1, x1, x2, y1, split);
         Divide(&list2, x1, x2, split, y2);
      } else { // horizontal
         Float_t split = x1 + ratio * width;
         Divide(&list1, x1, split, y1, y2);
         Divide(&list2, split, x2, y1, y2);
      }
   } else if (list->GetSize() == 1) { // divide place to node
      TStructNode* node = (TStructNode*)(list->First());

      node->SetWidth(x2 - x1);
      node->SetHeight(y2 - y1);
      node->SetX(x1);
      node->SetY(y1);
      node->SetZ((Float_t)(1.50 * node->GetLevel()));

      if (node->GetVolumeRatio() > fMaxRatio) {
         fMaxRatio = node->GetVolumeRatio();
      }

      Float_t ratio = (Float_t)((node->GetLevel()+1.0)/node->GetLevel());
      node->GetMembers()->Sort(kSortDescending);
      Divide(node->GetMembers(), x1*ratio, x2*ratio, y1*ratio, y2*ratio);
   }
}

//________________________________________________________________________
void TStructViewerGUI::DoubleClickedSlot()
{
   // Activated when user double click on objects on 3D scene. Sets clicked node to top node 
   // and updates scene with camers reset.

   if (fSelectedObject) {
      if(fSelectedObject == fNodePtr) {
         return;
      }

      fUndoList.Add(fNodePtr);
      fNodePtr = fSelectedObject;
      fUndoButton->SetEnabled(true);

      Update(kTRUE);
   }
}
//________________________________________________________________________
void TStructViewerGUI::Draw(Option_t* /*option*/)
{
   // Check limits and draws nodes and links

   CheckMaxObjects(fNodePtr);

   CalculatePosistion(fNodePtr);
   DrawVolumes(fNodePtr);

   if(fShowLinksCheckButton->GetState() == kButtonDown) {
      DrawLink(fNodePtr);
   }

   UnCheckMaxObjects();
}

//________________________________________________________________________
void TStructViewerGUI::DrawLink(TStructNode* parent)
{
   // Recursive method to draw links
   if(parent->GetLevel() - fNodePtr->GetLevel() >= fNodePtr->GetMaxLevel()) {
      return;
   }

   if(parent->IsCollapsed()) {
      return;
   }

   TIter it(parent->GetMembers());
   TStructNode* node;
   while((node = (TStructNode*) it())) {
      if(node->GetLevel() - parent->GetLevel() > 1) {
         printf("node = %s, parent = %s\n", node->GetName(), parent->GetName());
         printf("nodelvl = %d, parentlvl = %d\n", node->GetLevel(), parent->GetLevel());
      } 

      TPolyLine3D *l = new TPolyLine3D(2);
      l->SetPoint(0 ,node->GetCenter(), node->GetMiddle(), -node->GetZ());
      l->SetPoint(1 ,parent->GetCenter(), parent->GetMiddle(), -parent->GetZ());

      l->SetLineColor(GetColor(node));
      l->SetLineWidth(1);
      l->Draw();

      if(!node->IsCollapsed()) {
         DrawLink(node);
      }
   }
}

//________________________________________________________________________
void TStructViewerGUI::DrawNode(TStructNode* node)
{
   // Creates and draws TGeoVolume from given "node"

   TGeoVolume* vol;

   /*if(node->IsCollapsed())
   {
   //float r = (node->GetWidth() < node->GetHeight() ? 0.5 * node->GetWidth() : 0.5 * node->GetHeight());
   //vol = gGeoManager->MakeTorus(node->GetName(),TStructNode::GetMedium(), 0.75*r, 0, r/4);

   vol = gGeoManager->MakeBox(TString(node->GetName()) + "up",TStructNode::GetMedium(), 0.45*node->GetWidth(), 0.45*node->GetHeight(), (node->GetWidth() < node->GetHeight() ? 0.45 * node->GetWidth() : 0.45 * node->GetHeight()));
   Double_t max = TMath::Max(0.22 * node->GetWidth(), 0.22 * node->GetHeight());
   TGeoVolume* subvol = gGeoManager->MakeTrd2(node->GetName(), TStructNode::GetMedium(), 0, 0.45 * node->GetWidth(), 0, 0.45 * node->GetHeight(), max);
   subvol->SetLineColor(GetColor(node));
   subvol->SetNumber((Int_t)node);
   TGeoTranslation* subtrans = new TGeoTranslation("subtranslation", 0, 0, -max);
   vol->AddNodeOverlap(subvol, 1, subtrans);

   subvol = gGeoManager->MakeTrd2(TString(node->GetName()) + "down", TStructNode::GetMedium(), 0.45 * node->GetWidth(), 0, 0.45 * node->GetHeight(), 0, max);
   subvol->SetLineColor(GetColor(node));
   subvol->SetNumber((Int_t)node);
   subtrans = new TGeoTranslation("subtranslation", 0, 0, max);
   vol->AddNodeOverlap(subvol, 1, subtrans);
   }
   else*/ if(node->GetNodeType() == kCollection) {
      vol = gGeoManager->MakeBox(node->GetName(),TStructNode::GetMedium(), 0.45*node->GetWidth(), 0.45*node->GetHeight(), 0.1);
      // subboxes
      Float_t slices = (Float_t)(node->GetMembersCount());
      if (slices > fMaxSlices) {
         slices = (Float_t)fMaxSlices;
      }

      for (Float_t i = -(slices-1)/2; i < slices/2; i++) {
         TString name = node->GetName();
         name += i;
         TGeoVolume* sub = gGeoManager->MakeBox(name,TStructNode::GetMedium(),0.45*node->GetWidth() * 0.7 / slices, 0.45*node->GetHeight(), 0.1);
         sub->SetLineColor(GetColor(node));
         sub->SetTitle(TString::Format("represents node %ld", (Long_t)node));
         TGeoTranslation* subtrans = new TGeoTranslation("subtranslation", i * node->GetWidth() / slices, 0, 0);
         vol->AddNodeOverlap(sub, 1, subtrans);
      }
   } else {
      vol = gGeoManager->MakeBox(node->GetName(),TStructNode::GetMedium(), 0.45*node->GetWidth(), 0.45*node->GetHeight(), 0.1);
   }

   vol->SetLineColor(GetColor(node));
   vol->SetLineWidth(1);

   TGeoTranslation* trans = new TGeoTranslation("translation", node->GetCenter(), node->GetMiddle(), -node->GetZ());
   vol->SetTitle(TString::Format("represents node %ld", (Long_t)node));

   fTopVolume->AddNode(vol,1, trans);
}

//________________________________________________________________________
void TStructViewerGUI::DrawVolumes(TStructNode* parent)
{
   // Recursive method to draw GeoVolumes

   if(parent->GetLevel() - fNodePtr->GetLevel() >= fNodePtr->GetMaxLevel()) {
      return;
   }

   DrawNode(parent);

   if(parent->IsCollapsed()) {
      return;
   }

   TIter nextVis(parent->GetMembers());
   TStructNode* node;
   while((node = (TStructNode*)nextVis())) {
      DrawVolumes(node);
   }
}

//________________________________________________________________________
TStructNodeProperty* TStructViewerGUI::FindNodeProperty(TStructNode* node)
{
   // Returns poitner to property associated with node "node". If property is not found
   // then it returns default property

   TIter it(fColors);
   TStructNodeProperty* prop;
   while ((prop = (TStructNodeProperty*) it() )) {
      TString propName(prop->GetName());
      if (propName.EndsWith("+")) {

         if (TClass* cl = TClass::GetClass(node->GetTypeName())) {
            propName.Remove(propName.Length()-1, 1);
            if (cl->InheritsFrom(propName.Data())) {
               return prop;
            }
         }
      } else {
         if (propName == TString(node->GetTypeName())) {
            return prop;
         }
      }
   }

   return (TStructNodeProperty*)fColors->Last();
}

//________________________________________________________________________`
TCanvas* TStructViewerGUI::GetCanvas()
{
   // Returns canvas used to keep TGeoVolumes

   return fCanvas;
}
//________________________________________________________________________
Int_t TStructViewerGUI::GetColor(TStructNode* node) 
{
   // Returns color form fColors for given "node"

   TStructNodeProperty* prop = FindNodeProperty(node);
   if (prop) {
      return prop->GetColor().GetNumber();
   }

   return 2;
}

//________________________________________________________________________
Bool_t TStructViewerGUI::GetLinksVisibility() const
{
   // Returns true if links are visible, otherwise return false.

   if (fShowLinksCheckButton->GetState() == kButtonDown) {
      return true;
   } else {
      return false;
   }
}

//________________________________________________________________________
TStructNode* TStructViewerGUI::GetNodePtr() const
{
   // Returns top node pointer

   return fNodePtr;
}

//________________________________________________________________________
void TStructViewerGUI::GLWidgetProcessedEventSlot(Event_t* event)
{
   // Handle events. Sets fMouseX and fMouseY when user move a mouse over viewer and hides ToolTip

   switch (event->fType) {
      case kMotionNotify:
         fMouseX = event->fXRoot + 15;
         fMouseY = event->fYRoot + 15;
         break;

      case kButtonPress:
         fToolTip->Hide();
         if (fSelectedObject) {
            UpdateLabels(fSelectedObject);
            fEditor->SetModel(fSelectedObject);
         }         
         break;

      default:
         break;
   }
}

//________________________________________________________________________
void TStructViewerGUI::MouseOverSlot(TGLPhysicalShape* shape)
{
   // MouseOver slot. Activated when user out mouse over object on scene. 
   // Sets ToolTip and updates labels

   fToolTip->Hide();
   fSelectedObject = NULL;
   if (shape && shape->GetLogical()) {
      fSelectedObject =  (TStructNode*)(shape->GetLogical()->ID());
      if (fSelectedObject) {
         if (fSelectedObject->IsA()->InheritsFrom("TPolyLine3D")) {
            fSelectedObject = NULL;
            return;
         }
         // + 16 skips "represents node "
         fSelectedObject = (TStructNode*)(TString(fSelectedObject->GetTitle() + 16).Atoll()); 
         fToolTip->SetText(TString(fSelectedObject->GetName()) + "\n" + fSelectedObject->GetTypeName());
         fToolTip->SetPosition(fMouseX, fMouseY);
         fToolTip->Reset();
         UpdateLabels(fSelectedObject);
      }
   }
}

//________________________________________________________________________
void TStructViewerGUI::RedoButtonSlot()
{
   // Activated when user click Redo button. Repeat last Undo action.

   fUndoList.Add(fNodePtr);
   fUndoButton->SetEnabled(true);
   fNodePtr = (TStructNode*) fRedoList.Last();
   fRedoList.RemoveLast();
   if (!fRedoList.First()) {
      fRedoButton->SetEnabled(false);
   }
   Update(kTRUE);
   UpdateLabels(fNodePtr);
}

//________________________________________________________________________
void TStructViewerGUI::ResetButtonSlot()
{
   // Resets camera

   fGLViewer->UpdateScene();
   fGLViewer->ResetCurrentCamera();
}

//________________________________________________________________________
void TStructViewerGUI::Scale(TStructNode* parent)
{
   // Recursive method to scaling all modes on scene. We have to scale nodes to get real ratio between nodes. 
   // Uses fMaxRatio.

   // newRatio = sqrt(ratio/maxratio)
   Float_t newRatio = (Float_t)(TMath::Sqrt(parent->GetRelativeVolumeRatio()/fMaxRatio));
   // set left top conner in the center
   parent->SetX(parent->GetX() + parent->GetWidth()/2);
   parent->SetY(parent->GetY() + parent->GetHeight()/2);
   // set new size
   Float_t min = (Float_t)TMath::Min(parent->GetWidth(), parent->GetHeight());
   parent->SetWidth(parent->GetWidth() * newRatio);
   parent->SetHeight(parent->GetHeight() * newRatio);
   // fit the ratio -> height to width
   Float_t sqrt = (Float_t)(TMath::Sqrt(parent->GetWidth() * parent->GetHeight()));
   // it's a square
   if (min > sqrt) {
      parent->SetWidth(sqrt);
      parent->SetHeight(sqrt);
   } else { // it's rectangle
      if (parent->GetHeight() > parent->GetWidth()) {
         parent->SetWidth(min);
         parent->SetHeight(sqrt * sqrt / min);
      } else {
         parent->SetWidth(sqrt * sqrt / min);
         parent->SetHeight(min);
      }
   }
   // move left top corner
   parent->SetX(parent->GetX() - parent->GetWidth()/2);
   parent->SetY(parent->GetY() - parent->GetHeight()/2);

   // scale others nodes
   TStructNode* node;
   TIter it(parent->GetMembers());
   while ((node = (TStructNode*) it() )) {
      Scale(node);
   }
}

//________________________________________________________________________
void TStructViewerGUI::SetNodePtr(TStructNode* val)
{
   // Sets top node pointer and updates view

   fNodePtr = val;
   Update(kTRUE);
}

//________________________________________________________________________
void TStructViewerGUI::SetLinksVisibility(Bool_t visible)
{
   // Sets links visibility to "visible"

   if (visible) {
      fShowLinksCheckButton->SetState(kButtonDown);
   } else {
      fShowLinksCheckButton->SetState(kButtonUp);
   }
}

//________________________________________________________________________
void TStructViewerGUI::SetPointerButtonSlot()
{
   // Sets pointer given in fPointerTestEntry to the main pointer

   TObject* obj = (TObject*)gROOT->ProcessLine(fPointerTextEntry->GetText());
   fParent->SetPointer(obj);
}

//________________________________________________________________________
void TStructViewerGUI::ShowLinksToggled(Bool_t /*on*/)
{
   // Changes links visibility and refresh view.

   Update();
}

//________________________________________________________________________
void TStructViewerGUI::UnCheckMaxObjects()
{
   // Shows hidden nodes

   TStructNode* node;
   TIter it(&fVisibleObjects);

   while ((node = (TStructNode*) it() )) {
      node->SetCollapsed(false);
      node->SetVisible(false);
   }

   fVisibleObjects.Clear();
}

//________________________________________________________________________
void TStructViewerGUI::Update(Bool_t resetCamera)
{
   // Updates view. Clear all the nodes, call draw function and update scene. Doesn't reset camera.

   fCanvas->GetListOfPrimitives()->Clear();
   fTopVolume->ClearNodes();
   Draw();
   fCanvas->GetListOfPrimitives()->Add(fTopVolume);
   fGLViewer->UpdateScene();

   if(resetCamera) {
      fGLViewer->ResetCurrentCamera();
   }
}

//________________________________________________________________________
void TStructViewerGUI::UpdateButtonSlot()
{
   // Update button slot. Updates scene

   Update();
}

//________________________________________________________________________
void TStructViewerGUI::UpdateLabels(TStructNode* node)
{
   // Refresh information in labels when user put mouse over object

   fNodeNameLabel->SetText(node->GetName());
   fNodeTypelabel->SetText(node->GetTypeName());

   TString name = "Members: ";
   name += node->GetMembersCount();
   fMembersCountLabel->SetText(name);
   name = "All members: ";
   name += node->GetAllMembersCount();
   fAllMembersCountLabel->SetText(name);
   name = "Level: ";
   name += node->GetLevel();
   fLevelLabel->SetText(name);
   name = "Size: ";
   name += node->GetSize();
   fSizeLabel->SetText(name);
   name = "Total size: ";
   name += node->GetTotalSize();
   fTotalSizeLabel->SetText(name);
}

//________________________________________________________________________
void TStructViewerGUI::UndoButtonSlot()
{
   // UndoButton Slot. Activated when user press Undo button. Restore last top node pointer. 

   fRedoList.Add(fNodePtr);
   fRedoButton->SetEnabled(true);
   fNodePtr = (TStructNode*) fUndoList.Last();
   fUndoList.RemoveLast();
   if (!fUndoList.First()) {
      fUndoButton->SetEnabled(false);
   }
   Update(kTRUE);
   UpdateLabels(fNodePtr);
}
