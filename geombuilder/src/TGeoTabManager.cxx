// @(#):$Name:  $:$Id: Exp $
// Author: M.Gheata 

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTabManager                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualPad.h"
#include "TGedFrame.h"
#include "TGTab.h"
#include "TGLabel.h"
#include "TGComboBox.h"
#include "TGListBox.h"
#include "TGListTree.h"
#include "TGTextEntry.h"
#include "TGCanvas.h"
#include "TGMimeTypes.h"

#include "TGeoManager.h"
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoMedium.h"
#include "TGeoMaterial.h"
#include "TGeoMatrix.h"

#include "TGeoTabManager.h"

ClassImp(TGeoTabManager)

//______________________________________________________________________________
TGeoTabManager::TGeoTabManager(TVirtualPad *pad, TGTab *tab)
{
// Ctor.
   fPad = pad;
   fTab = tab;
//   fShape = 0;
   fVolume = 0;
//   fMedium = 0;
//   fMaterial = 0;
//   fMatrix = 0;
   fShapeCombos = new TList();
   fShapePanel = 0;
   fMediumPanel = 0;
   fMaterialPanel = 0;
   fMatrixPanel = 0;
//   fVolumeCombos = new TList();
   fMatrixCombos = new TList();
   fMediumCombos = new TList();
   fMaterialCombos = new TList();
   CreateTabs();
   fTab->MapSubwindows();
   fTab->Layout();
   fTab->MapWindow();
//   AppendPad();
   TClass *cl = TGeoTabManager::Class();
   cl->GetEditorList()->Add(this);
}   

//______________________________________________________________________________
TGeoTabManager::~TGeoTabManager()
{
// Dtor.
   if (fShapePanel) delete fShapePanel;
   if (fMaterialPanel) delete fMaterialPanel;
   if (fMatrixPanel) delete fMatrixPanel;
   if (fMediumPanel) delete fMediumPanel;
   delete fShapeCombos;
//   delete fVolumeCombos;
   delete fMatrixCombos;
   delete fMediumCombos;
   delete fMaterialCombos;
}   

//______________________________________________________________________________
void TGeoTabManager::AddComboShape(TGComboBox *combo)
{
// Add an element to the list.
   fShapeCombos->Add(combo);
   TIter next(gGeoManager->GetListOfShapes());
   TNamed *obj;
   Int_t id = 0;
   while ((obj=(TNamed*)next())) combo->AddEntry(obj->GetName(), id++);
}   

//______________________________________________________________________________
void TGeoTabManager::AddShape(const char *name, Int_t id)
{
// Add an element to the list.
   TIter next(fShapeCombos);
   TGComboBox *combo;
   while ((combo=(TGComboBox*)next())) combo->AddEntry(name, id);
}   

//______________________________________________________________________________
void TGeoTabManager::UpdateShape(Int_t id)
{
// Update an element from the list.
   TIter next(fShapeCombos);
   TGComboBox *combo;
   TNamed *obj = (TNamed*)gGeoManager->GetListOfShapes()->At(id); 
   while ((combo=(TGComboBox*)next())) {
      ((TGTextLBEntry*)combo->GetListBox()->GetEntry(id))->SetText(new TGString(obj->GetName()));
   }
}   

//______________________________________________________________________________
void TGeoTabManager::AddComboMatrix(TGComboBox *combo)
{
// Add an element to the list.
   fMatrixCombos->Add(combo);
   TIter next(gGeoManager->GetListOfMatrices());
   TNamed *obj;
   Int_t id = 0;
   while ((obj=(TNamed*)next())) combo->AddEntry(obj->GetName(), id++);
}   

//______________________________________________________________________________
void TGeoTabManager::AddMatrix(const char *name, Int_t id)
{
// Add an element to the list.
   TIter next(fMatrixCombos);
   TGComboBox *combo;
   while ((combo=(TGComboBox*)next())) combo->AddEntry(name, id);
}   

//______________________________________________________________________________
void TGeoTabManager::UpdateMatrix(Int_t id)
{
// Update an element from the list.
   TIter next(fMatrixCombos);
   TGComboBox *combo;
   TNamed *obj = (TNamed*)gGeoManager->GetListOfMatrices()->At(id); 
   while ((combo=(TGComboBox*)next())) {
      ((TGTextLBEntry*)combo->GetListBox()->GetEntry(id))->SetText(new TGString(obj->GetName()));
   }
}   

//______________________________________________________________________________
void TGeoTabManager::AddComboMedium(TGComboBox *combo)
{
// Add an element to the list.
   fMediumCombos->Add(combo);
   TIter next(gGeoManager->GetListOfMedia());
   TNamed *obj;
   Int_t id = 0;
   while ((obj=(TNamed*)next())) combo->AddEntry(obj->GetName(), id++);
}   

//______________________________________________________________________________
void TGeoTabManager::AddMedium(const char *name, Int_t id)
{
// Add an element to the list.
   TIter next(fMediumCombos);
   TGComboBox *combo;
   while ((combo=(TGComboBox*)next())) combo->AddEntry(name, id);
}   

//______________________________________________________________________________
void TGeoTabManager::UpdateMedium(Int_t id)
{
// Update an element from the list.
   TIter next(fMediumCombos);
   TGComboBox *combo;
   TNamed *obj = (TNamed*)gGeoManager->GetListOfMedia()->At(id); 
   while ((combo=(TGComboBox*)next())) {
      ((TGTextLBEntry*)combo->GetListBox()->GetEntry(id))->SetText(new TGString(obj->GetName()));
   }
}   

//______________________________________________________________________________
void TGeoTabManager::AddComboMaterial(TGComboBox *combo)
{
// Add an element to the list.
   fMaterialCombos->Add(combo);
   TIter next(gGeoManager->GetListOfMaterials());
   TNamed *obj;
   Int_t id = 0;
   while ((obj=(TNamed*)next())) combo->AddEntry(obj->GetName(), id++);
}   

//______________________________________________________________________________
void TGeoTabManager::AddMaterial(const char *name, Int_t id)
{
// Add an element to the list.
   TIter next(fMaterialCombos);
   TGComboBox *combo;
   while ((combo=(TGComboBox*)next())) combo->AddEntry(name, id);
}   

//______________________________________________________________________________
void TGeoTabManager::UpdateMaterial(Int_t id)
{
// Update an element from the list.
   TIter next(fMaterialCombos);
   TGComboBox *combo;
   TNamed *obj = (TNamed*)gGeoManager->GetListOfMaterials()->At(id); 
   while ((combo=(TGComboBox*)next())) {
      ((TGTextLBEntry*)combo->GetListBox()->GetEntry(id))->SetText(new TGString(obj->GetName()));
   }
}   

//______________________________________________________________________________
void TGeoTabManager::CreateTabs()
{
// Create all needed tab elements.
   fVolumeCont = fTab->AddTab("Volume");   
   fVolumeTab = new TGCompositeFrame(fVolumeCont, 110, 30, kVerticalFrame);
   fVolumeCont->AddFrame(fVolumeTab, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 2, 2));
   fTab->SetEnabled(GetTabIndex(kTabVolume), kFALSE);
}

//______________________________________________________________________________
void TGeoTabManager::GetShapeEditor(TGeoShape *shape)
{
// Get editor for a shape.
   if (!shape) return;
   if (!fShapePanel) fShapePanel = new TGeoTransientPanel("Shape", shape);
   else {
      fShapePanel->SetModel(shape);   
      fShapePanel->Show();
   }   
}

//______________________________________________________________________________
void TGeoTabManager::GetVolumeEditor(TGeoVolume *volume)
{
// Get editor for a volume.
   if (!volume || !fVolumeTab) return;
   GetEditors(TAttLine::Class(), fVolumeTab);
   GetEditors(TGeoVolume::Class(), fVolumeTab);
   SetModel(kTabVolume, volume, 0);
}
   
//______________________________________________________________________________
void TGeoTabManager::GetMatrixEditor(TGeoMatrix *matrix)
{
// Get editor for a matrix.
   if (!matrix) return;
   if (!fMatrixPanel) fMatrixPanel = new TGeoTransientPanel("Matrix", matrix);
   else {
      fMatrixPanel->SetModel(matrix);   
      fMatrixPanel->Show();
   }   
}

//______________________________________________________________________________
void TGeoTabManager::GetMediumEditor(TGeoMedium *medium)
{
// Get editor for a medium.
   if (!medium) return;
   if (!fMediumPanel) fMediumPanel = new TGeoTransientPanel("Medium", medium);
   else {
      fMediumPanel->SetModel(medium);   
      fMediumPanel->Show();
   }   
}

//______________________________________________________________________________
void TGeoTabManager::GetMaterialEditor(TGeoMaterial *material)
{
// Get editor for a material.
   if (!material) return;
   if (!fMaterialPanel) fMaterialPanel = new TGeoTransientPanel("Material", material);
   else {
      fMaterialPanel->SetModel(material);   
      fMaterialPanel->Show();
   }   
}

//______________________________________________________________________________
void TGeoTabManager::GetEditors(TClass *cl, TGCompositeFrame *style)
{
// Get editor for a class.
   // Look in TClass::GetEditorList() for any object deriving from TGedFrame,
   static Int_t icount = 0;
   TGedElement *ge;
   TList *list = cl->GetEditorList();
   TIter next1(list);
   // Iterate existing editors for class "cl"
   while ((ge = (TGedElement *) next1())) {
      // check if the editor ge->fGedframe is already in the list of style
      if (ge->fCanvas != (TObject*)fPad->GetCanvas()) continue;
      TGedFrame *f = ge->fGedFrame;
      TList *l = style->GetList();
      TGFrameElement *fr;
      TIter next(l);
      // Iterate all ged frames in style
      while ((fr = (TGFrameElement *) next())) 
         if (fr->fFrame->InheritsFrom(f->ClassName())) return;
   }
   TClass *class2, *class3;
   class2 = gROOT->GetClass(Form("%sEditor",cl->GetName()));
   if (class2 && class2->InheritsFrom(TGedFrame::Class())) {
      list = style->GetList();
      TGFrameElement *fr;
      TIter next(list);
      while ((fr = (TGFrameElement *) next())) if (fr->fFrame->IsA() == class2) return;
      gROOT->ProcessLine(Form("((TGCompositeFrame *)0x%lx)->AddFrame(new %s((TGWindow *)0x%lx, %d),\
                           new TGLayoutHints(kLHintsTop | kLHintsExpandX,0, 0, 2, 2))",\
                           (Long_t)style, class2->GetName(), (Long_t)style, 3000+icount));
      class3 = (TClass*)gROOT->GetListOfClasses()->FindObject(cl->GetName());
      TIter next3(class3->GetEditorList());
      while ((ge = (TGedElement *)next3())) {
         if (!strcmp(ge->fGedFrame->ClassName(), class2->GetName()) && (ge->fCanvas == 0)) {
            ge->fCanvas = (TObject*)fPad->GetCanvas();
         }
      }
   }
}

//______________________________________________________________________________
TGeoTabManager *TGeoTabManager::GetMakeTabManager(TVirtualPad *pad, TGTab *tab)
{
// Static method to return the tab manager currently appended to the pad or create one 
// if not existing.
   if (!pad) return NULL;
   // search for a tab manager in the list of primitives appended to ther pad
   TClass *cl = TGeoTabManager::Class();
   TIter next(cl->GetEditorList());
   TGeoTabManager *tabmgr;
   while ((tabmgr=(TGeoTabManager*)next())) {
      if (tabmgr->GetPad()==pad /*&& tabmgr->GetTab()==tab*/) return tabmgr;
   }
   // tab manager not found -> create one
   tabmgr = new TGeoTabManager(pad,tab);
   return tabmgr;
}   

//______________________________________________________________________________
Int_t TGeoTabManager::GetTabIndex(EGeoTabType type) const
{
// Get index for a given tab element.
   Int_t ntabs = fTab->GetNumberOfTabs();
   TString tabname;
   switch (type) {
      case kTabVolume:
         tabname = "Volume";
         break;
      default:
         return 0;
   }
                     
   TGTabElement *tel;
   for (Int_t i=0; i<ntabs; i++) {
      tel = fTab->GetTabTab(i);
      if (tel && !strcmp(tel->GetString(),tabname.Data())) return i;
   }   
   return 0;
}

//______________________________________________________________________________
void TGeoTabManager::SetEnabled(EGeoTabType type, Bool_t flag)
{
// Enable/disable tabs
   fTab->SetEnabled(GetTabIndex(type), flag);
}

//______________________________________________________________________________
void TGeoTabManager::SetModel(EGeoTabType type, TObject *model, Int_t event)
{
// Send the SetModel signal to all editors in the tab TYPE.
   TGCompositeFrame *tab;
   switch (type) {
      case kTabVolume:
         tab = fVolumeTab;
         fVolume = (TGeoVolume*)model;
         break;
      default:
         tab = (TGCompositeFrame*)((TGFrameElement*)fTab->GetTabContainer(0)->GetList()->At(0))->fFrame;
   }
   TGFrameElement *el;
   TIter next(tab->GetList());
   while ((el = (TGFrameElement *) next())) {
      if ((el->fFrame)->InheritsFrom(TGedFrame::Class())) {
         ((TGedFrame *)(el->fFrame))->SetModel(fPad, model, event);
      }   
   }
   tab->MapSubwindows();
}      

//______________________________________________________________________________
void TGeoTabManager::SetTab(EGeoTabType type)
{
// Set a given tab element as active one.
   fTab->SetTab(GetTabIndex(type));
}
   
ClassImp(TGeoTreeDialog)

TObject *TGeoTreeDialog::fgSelectedObj = 0;

//______________________________________________________________________________
TGeoTreeDialog::TGeoTreeDialog(TGFrame *caller, const TGWindow *main, UInt_t w, UInt_t h)
               :TGTransientFrame(main, main, w, h)
{
// Ctor
   fgSelectedObj = 0;
   TGCanvas *tgcv = new TGCanvas(this, 100, 200,  kSunkenFrame | kDoubleBorder);
   fLT = new TGListTree(tgcv->GetViewPort(), 100, 200);
   fLT->Associate(this);
   tgcv->SetContainer(fLT);
   AddFrame(tgcv, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 2,2,2,2));
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 100, 10, kHorizontalFrame | kLHintsExpandX);
   fObjLabel = new TGLabel(f1, "Selected: -none-");
   Pixel_t color;
   gClient->GetColorByName("#0000ff", color);
   fObjLabel->SetTextColor(color);
   fObjLabel->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fObjLabel, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 2,2,2,2));
   fClose = new TGTextButton(f1, "&Close");
   f1->AddFrame(fClose, new TGLayoutHints(kLHintsRight, 2,2,2,2)); 
   AddFrame(f1, new TGLayoutHints(kLHintsBottom | kLHintsExpandX, 2,2,2,2));
   
   Int_t ww = caller->GetWidth();
   Window_t wdum;
   Int_t    ax, ay;
   gVirtualX->TranslateCoordinates(caller->GetId(), main->GetId(), 0,0,ax,ay,wdum);
   Move(ax + ww, ay);
   SetWMPosition(ax, ay);
   
   MapSubwindows();
   Layout();
   MapWindow();
}

//______________________________________________________________________________
TGeoTreeDialog::~TGeoTreeDialog()
{
// Dtor
}

//______________________________________________________________________________
void TGeoTreeDialog::DoSelect(TGListTreeItem *item)
{
// Update dialog to reflect current clicked object.
   static char name[256];
   if (!item || !item->GetUserData()) {
      fgSelectedObj = 0;
      if (!strcmp(name, "Selected: -none-")) return;
      sprintf(name,"Selected: -none-");
      fObjLabel->SetText(name);
   }
   fgSelectedObj = (TObject *)item->GetUserData();
   if (fgSelectedObj) {
      sprintf(name, "Selected %s", fgSelectedObj->GetName());
      fObjLabel->SetText(name);
   }   
}   
      
ClassImp(TGeoVolumeDialog)

//______________________________________________________________________________
TGeoVolumeDialog::TGeoVolumeDialog(TGFrame *caller, const TGWindow *main, UInt_t w, UInt_t h)
                 :TGeoTreeDialog(caller, main, w, h)
{
// Ctor.
   SetCleanup(kDeepCleanup);
   BuildListTree();   
   ConnectSignalsToSlots();
   gClient->WaitFor(this);
}

//______________________________________________________________________________
void TGeoVolumeDialog::BuildListTree()
{
// Build volume specific list tree.
   const TGPicture *pic_fld = gClient->GetPicture("folder_t.xpm");
   const TGPicture *pic_fldo = gClient->GetPicture("ofolder_t.xpm");
   const TGPicture *pic_file = gClient->GetPicture("mdi_default.xpm");
   const TGPicture *pic_fileo = gClient->GetPicture("fileopen.xpm");
   TGListTreeItem *parent_item=0;
   TGeoVolume *parent_vol = gGeoManager->GetMasterVolume();
   TGeoVolume *vol;
   // Existing volume hierarchy
   parent_item = fLT->AddItem(parent_item, "Volume hierarchy", pic_fldo, pic_fld);
   parent_item->SetTipText("Select a volume from the existing hierarchy");
   fLT->OpenItem(parent_item);
   if (parent_vol) {
      if (!parent_vol->GetNdaughters()) {
         parent_item = fLT->AddItem(parent_item, parent_vol->GetName(), parent_vol, pic_fileo, pic_file);
         parent_item->SetTipText("Master volume");
         fLT->SetSelected(parent_item);
      } else {
         parent_item = fLT->AddItem(parent_item, parent_vol->GetName(), parent_vol, pic_fldo, pic_fld);
         parent_item->SetTipText("Master volume");
         fLT->SetSelected(parent_item);
      }
   }
   parent_item = fLT->AddItem(NULL, "Other volumes", pic_fldo, pic_fld);
   parent_item->SetTipText("Select a volume from the list of unconnected volumes");
   TIter next1(gGeoManager->GetListOfVolumes());
   Bool_t found = kFALSE;
   while ((vol=(TGeoVolume*)next1())) {
      if (vol->IsAdded()) continue;
      fLT->AddItem(parent_item, vol->GetName(), vol, pic_fileo, pic_file);
      found = kTRUE;
   }
   if (found) {    
//      fLT->OpenItem(parent_item);   
      if (!parent_vol) fLT->SetSelected(parent_item->GetFirstChild());
   }        
}

//______________________________________________________________________________
void TGeoVolumeDialog::DoClose()
{
// Handle close button.
   DeleteWindow();
}   

//______________________________________________________________________________
void TGeoVolumeDialog::DoItemClick(TGListTreeItem *item, Int_t btn)
{
// Handle item click.
   // Iterate daughters
   if (btn!=kButton1) return;
   DoSelect(item);   
   if (!item || !item->GetUserData()) return;
   const TGPicture *pic_fld = gClient->GetPicture("folder_t.xpm");
   const TGPicture *pic_fldo = gClient->GetPicture("ofolder_t.xpm");
   const TGPicture *pic_file = gClient->GetPicture("mdi_default.xpm");
   const TGPicture *pic_fileo = gClient->GetPicture("fileopen.xpm");
   TGeoVolume *parent_vol = (TGeoVolume*)item->GetUserData();
   TGeoVolume *vol;
   TGeoNode *crtnode;
   TGListTreeItem *daughter_item;
   Int_t i,j,ind,icopy;
   Int_t nd = parent_vol->GetNdaughters();
   for (i=0; i<nd; i++) {
      icopy = 0;
      crtnode = parent_vol->GetNode(i);
      vol = crtnode->GetVolume();
      // check if the volume is replicated in the parent
      ind = parent_vol->GetIndex(crtnode);
      for (j=0; j<ind; j++) if (parent_vol->GetNode(j)->GetVolume() == vol) break;
      if (i<ind) continue;
      icopy++;
      for (j=ind+1; j<nd; j++) if (parent_vol->GetNode(j)->GetVolume() == vol) icopy++;
      daughter_item = fLT->AddItem(item, ((icopy>1)?Form("%s (%i)",vol->GetName(),icopy):vol->GetName()), 
                    vol,((vol->GetNdaughters())?pic_fldo:pic_fileo), ((vol->GetNdaughters())?pic_fld:pic_file));
      if (strlen(vol->GetTitle())) daughter_item->SetTipText(vol->GetTitle());            
   }
   if (nd) gClient->NeedRedraw(fLT);
}   

//______________________________________________________________________________
void TGeoVolumeDialog::ConnectSignalsToSlots()
{
// Connect signals to slots.
   fClose->Connect("Clicked()", "TGeoVolumeDialog", this, "DoClose()");
   fLT->Connect("Clicked(TGListTreeItem *, Int_t)", "TGeoVolumeDialog", this, 
                "DoItemClick(TGListTreeItem *, Int_t)");
}

ClassImp(TGeoShapeDialog)

//______________________________________________________________________________
TGeoShapeDialog::TGeoShapeDialog(TGFrame *caller, const TGWindow *main, UInt_t w, UInt_t h)
                 :TGeoTreeDialog(caller, main, w, h)
{
// Ctor.
   SetCleanup(kDeepCleanup);
   BuildListTree();   
   ConnectSignalsToSlots();
   gClient->WaitFor(this);
}

//______________________________________________________________________________
void TGeoShapeDialog::BuildListTree()
{
// Build shape specific list tree.
   const TGPicture *pic_fld = gClient->GetPicture("folder_t.xpm");
   const TGPicture *pic_fldo = gClient->GetPicture("ofolder_t.xpm");
   const TGPicture *pic_shape;
   TGListTreeItem *parent_item=0;
   TGeoShape *shape;
   const char *shapename;
   TString fld_name;
   Int_t nshapes = gGeoManager->GetListOfShapes()->GetEntriesFast();
   if (!nshapes) return;
   // Existing shapes
   for (Int_t i=0; i<nshapes; i++) {
      shape = (TGeoShape*)gGeoManager->GetListOfShapes()->At(i);
      shapename = shape->IsA()->GetName();
      pic_shape = fClient->GetMimeTypeList()->GetIcon(shapename, kTRUE);
      fld_name = shapename;  // e.g. "TGeoBBox"
      fld_name.Remove(0,4); // remove "TGeo" part -> "BBox"
      fld_name += " Shapes";
      parent_item = fLT->FindChildByName(NULL, fld_name.Data());
      if (!parent_item) {
         parent_item = fLT->AddItem(NULL, fld_name.Data(), pic_fldo, pic_fld);
         parent_item->SetTipText(Form("List of %s shapes",fld_name.Data()));
      }
      fLT->AddItem(parent_item, shape->GetName(), shape, pic_shape, pic_shape);
   }   
}

//______________________________________________________________________________
void TGeoShapeDialog::DoClose()
{
// Handle close button.
   DeleteWindow();
}   

//______________________________________________________________________________
void TGeoShapeDialog::DoItemClick(TGListTreeItem *item, Int_t btn)
{
// Handle item click.
   // Iterate daughters
   if (btn!=kButton1) return;
   DoSelect(item);   
   if (!item || !item->GetUserData()) return;
   //gClient->NeedRedraw(fLT);
}   

//______________________________________________________________________________
void TGeoShapeDialog::ConnectSignalsToSlots()
{
// Connect signals to slots.
   fClose->Connect("Clicked()", "TGeoShapeDialog", this, "DoClose()");
   fLT->Connect("Clicked(TGListTreeItem *, Int_t)", "TGeoShapeDialog", this, 
                "DoItemClick(TGListTreeItem *, Int_t)");
}

ClassImp(TGeoMediumDialog)

//______________________________________________________________________________
TGeoMediumDialog::TGeoMediumDialog(TGFrame *caller, const TGWindow *main, UInt_t w, UInt_t h)
                 :TGeoTreeDialog(caller, main, w, h)
{
// Ctor.
   SetCleanup(kDeepCleanup);
   BuildListTree();   
   ConnectSignalsToSlots();
   gClient->WaitFor(this);
}

//______________________________________________________________________________
void TGeoMediumDialog::BuildListTree()
{
// Build volume specific list tree.
   const TGPicture *pic_med = gClient->GetPicture("geomedium_t.xpm");;
   TGeoMedium *med;
   Int_t nmed = gGeoManager->GetListOfMedia()->GetSize();
   if (!nmed) return;
   // Existing media
   for (Int_t i=0; i<nmed; i++) {
      med = (TGeoMedium*)gGeoManager->GetListOfMedia()->At(i);
      fLT->AddItem(NULL, med->GetName(), med, pic_med, pic_med);
   }   
}

//______________________________________________________________________________
void TGeoMediumDialog::DoClose()
{
// Handle close button.
   DeleteWindow();
}   

//______________________________________________________________________________
void TGeoMediumDialog::DoItemClick(TGListTreeItem *item, Int_t btn)
{
// Handle item click.
   // Iterate daughters
   if (btn!=kButton1) return;
   DoSelect(item);   
   if (!item || !item->GetUserData()) return;
   //gClient->NeedRedraw(fLT);
}   

//______________________________________________________________________________
void TGeoMediumDialog::ConnectSignalsToSlots()
{
// Connect signals to slots.
   fClose->Connect("Clicked()", "TGeoMediumDialog", this, "DoClose()");
   fLT->Connect("Clicked(TGListTreeItem *, Int_t)", "TGeoMediumDialog", this, 
                "DoItemClick(TGListTreeItem *, Int_t)");
}

ClassImp(TGeoMaterialDialog)

//______________________________________________________________________________
TGeoMaterialDialog::TGeoMaterialDialog(TGFrame *caller, const TGWindow *main, UInt_t w, UInt_t h)
                 :TGeoTreeDialog(caller, main, w, h)
{
// Ctor.
   SetCleanup(kDeepCleanup);
   BuildListTree();   
   ConnectSignalsToSlots();
   gClient->WaitFor(this);
}

//______________________________________________________________________________
void TGeoMaterialDialog::BuildListTree()
{
// Build volume specific list tree.
   const TGPicture *pic_mat = gClient->GetPicture("geomaterial_t.xpm");;
   TGeoMaterial *mat;
   Int_t nmat = gGeoManager->GetListOfMaterials()->GetSize();
   if (!nmat) return;
   // Existing materials
   for (Int_t i=0; i<nmat; i++) {
      mat = (TGeoMaterial*)gGeoManager->GetListOfMaterials()->At(i);
      fLT->AddItem(NULL, mat->GetName(), mat, pic_mat, pic_mat);
   }   
}

//______________________________________________________________________________
void TGeoMaterialDialog::DoClose()
{
// Handle close button.
   DeleteWindow();
}   

//______________________________________________________________________________
void TGeoMaterialDialog::DoItemClick(TGListTreeItem *item, Int_t btn)
{
// Handle item click.
   // Iterate daughters
   if (btn!=kButton1) return;
   DoSelect(item);   
   if (!item || !item->GetUserData()) return;
   //gClient->NeedRedraw(fLT);
}   

//______________________________________________________________________________
void TGeoMaterialDialog::ConnectSignalsToSlots()
{
// Connect signals to slots.
   fClose->Connect("Clicked()", "TGeoMaterialDialog", this, "DoClose()");
   fLT->Connect("Clicked(TGListTreeItem *, Int_t)", "TGeoMaterialDialog", this, 
                "DoItemClick(TGListTreeItem *, Int_t)");
}

ClassImp(TGeoMatrixDialog)

//______________________________________________________________________________
TGeoMatrixDialog::TGeoMatrixDialog(TGFrame *caller, const TGWindow *main, UInt_t w, UInt_t h)
                 :TGeoTreeDialog(caller, main, w, h)
{
// Ctor.
   SetCleanup(kDeepCleanup);
   BuildListTree();   
   ConnectSignalsToSlots();
   gClient->WaitFor(this);
}

//______________________________________________________________________________
void TGeoMatrixDialog::BuildListTree()
{
// Build matrix specific list tree.
   const TGPicture *pic_tr = gClient->GetPicture("geotranslation_t.xpm");
   const TGPicture *pic_rot = gClient->GetPicture("georotation_t.xpm");
   const TGPicture *pic_combi = gClient->GetPicture("geocombi_t.xpm");
   const TGPicture *pic;
   TGListTreeItem *parent_item=0;
   TGeoMatrix *matrix;
   Int_t nmat = gGeoManager->GetListOfMatrices()->GetEntriesFast();
   if (!nmat) return;
   // Existing matrices
   for (Int_t i=0; i<nmat; i++) {
      matrix = (TGeoMatrix*)gGeoManager->GetListOfMatrices()->At(i);
      if (matrix->IsIdentity()) continue;
      if (!strcmp(matrix->IsA()->GetName(),"TGeoTranslation")) {
         pic = pic_tr;
         parent_item = fLT->FindChildByName(NULL, "Translations");
         if (!parent_item) {
            parent_item = fLT->AddItem(NULL, "Translations", pic, pic);
            parent_item->SetTipText("List of translations");
         }            
      } else if (!strcmp(matrix->IsA()->GetName(),"TGeoRotation")) {
         pic = pic_rot;
         parent_item = fLT->FindChildByName(NULL, "Rotations");
         if (!parent_item) {
            parent_item = fLT->AddItem(NULL, "Rotations", pic, pic);
            parent_item->SetTipText("List of rotations");
         } 
      } else if (!strcmp(matrix->IsA()->GetName(),"TGeoCombiTrans") ||
                  !strcmp(matrix->IsA()->GetName(),"TGeoHMatrix")) {
         pic = pic_combi;
         parent_item = fLT->FindChildByName(NULL, "Combined");
         if (!parent_item) {
            parent_item = fLT->AddItem(NULL, "Combined", pic, pic);
            parent_item->SetTipText("List of combined transformations");
         } 
      } else continue;
      fLT->AddItem(parent_item, matrix->GetName(), matrix, pic, pic);
   }   
}

//______________________________________________________________________________
void TGeoMatrixDialog::DoClose()
{
// Handle close button.
   DeleteWindow();
}   

//______________________________________________________________________________
void TGeoMatrixDialog::DoItemClick(TGListTreeItem *item, Int_t btn)
{
// Handle item click.
   // Iterate daughters
   if (btn!=kButton1) return;
   DoSelect(item);   
   if (!item || !item->GetUserData()) return;
   //gClient->NeedRedraw(fLT);
}   

//______________________________________________________________________________
void TGeoMatrixDialog::ConnectSignalsToSlots()
{
// Connect signals to slots.
   fClose->Connect("Clicked()", "TGeoMatrixDialog", this, "DoClose()");
   fLT->Connect("Clicked(TGListTreeItem *, Int_t)", "TGeoMatrixDialog", this, 
                "DoItemClick(TGListTreeItem *, Int_t)");
}

ClassImp(TGeoTransientPanel)

//______________________________________________________________________________
TGeoTransientPanel::TGeoTransientPanel(const char *name, TObject *obj)
                   :TGMainFrame(gClient->GetRoot(),175,20)
{
// Transient panel ctor.
   fModel = obj;
   TGCanvas *can = new TGCanvas(this, 170, 100);
   fTab = new TGTab(can->GetViewPort(), 10, 10);
   can->SetContainer(fTab);
   AddFrame(can, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));
   fTab->Associate(can);
   fTabContainer = fTab->AddTab(name);
   fStyle = new TGCompositeFrame(fTabContainer, 110, 30, kVerticalFrame);
   fTabContainer->AddFrame(fStyle, new TGLayoutHints(kLHintsTop | kLHintsExpandX,\
                                                     5, 0, 2, 2));
   TString wname = name;
   wname += " Editor";
   SetWindowName(wname.Data());
   SetModel(fModel);
   fClose = new TGTextButton(this, "Close");
   AddFrame(fClose, new TGLayoutHints(kLHintsBottom | kLHintsRight, 0,10,5,5));
   MapSubwindows();
   Layout();
   Resize(fTabContainer->GetDefaultWidth()+30, fTabContainer->GetDefaultHeight()+65);
   MapWindow();
   gROOT->GetListOfCleanups()->Add(this);
   fClose->Connect("Clicked()", "TGeoTransientPanel", this, "Hide()");
} 

//______________________________________________________________________________
TGeoTransientPanel::~TGeoTransientPanel()
{
// Destructor.
   gROOT->GetListOfCleanups()->Remove(this);
   Cleanup(); 
}

//______________________________________________________________________________
void TGeoTransientPanel::CloseWindow()
{
   // When closed via WM close button, just unmap (i.e. hide) editor
   // for later use.
   UnmapWindow();
   gROOT->GetListOfCleanups()->Remove(this);
}

//______________________________________________________________________________
void TGeoTransientPanel::GetEditors(TClass *cl)
{
// Get editor for a class.
   // Look in TClass::GetEditorList() for any object deriving from TGedFrame,
   static Int_t icount = 0;
   TGedElement *ge;
   TList *list = cl->GetEditorList();
   TIter next1(list);
   while ((ge = (TGedElement *) next1())) {
      // check if the editor ge->fGedframe is already in the list of style
      if (ge->fCanvas != (TObject*)gPad->GetCanvas()) continue;
      TGedFrame *f = ge->fGedFrame;
      TList *l = fStyle->GetList();
      TGFrameElement *fr;
      TIter next(l);
      while ((fr = (TGFrameElement *) next())) 
         if (fr->fFrame->InheritsFrom(f->ClassName())) return;
   }
   TClass *class2, *class3;
   class2 = gROOT->GetClass(Form("%sEditor",cl->GetName()));
   if (class2 && class2->InheritsFrom(TGedFrame::Class())) {
      list = fStyle->GetList();
      TGFrameElement *fr;
      TIter next(list);
      while ((fr = (TGFrameElement *) next()))
         if (fr->fFrame->IsA() == class2) return;
      gROOT->ProcessLine(Form("((TGCompositeFrame *)0x%lx)->AddFrame(new %s((TGWindow *)0x%lx, %d),\
                              new TGLayoutHints(kLHintsTop | kLHintsExpandX,0, 0, 2, 2))",\
                              (Long_t)fStyle, class2->GetName(), (Long_t)fStyle, 3000+icount));
      class3 = (TClass*)gROOT->GetListOfClasses()->FindObject(cl->GetName());
      TIter next3(class3->GetEditorList());
      while ((ge = (TGedElement *)next3())) {
         if (!strcmp(ge->fGedFrame->ClassName(), class2->GetName()) && (ge->fCanvas == 0)) {
            ge->fCanvas = (TObject*)gPad->GetCanvas();
         }
      }
   }
}

//______________________________________________________________________________
void TGeoTransientPanel::SetModel(TObject *model, Int_t event)
{
// Update the editors in the main tab to reflect the selected object.
   if (!model) return;
   fModel = model;
   GetEditors(model->IsA());
   TGFrameElement *el;
   TIter next(fStyle->GetList());
   while ((el = (TGFrameElement *) next())) {
      if ((el->fFrame)->InheritsFrom(TGedFrame::Class())) {
         ((TGedFrame *)(el->fFrame))->SetModel(gPad, model, event);
      }   
   }
   Resize(fTabContainer->GetDefaultWidth()+30, fTabContainer->GetDefaultHeight()+65);
}

//______________________________________________________________________________
void TGeoTransientPanel::Hide()
{
// Hide the transient frame
   UnmapWindow();
}

//______________________________________________________________________________
void TGeoTransientPanel::Show()
{
// Hide the transient frame
   MapWindow();
}

//______________________________________________________________________________
void TGeoTransientPanel::DeleteEditors()
{
}

   
