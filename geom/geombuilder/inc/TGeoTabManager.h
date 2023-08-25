// @(#):$Id$
// Author: M.Gheata

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTabManager
#define ROOT_TGeoTabManager

#include "TGFrame.h"

#include "TMap.h"

class TVirtualPad;
class TClass;

class TList;
class TGCompositeFrame;
class TGLabel;
class TGTab;
class TGComboBox;
class TGListTree;
class TGListTreeItem;
class TGCanvas;

class TGedEditor;

class TGeoShape;
class TGeoVolume;
class TGeoMedium;
class TGeoMaterial;
class TGeoMatrix;

class TGeoTreeDialog;
class TGeoTransientPanel;

class TGeoTabManager : public TObject {
   friend class TGeoManagerEditor;

private:
   TGedEditor *fGedEditor;             // Parent editor
   TVirtualPad *fPad;                  // Pad to which this applies
   TGTab *fTab;                        // Parent tab
   TGeoVolume *fVolume;                // Edited volume
   TGeoTransientPanel *fShapePanel;    // Panel for editing shapes
   TGeoTransientPanel *fMediumPanel;   // Panel for editing media
   TGeoTransientPanel *fMaterialPanel; // Panel for editing materials
   TGeoTransientPanel *fMatrixPanel;   // Panel for editing matrices
   TGCompositeFrame *fVolumeTab;       // Volume tab

   static TMap fgEditorToMgrMap; // Map from ged-editor to associated tab-manager

   void GetEditors(TClass *cl);

public:
   TGeoTabManager(TGedEditor *ged);
   ~TGeoTabManager() override;

   static TGeoTabManager *GetMakeTabManager(TGedEditor *ged);
   static void Cleanup(TGCompositeFrame *frame);
   TVirtualPad *GetPad() const { return fPad; }
   TGTab *GetTab() const { return fTab; }
   Int_t GetTabIndex() const;
   static void MoveFrame(TGCompositeFrame *fr, TGCompositeFrame *p);
   void SetVolTabEnabled(Bool_t flag = kTRUE);
   void SetModel(TObject *model);
   void SetTab();

   void GetShapeEditor(TGeoShape *shape);
   void GetVolumeEditor(TGeoVolume *vol);
   void GetMatrixEditor(TGeoMatrix *matrix);
   void GetMediumEditor(TGeoMedium *medium);
   void GetMaterialEditor(TGeoMaterial *material);

   TGCompositeFrame *GetVolumeTab() const { return fVolumeTab; }
   TGeoVolume *GetVolume() const { return fVolume; }

   ClassDefOverride(TGeoTabManager, 0) // Tab manager for geometry editors
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTreeDialog                                                      //
//                                                                      //
//  Dialog frame for selecting objects with a tree hierarchy            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoTreeDialog : public TGTransientFrame {

protected:
   static TObject *fgSelectedObj; // Selected object
   TGCanvas *fCanvas;             // TGCanvas containing the list tree
   TGLabel *fObjLabel;            // Label for selected object
   TGListTree *fLT;               // List tree for selecting
   TGCompositeFrame *f1;          // Composite frame containing the selection
   TGTextButton *fClose;          // Close button

   virtual void BuildListTree() = 0;
   virtual void ConnectSignalsToSlots() = 0;

public:
   TGeoTreeDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   ~TGeoTreeDialog() override;

   static TObject *GetSelected();
   // Slots
   virtual void DoClose() = 0;
   virtual void DoItemClick(TGListTreeItem *item, Int_t btn) = 0;
   void DoSelect(TGListTreeItem *item);

   ClassDefOverride(TGeoTreeDialog, 0) // List-Tree based dialog
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoVolumeDialog                                                    //
//                                                                      //
//  Special tree dialog class for selecting volumes.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoVolumeDialog : public TGeoTreeDialog {

protected:
   void BuildListTree() override;
   void ConnectSignalsToSlots() override;

public:
   TGeoVolumeDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   ~TGeoVolumeDialog() override {}

   // Slots
   void DoClose() override;
   void DoItemClick(TGListTreeItem *item, Int_t btn) override;

   ClassDefOverride(TGeoVolumeDialog, 0) // List-Tree based volume dialog
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoShapeDialog                                                     //
//                                                                      //
//  Special tree dialog class for selecting shapes.                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoShapeDialog : public TGeoTreeDialog {

protected:
   void BuildListTree() override;
   void ConnectSignalsToSlots() override;

public:
   TGeoShapeDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   ~TGeoShapeDialog() override {}

   // Slots
   void DoClose() override;
   void DoItemClick(TGListTreeItem *item, Int_t btn) override;

   ClassDefOverride(TGeoShapeDialog, 0) // List-Tree based shape dialog
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoMediumDialog                                                    //
//                                                                      //
//  Special tree dialog class for selecting media.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoMediumDialog : public TGeoTreeDialog {

protected:
   void BuildListTree() override;
   void ConnectSignalsToSlots() override;

public:
   TGeoMediumDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   ~TGeoMediumDialog() override {}

   // Slots
   void DoClose() override;
   void DoItemClick(TGListTreeItem *item, Int_t btn) override;

   ClassDefOverride(TGeoMediumDialog, 0) // List-Tree based medium dialog
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoMaterialDialog                                                  //
//                                                                      //
//  Special tree dialog class for selecting materials.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoMaterialDialog : public TGeoTreeDialog {

protected:
   void BuildListTree() override;
   void ConnectSignalsToSlots() override;

public:
   TGeoMaterialDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   ~TGeoMaterialDialog() override {}

   // Slots
   void DoClose() override;
   void DoItemClick(TGListTreeItem *item, Int_t btn) override;

   ClassDefOverride(TGeoMaterialDialog, 0) // List-Tree based material dialog
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoMatrixDialog                                                    //
//                                                                      //
//  Special tree dialog class for selecting matrices.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoMatrixDialog : public TGeoTreeDialog {

protected:
   void BuildListTree() override;
   void ConnectSignalsToSlots() override;

public:
   TGeoMatrixDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   ~TGeoMatrixDialog() override {}

   // Slots
   void DoClose() override;
   void DoItemClick(TGListTreeItem *item, Int_t btn) override;

   ClassDefOverride(TGeoMatrixDialog, 0) // List-Tree based matrix dialog
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTransientPanel                                                  //
//                                                                      //
//  Special transient tab holding TGeo editors.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGeoTransientPanel : public TGMainFrame {
   TGedEditor *fGedEditor;          // ged-editor steering this panel
   TGCanvas *fCan;                  // TGCanvas containing a TGTab
   TGTab *fTab;                     // tab widget holding the editor
   TGCompositeFrame *fTabContainer; // main tab container
   TGCompositeFrame *fStyle;        // style tab container frame
   TObject *fModel;                 // selected object
   TGTextButton *fClose;            // close button

public:
   TGeoTransientPanel(TGedEditor *ged, const char *name, TObject *obj);
   ~TGeoTransientPanel() override;

   void CloseWindow() override;
   virtual void DeleteEditors();

   TGTab *GetTab() const { return fTab; }
   TGCompositeFrame *GetStyle() const { return fStyle; }
   TObject *GetModel() const { return fModel; }

   void GetEditors(TClass *cl);
   virtual void Hide();
   virtual void Show();
   void SetModel(TObject *model);

   ClassDefOverride(TGeoTransientPanel, 0) // List-Tree based dialog
};

#endif
