// @(#):$Name:  $:$Id: TGeoTabManager.h,v 1.1 2006/06/13 15:27:11 brun Exp $
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTabManager                                                      //
//                                                                      //
//  Manager for editor tabs                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

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

class TGeoShape;
class TGeoVolume;
class TGeoMedium;
class TGeoMaterial;
class TGeoMatrix;

class TGeoTreeDialog;
class TGeoTransientPanel;

class TGeoTabManager : public TObject {

private:
   TVirtualPad         *fPad;               // Pad to which this applies
   TGTab               *fTab;               // Parent tab
   TGeoVolume          *fVolume;            // Edited volume
   TGeoTransientPanel  *fShapePanel;        // Panel for editing shapes
   TGeoTransientPanel  *fMediumPanel;       // Panel for editing media
   TGeoTransientPanel  *fMaterialPanel;     // Panel for editing materials
   TGeoTransientPanel  *fMatrixPanel;       // Panel for editing matrices
   TGCompositeFrame    *fVolumeTab;         // Volume tab
   TGCompositeFrame    *fVolumeCont;        // Volume tab container

   void                CreateTabs();
   void                GetEditors(TClass *cl, TGCompositeFrame *style);
public:
   TGeoTabManager(TVirtualPad *pad, TGTab *tab);
   virtual ~TGeoTabManager();

   static TGeoTabManager *GetMakeTabManager(TVirtualPad *pad, TGTab *tab); 
   static void         Cleanup(TGCompositeFrame *frame);  
   TVirtualPad        *GetPad() const {return fPad;}
   TGTab              *GetTab() const {return fTab;}
   Int_t               GetTabIndex() const;
   void                SetVolTabEnabled(Bool_t flag=kTRUE);
   void                SetModel(TObject *model, Int_t event=0);
   void                SetTab();
   
   void                GetShapeEditor(TGeoShape *shape);
   void                GetVolumeEditor(TGeoVolume *vol);
   void                GetMatrixEditor(TGeoMatrix *matrix);
   void                GetMediumEditor(TGeoMedium *medium);
   void                GetMaterialEditor(TGeoMaterial *material);


   TGCompositeFrame   *GetVolumeTab()    const {return fVolumeTab;}
   TGCompositeFrame   *GetVolumeCont()   const {return fVolumeCont;}
   TGeoVolume         *GetVolume() const   {return fVolume;}

   ClassDef(TGeoTabManager, 0)   // Tab manager for geometry editors
};

class TGeoTreeDialog : public TGTransientFrame {

protected:
   static TObject     *fgSelectedObj;       // Selected object
   TGCanvas           *fCanvas;             // TGCanvas containing the list tree
   TGLabel            *fObjLabel;           // Label for selected object
   TGListTree         *fLT;                 // List tree for selecting
   TGCompositeFrame   *f1;                  // Composite frame containing the selection
   TGTextButton       *fClose;              // Close button

   virtual void        BuildListTree() = 0;
   virtual void        ConnectSignalsToSlots() = 0;
public:
   TGeoTreeDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   virtual ~TGeoTreeDialog();
   
   static TObject     *GetSelected() {return fgSelectedObj;}
   // Slots
   virtual void        DoClose() = 0;
   virtual void        DoItemClick(TGListTreeItem *item, Int_t btn) = 0;
   void                DoSelect(TGListTreeItem *item);

   ClassDef(TGeoTreeDialog, 0)   // List-Tree based dialog
};

class TGeoVolumeDialog : public TGeoTreeDialog {

protected:
   virtual void        BuildListTree();
   virtual void        ConnectSignalsToSlots();

public:
   TGeoVolumeDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   virtual ~TGeoVolumeDialog() {;}
   
   // Slots
   virtual void        DoClose();
   virtual void        DoItemClick(TGListTreeItem *item, Int_t btn);

   ClassDef(TGeoVolumeDialog, 0)   // List-Tree based volume dialog
};

class TGeoShapeDialog : public TGeoTreeDialog {

protected:
   virtual void        BuildListTree();
   virtual void        ConnectSignalsToSlots();

public:
   TGeoShapeDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   virtual ~TGeoShapeDialog() {;}
   
   // Slots
   virtual void        DoClose();
   virtual void        DoItemClick(TGListTreeItem *item, Int_t btn);

   ClassDef(TGeoShapeDialog, 0)   // List-Tree based shape dialog
};

class TGeoMediumDialog : public TGeoTreeDialog {

protected:
   virtual void        BuildListTree();
   virtual void        ConnectSignalsToSlots();

public:
   TGeoMediumDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   virtual ~TGeoMediumDialog() {;}
   
   // Slots
   virtual void        DoClose();
   virtual void        DoItemClick(TGListTreeItem *item, Int_t btn);

   ClassDef(TGeoMediumDialog, 0)   // List-Tree based medium dialog
};

class TGeoMaterialDialog : public TGeoTreeDialog {

protected:
   virtual void        BuildListTree();
   virtual void        ConnectSignalsToSlots();

public:
   TGeoMaterialDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   virtual ~TGeoMaterialDialog() {;}
   
   // Slots
   virtual void        DoClose();
   virtual void        DoItemClick(TGListTreeItem *item, Int_t btn);

   ClassDef(TGeoMaterialDialog, 0)   // List-Tree based material dialog
};

class TGeoMatrixDialog : public TGeoTreeDialog {

protected:
   virtual void        BuildListTree();
   virtual void        ConnectSignalsToSlots();

public:
   TGeoMatrixDialog(TGFrame *caller, const TGWindow *main, UInt_t w = 1, UInt_t h = 1);
   virtual ~TGeoMatrixDialog() {;}
   
   // Slots
   virtual void        DoClose();
   virtual void        DoItemClick(TGListTreeItem *item, Int_t btn);

   ClassDef(TGeoMatrixDialog, 0)   // List-Tree based matrix dialog
};

class TGeoTransientPanel : public TGMainFrame {
   TGCanvas         *fCan;              // TGCanvas containing a TGTab
   TGTab            *fTab;              //tab widget holding the editor
   TGCompositeFrame *fTabContainer;     //main tab container
   TGCompositeFrame *fStyle;            //style tab container frame
   TObject          *fModel;            //selected object
   TGTextButton     *fClose;            //close button
   
public:
   TGeoTransientPanel(const char *name, TObject *obj);
   virtual ~TGeoTransientPanel();
   
   virtual void        CloseWindow();
   virtual void        DeleteEditors();
   
   TGTab              *GetTab() const {return fTab;}
   TGCompositeFrame   *GetStyle() const {return fStyle;}
   TObject            *GetModel() const {return fModel;}

   void                GetEditors(TClass *cl);
   virtual void        Hide();
   virtual void        Show();
   void                SetModel(TObject *model, Int_t event=0);

   ClassDef(TGeoTransientPanel, 0)   // List-Tree based dialog
};

#endif
