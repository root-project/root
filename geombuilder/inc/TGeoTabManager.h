// @(#):$Name:  $:$Id: Exp $
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

class TGeoShape;
class TGeoVolume;
class TGeoMedium;
class TGeoMaterial;
class TGeoMatrix;

class TGeoTreeDialog;
class TGeoTransientPanel;

class TGeoTabManager : public TObject {
public:
   enum EGeoTabType {
//      kTabShape, 
      kTabVolume
//      kTabMatrix,
//      kTabMedium,
//      kTabMaterial
   };   
private:
   TVirtualPad         *fPad;               // Pad to which this applies
   TGTab               *fTab;               // Parent tab
//   TGeoShape           *fShape;             // Edited shape
   TGeoVolume          *fVolume;            // Edited volume
//   TGeoMedium          *fMedium;            // Edited medium
//   TGeoMaterial        *fMaterial;          // Edited material
//   TGeoMatrix          *fMatrix;            // Edited matrix
//   TGCompositeFrame    *fShapeTab;          // Shape tab
//   TGCompositeFrame    *fShapeCont;         // Shape tab container
   TGeoTransientPanel  *fShapePanel;        // Panel for editing shapes
   TGeoTransientPanel  *fMediumPanel;       // Panel for editing media
   TGeoTransientPanel  *fMaterialPanel;     // Panel for editing materials
   TGeoTransientPanel  *fMatrixPanel;       // Panel for editing matrices
   TGCompositeFrame    *fVolumeTab;         // Volume tab
   TGCompositeFrame    *fVolumeCont;        // Volume tab container
//   TGCompositeFrame    *fMatrixTab;         // Matrix tab
//   TGCompositeFrame    *fMatrixCont;        // Matrix tab container
//   TGCompositeFrame    *fMediumTab;         // Medium tab
//   TGCompositeFrame    *fMediumCont;        // Medium tab container
//   TGCompositeFrame    *fMaterialTab;       // Material tab
//   TGCompositeFrame    *fMaterialCont;      // Material tab container
   TList               *fShapeCombos;       // List of combo boxes refering to shapes
//   TList               *fVolumeCombos;      // List of combo boxes refering to volumes
   TList               *fMatrixCombos;      // List of combo boxes refering to matrices
   TList               *fMediumCombos;      // List of combo boxes refering to media
   TList               *fMaterialCombos;    // List of combo boxes refering to materials

   void                CreateTabs();
   void                GetEditors(TClass *cl, TGCompositeFrame *style);
public:
   TGeoTabManager(TVirtualPad *pad, TGTab *tab);
   virtual ~TGeoTabManager();

   static TGeoTabManager *GetMakeTabManager(TVirtualPad *pad, TGTab *tab);   
   TVirtualPad        *GetPad() const {return fPad;}
   TGTab              *GetTab() const {return fTab;}
   Int_t               GetTabIndex(EGeoTabType type) const;
   void                SetEnabled(EGeoTabType type, Bool_t flag=kTRUE);
   void                SetModel(EGeoTabType type, TObject *model, Int_t event=0);
   void                SetTab(EGeoTabType type);
   
   void                AddComboShape(TGComboBox *combo);
   void                AddShape(const char *name, Int_t id);
   void                UpdateShape(Int_t id);
//   void                AddComboVolume(TGComboBox *combo);
//   void                AddVolume(const char *name, Int_t id);
//   void                UpdateVolume(Int_t id);
   void                AddComboMatrix(TGComboBox *combo);
   void                AddMatrix(const char *name, Int_t id);
   void                UpdateMatrix(Int_t id);
   void                AddComboMedium(TGComboBox *combo);
   void                AddMedium(const char *name, Int_t id);
   void                UpdateMedium(Int_t id);
   void                AddComboMaterial(TGComboBox *combo);
   void                AddMaterial(const char *name, Int_t id);
   void                UpdateMaterial(Int_t id);

   void                GetShapeEditor(TGeoShape *shape);
   void                GetVolumeEditor(TGeoVolume *vol);
   void                GetMatrixEditor(TGeoMatrix *matrix);
   void                GetMediumEditor(TGeoMedium *medium);
   void                GetMaterialEditor(TGeoMaterial *material);


//   TGCompositeFrame   *GetShapeTab()     const {return fShapeTab;}
//   TGCompositeFrame   *GetShapeCont()    const {return fShapeCont;}
   TGCompositeFrame   *GetVolumeTab()    const {return fVolumeTab;}
   TGCompositeFrame   *GetVolumeCont()   const {return fVolumeCont;}
//   TGCompositeFrame   *GetMatrixTab()    const {return fMatrixTab;}
//   TGCompositeFrame   *GetMatrixCont()   const {return fMatrixCont;}
//   TGCompositeFrame   *GetMediumTab()    const {return fMediumTab;}
//   TGCompositeFrame   *GetMediumCont()   const {return fMediumCont;}
//   TGCompositeFrame   *GetMaterialTab()  const {return fMaterialTab;}
//   TGCompositeFrame   *GetMaterialCont() const {return fMaterialCont;}
//   TGeoShape          *GetShape() const    {return fShape;}
   TGeoVolume         *GetVolume() const   {return fVolume;}
//   TGeoMedium         *GetMedium() const   {return fMedium;}
//   TGeoMaterial       *GetMaterial() const {return fMaterial;}
//   TGeoMatrix         *GetMatrix() const   {return fMatrix;}

   ClassDef(TGeoTabManager, 0)   // Tab manager for geometry editors
};

class TGeoTreeDialog : public TGTransientFrame {

protected:
   static TObject     *fgSelectedObj;       // Selected object
   TGLabel            *fObjLabel;           // Label for selected object
   TGListTree         *fLT;                 // List tree for selecting
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
