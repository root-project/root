// @(#)root/ged:$Id$
// Author: Marek Biskup, Ilka Antcheva   02/12/2003

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedEditor
#define ROOT_TGedEditor


#include "TGFrame.h"
#include "TVirtualPadEditor.h"
#include "TList.h"
#include "TMap.h"

class TCanvas;
class TGCanvas;
class TGTab;
class TGTabElement;
class TVirtualPad;
class TGedFrame;
class TGedNameFrame;
class TGedTabInfo;

class TGedEditor : public TVirtualPadEditor, public TGMainFrame
{
private:
   TGedEditor(const TGedEditor&) = delete;
   TGedEditor& operator=(const TGedEditor&) = delete;

protected:
   TMap              fFrameMap;         ///< global map of available frames
   TMap              fExclMap;          ///< map of excluded editors for selected model
   TList             fGedFrames;        ///< list visible of frames

   TGCanvas         *fCan;              ///< provides scroll bars
   TGTab            *fTab;              ///< tab widget holding the editor

   TList             fCreatedTabs;      ///< list of created tabs
   TList             fVisibleTabs;      ///< list ofcurrently used tabs
   TGCompositeFrame *fTabContainer;     ///< main tab container

   TObject          *fModel;            ///< selected object
   TVirtualPad      *fPad;              ///< selected pad
   TCanvas          *fCanvas;           ///< canvas related to the editor
   TClass           *fClass;            ///< class of the selected object
   Bool_t            fGlobal;           ///< true if editor is global

   void              ConfigureGedFrames(Bool_t objChaged);

   virtual TGedFrame* CreateNameFrame(const TGWindow* parent, const char* tab_name);

   static TGedEditor *fgFrameCreator;

public:
   TGedEditor(TCanvas* canvas = nullptr, UInt_t width = 175, UInt_t height = 20);
   virtual ~TGedEditor();

   void          PrintFrameStat();
   virtual void  Update(TGedFrame* frame = nullptr);
   void          ReinitWorkspace();
   void          ActivateEditor (TClass* cl, Bool_t recurse);
   void          ActivateEditors(TList* bcl, Bool_t recurse);
   void          ExcludeClassEditor(TClass* cl, Bool_t recurse = kFALSE);
   void          InsertGedFrame(TGedFrame* f);

   TGCanvas*                 GetTGCanvas() const { return fCan; }
   TGTab*                    GetTab()      const { return fTab; }
   virtual TGCompositeFrame* GetEditorTab(const char* name);
   virtual TGedTabInfo*      GetEditorTabInfo(const char* name);

   TCanvas*                  GetCanvas() const override { return fCanvas; }
   virtual TVirtualPad*      GetPad()    const { return fPad; }
   virtual TObject*          GetModel()  const { return fModel; }


   void           CloseWindow() override;
   virtual void   ConnectToCanvas(TCanvas *c);
   virtual void   DisconnectFromCanvas();
   Bool_t         IsGlobal() const  override { return fGlobal; }
   void           Hide() override;
   virtual void   GlobalClosed();
   virtual void   SetCanvas(TCanvas *c);
   void           SetGlobal(Bool_t global) override;
   virtual void   GlobalSetModel(TVirtualPad *, TObject *, Int_t);
   virtual void   SetModel(TVirtualPad* pad, TObject* obj, Int_t event, Bool_t force=kFALSE);
   void           Show() override;
   void           RecursiveRemove(TObject* obj) override;

   static TGedEditor* GetFrameCreator();
   static void SetFrameCreator(TGedEditor* e);

   ClassDefOverride(TGedEditor,0)  // ROOT graphics editor
};

#endif
