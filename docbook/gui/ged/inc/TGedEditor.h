// @(#)root/ged:$Id$
// Author: Marek Biskup, Ilka Antcheva   02/12/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedEditor
#define ROOT_TGedEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedEditor                                                           //
//                                                                      //
// Editor is a composite frame that contains GUI for editting objects   //
// in a canvas. It looks for the class ROOT_classname + 'Editor'.       //
//                                                                      //
// It connects to a Canvas and listens for selected objects             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TVirtualPadEditor
#include "TVirtualPadEditor.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TMap
#include "TMap.h"
#endif

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
   TGedEditor(const TGedEditor&);            // Not implemented
   TGedEditor& operator=(const TGedEditor&); // Not implemented

protected:
   TMap              fFrameMap;         //global map of available frames
   TMap              fExclMap;          //map of excluded editors for selected model
   TList             fGedFrames;        //list visible of frames 

   TGCanvas         *fCan;              //provides scroll bars
   TGTab            *fTab;              //tab widget holding the editor

   TList             fCreatedTabs;      //list of created tabs
   TList             fVisibleTabs;      //list ofcurrently used tabs
   TGCompositeFrame *fTabContainer;     //main tab container

   TObject          *fModel;            //selected object
   TVirtualPad      *fPad;              //selected pad
   TCanvas          *fCanvas;           //canvas related to the editor
   TClass           *fClass;            //class of the selected object
   Bool_t            fGlobal;           //true if editor is global

   void              ConfigureGedFrames(Bool_t objChaged);

   virtual TGedFrame* CreateNameFrame(const TGWindow* parent, const char* tab_name); 

   static TGedEditor *fgFrameCreator; 

public:
   TGedEditor(TCanvas* canvas = 0, UInt_t width = 175, UInt_t height = 20);
   virtual ~TGedEditor();

   void          PrintFrameStat();
   virtual void  Update(TGedFrame* frame = 0);
   void          ReinitWorkspace();
   void          ActivateEditor (TClass* cl, Bool_t recurse);
   void          ActivateEditors(TList* bcl, Bool_t recurse);
   void          ExcludeClassEditor(TClass* cl, Bool_t recurse = kFALSE);
   void          InsertGedFrame(TGedFrame* f);

   TGCanvas*                 GetTGCanvas() const { return fCan; }
   TGTab*                    GetTab()      const { return fTab; }
   virtual TGCompositeFrame* GetEditorTab(const char* name);
   virtual TGedTabInfo*      GetEditorTabInfo(const char* name);

   virtual TCanvas*          GetCanvas() const { return fCanvas; }
   virtual TVirtualPad*      GetPad()    const { return fPad; }
   virtual TObject*          GetModel()  const { return fModel; }


   virtual void   CloseWindow();
   virtual void   ConnectToCanvas(TCanvas *c);
   virtual void   DisconnectFromCanvas();
   virtual Bool_t IsGlobal() const  { return fGlobal; } 
   virtual void   Hide();
   virtual void   GlobalClosed();
   virtual void   SetCanvas(TCanvas *c);
   virtual void   SetGlobal(Bool_t global);
   virtual void   GlobalSetModel(TVirtualPad *, TObject *, Int_t);
   virtual void   SetModel(TVirtualPad* pad, TObject* obj, Int_t event);
   virtual void   Show();
   virtual void   RecursiveRemove(TObject* obj);

   static TGedEditor* GetFrameCreator();
   static void SetFrameCreator(TGedEditor* e);

   ClassDef(TGedEditor,0)  // ROOT graphics editor
};

#endif
