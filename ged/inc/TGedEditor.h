// @(#)root/ged:$Name:  $:$Id: TGedEditor.h,v 1.4 2004/06/18 15:50:43 brun Exp $
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

class TCanvas;
class TGTab;
class TVirtualPad;
class TGedFrame;
class TGedElement;

class TGedEditor : public TVirtualPadEditor, public TGMainFrame {

protected:
   TGTab            *fTab;              //tab widget holding the editor
   TGCompositeFrame *fTabContainer;     //main tab container
   TGCompositeFrame *fStyle;            //style tab container frame
   TObject          *fModel;            //selected object
   TVirtualPad      *fPad;              //selected pad
   TCanvas          *fCanvas;           //canvas related to the editor
   TClass           *fClass;            //class of the selected object
   Int_t             fWid;              //widget id 
   
   virtual void GetEditors();
   virtual void GetClassEditor(TClass *cl);
   virtual void GetBaseClassEditor(TClass *cl);

public:
   TGedEditor(TCanvas* canvas = 0);
   virtual ~TGedEditor();

   virtual void CloseWindow();
   virtual void ConnectToCanvas(TCanvas *c);
   virtual void SetModel(TVirtualPad* pad, TObject* obj, Int_t event);

   virtual void Show();
   virtual void Hide();
   virtual void DeleteEditors();
   virtual void RecursiveRemove(TObject* obj);
   
   ClassDef(TGedEditor,0)  //new editor 
};

#endif
