// @(#)root/ged:$Name:  $:$Id: TGedFrame.h,v 1.4 2004/09/15 14:56:35 brun Exp $
// Author: Ilka  Antcheva 10/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedFrame
#define ROOT_TGedFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGedFrame                                                           //
//                                                                      //
//  Base editor's attribute frame - a service class.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TVirtualPad;
class TCanvas;
class TGLabel;
class TList;
class TGTab;


class TGedFrame : public TGCompositeFrame, public TGWidget {
protected:
   TObject      *fModel;    //selected object, if exists
   TVirtualPad  *fPad;      //selected pad, if exists
   Bool_t        fInit;     //init flag for setting signals/slots
   TGTab        *fTab;      //pointer to the parent tab          

   virtual void MakeTitle(const char *title);
   
public:
   TGedFrame(const TGWindow *p, Int_t id,
             Int_t width = 140, Int_t height = 30,
             UInt_t options = kChildFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedFrame();

   TObject          *GetModel() const { return fModel;}
   TVirtualPad      *GetPad() const { return fPad;}
   virtual Option_t *GetDrawOption() const;
   virtual void      RecursiveRemove(TObject *obj);
   virtual void      Refresh();
   virtual void      SetActive(Bool_t active = kTRUE);
   virtual void      SetDrawOption(Option_t *option="");
   virtual void      SetModel(TVirtualPad *pad, TObject *obj, Int_t event) = 0;
   virtual void      Update();

   ClassDef(TGedFrame, 0); //base editor's frame
};


// The GUI editors and corresponding canvases will be registered 
// in the list TClass::fClassEditors via the class TGedElement

class TGedElement : public TObject {
public:
   TGedFrame  *fGedFrame;   //object editor
   TObject    *fCanvas;     //connected canvas (0 if disconnected)
   
   ClassDef(TGedElement, 0); //editor element
};


class TGedNameFrame : public TGedFrame {
protected:
   TGLabel          *fLabel;      //label of attribute frame
   TGCompositeFrame *f1, *f2;     //container frames

public:
   TGedNameFrame(const TGWindow *p, Int_t id,
                 Int_t width = 140, Int_t height = 30,
                 UInt_t options = kChildFrame,
                 Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedNameFrame();

   virtual void  SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   ClassDef(TGedNameFrame,0)  //frame showing the selected object name
};

#endif
