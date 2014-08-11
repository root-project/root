// @(#)root/ged:$Id$
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

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TVirtualPad;
class TCanvas;
class TGLabel;
class TGToolTip;
class TList;
class TGTab;
class TGedEditor;

class TGedFrame : public TGCompositeFrame {

public:
   // Inner class to store information for each extra tab.
   class TGedSubFrame : public TObject {
   private:
      TGedSubFrame(const TGedSubFrame&);            // Not implemented
      TGedSubFrame& operator=(const TGedSubFrame&); // Not implemented
   public:
      TString            fName;
      TGCompositeFrame  *fFrame;

      TGedSubFrame(TString n,  TGCompositeFrame* f) : fName(n), fFrame(f) {}
   };

private:
   TGedFrame(const TGedFrame&);            // Not implemented
   TGedFrame& operator=(const TGedFrame&); // Not implemented

protected:
   Bool_t          fInit;        // init flag for setting signals/slots
   TGedEditor     *fGedEditor;   // manager of this frame
   TClass         *fModelClass;  // class corresponding to instantiated GedFrame
   Bool_t          fAvoidSignal; // flag for executing slots

   TList          *fExtraTabs;   // addtional tabs in ged editor
   Int_t           fPriority;    // location in GedEditor

   virtual void MakeTitle(const char *title);

public:
   TGedFrame(const TGWindow *p = 0,
             Int_t width = 140, Int_t height = 30,
             UInt_t options = kChildFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedFrame();

   virtual void      Update();

   virtual Option_t *GetDrawOption() const;

   TClass*           GetModelClass()              { return fModelClass;  }
   Int_t             GetPriority()                { return fPriority;    }
   TList*            GetExtraTabs()               { return fExtraTabs;   }
   TGedEditor*       GetGedEditor()               { return fGedEditor;   }
   virtual void      AddExtraTab(TGedSubFrame* sf);
   virtual TGVerticalFrame* CreateEditorTabSubFrame(const char* name);

   virtual void      Refresh(TObject *model);
   virtual void      SetDrawOption(Option_t *option="");
   virtual Bool_t    AcceptModel(TObject*) { return kTRUE; }
   void              SetModelClass(TClass* mcl)   { fModelClass = mcl; }
   virtual void      SetModel(TObject* obj) = 0;
   virtual void      SetGedEditor(TGedEditor* ed) { fGedEditor = ed; }
   virtual void      ActivateBaseClassEditors(TClass* cl);

   ClassDef(TGedFrame, 0); //base editor's frame
};

class TGedNameFrame : public TGedFrame {
private:
   TGedNameFrame(const TGedNameFrame&);            // not implemented
   TGedNameFrame& operator=(const TGedNameFrame&); // not implemented

protected:
   TGLabel          *fLabel;      //label of attribute frame
   TGCompositeFrame *f1, *f2;     //container frames
   TGToolTip        *fTip;        //tool tip associated with button

public:
   TGedNameFrame(const TGWindow *p =0 ,
                 Int_t width = 170, Int_t height = 30,
                 UInt_t options = kChildFrame,
                 Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedNameFrame();

   virtual Bool_t   HandleButton(Event_t *event);
   virtual Bool_t   HandleCrossing(Event_t *event);

   virtual void     SetModel(TObject* obj);

   ClassDef(TGedNameFrame,0)      //frame showing the selected object name
};

#endif

