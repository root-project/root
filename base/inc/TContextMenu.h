// @(#)root/base:$Name:  $:$Id: TContextMenu.h,v 1.2 2000/11/21 16:06:09 brun Exp $
// Author: Nenad Buncic   08/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TContextMenu
#define ROOT_TContextMenu


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TContextMenu                                                               //
//                                                                            //
// This class provides an interface to  context sensitive popup menus.        //
// These menus pop up when the user hits  the right mouse button,  and        //
// are destroyed when the menu pops downs.                                    //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TContextMenuImp
#include "TContextMenuImp.h"
#endif

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TMethod;
class TMethodArg;
class TVirtualPad;
class TObjArray;
class TBrowser;
class TToggle;


class TContextMenu : public TNamed {

friend class  TContextMenuImp;

protected:
   TContextMenuImp *fContextMenuImp;      //!Context menu system specific implementation
   TMethod         *fSelectedMethod;      //selected method
   TObject         *fSelectedObject;      //selected object
   TVirtualPad     *fSelectedCanvas;      //selected canvas (if exist)
   TVirtualPad     *fSelectedPad;         //selected pad (if exist)
   TBrowser        *fBrowser;             //selected browser (if exist)

   virtual void DisplayPopUp(Int_t x, Int_t y)
      { if (fContextMenuImp) fContextMenuImp->DisplayPopup(x, y); }

private:
   TContextMenu();

public:
   TContextMenu(const char *name, const char *title = "Context sensitive popup menu");
   virtual ~TContextMenu();

   virtual void Action(TObject *object, TMethod *method);
   virtual void Action(TObject *object, TToggle *toggle);
   void Action(TMethod *method) { Action(fSelectedObject, method); }
   void Action(TToggle *toggle) { Action(fSelectedObject, toggle); }
   virtual Char_t *CreateArgumentTitle(TMethodArg *argument);
   virtual Char_t *CreateDialogTitle(TObject *object, TMethod *method);
   virtual Char_t *CreatePopupTitle(TObject *object );
   virtual void Execute(const char *method,  const char *params) { TObject::Execute(method, params); }
   virtual void Execute(TMethod *method, TObjArray *params) { TObject::Execute(method, params); }
   virtual void Execute(TObject *object, TMethod *method, const Char_t *params);
   virtual void Execute(TObject *object, TMethod *method, TObjArray *params);
   void Execute(const Char_t *params) { Execute(fSelectedObject, fSelectedMethod, params); }
   void Execute(TObjArray *params) { Execute(fSelectedObject, fSelectedMethod, params); }
   virtual TBrowser *GetBrowser() { return fBrowser; }
   virtual TContextMenuImp *GetContextMenuImp() { return fContextMenuImp; }
   virtual TVirtualPad *GetSelectedCanvas() { return fSelectedCanvas; }
   virtual TMethod *GetSelectedMethod() { return fSelectedMethod; }
   virtual TObject *GetSelectedObject() { return fSelectedObject; }
   virtual TVirtualPad *GetSelectedPad() { return fSelectedPad; }
   virtual void Popup(Int_t x, Int_t y, TObject *obj, TVirtualPad *c=0, TVirtualPad *p=0); // Create menu from canvas
   virtual void Popup(Int_t x, Int_t y, TObject *obj, TBrowser *b);  // Create menu from Browser
   virtual void SetCanvas(TVirtualPad *c) { fSelectedCanvas = c; }
   virtual void SetBrowser(TBrowser *b) { fBrowser = b; }
   virtual void SetMethod(TMethod *m) { fSelectedMethod = m; }
   virtual void SetNameTitle(const char *name, const char *title) { TNamed::SetNameTitle(name, title); }
   virtual void SetObject(TObject *o) { fSelectedObject = o; }
   virtual void SetPad(TVirtualPad *p) { fSelectedPad = p; }

   ClassDef(TContextMenu,0)  //Context sensitive popup menu
};

#endif
