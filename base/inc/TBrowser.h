// @(#)root/base:$Name:  $:$Id: TBrowser.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Fons Rademakers   25/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TBrowser
#define ROOT_TBrowser

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBrowser                                                             //
//                                                                      //
// Using a TBrowser on can browse all ROOT objects. It shows in a list  //
// on the left side of the window all browsable ROOT classes. Selecting //
// one of the classes displays, in the iconbox on the right side, all   //
// objects in the class. Selecting one of the objects in the iconbox,   //
// will place all browsable objects in a new list and draws the         //
// contents of the selected class in the iconbox. And so on....         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TBrowserImp
#include "TBrowserImp.h"
#endif


class TContextMenu;
class TBrowserTimer;


class TBrowser : public TNamed {

private:
   TBrowserImp   *fImp;                //!Window system specific browser implementation
   TBrowserTimer *fTimer;              //Browser's timer
   TObject       *fLastSelectedObject; //The last TObject selected by user

protected:
   TContextMenu  *fContextMenu;    //Context menu pointer
   Bool_t         fNeedRefresh;    //True if the browser needs refresh

public:
   TBrowser(const char *name="Browser", const char *title="ROOT Object Browser");
   TBrowser(const char *name, const char *title, UInt_t width, UInt_t height);
   TBrowser(const char *name, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   TBrowser(const char *name, TObject *obj, const char *title="ROOT Object Browser");
   TBrowser(const char *name, TObject *obj, const char *title, UInt_t width, UInt_t height);
   TBrowser(const char *name, TObject *obj, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TBrowser();

   void          Add(TObject *obj, const char *name = 0);
   void          Create(TObject *obj = 0);      // Create this Browser
   void          ExecuteDefaultAction(TObject *obj);
   TBrowserImp  *GetBrowserImp() const         { return fImp; }
   TContextMenu *GetContextMenu()              { return fContextMenu;}
   Bool_t        GetRefreshFlag()              { return fNeedRefresh; }
   TObject      *GetSelected();
   void          SetRefreshFlag(Bool_t flag)   { fNeedRefresh = flag; }
   void          Iconify()                     { fImp->Iconify(); }
   virtual void  RecursiveRemove(TObject *obj);
   void          Refresh();
   void          SetSelected(TObject *clickedObject);
   void          Show()                        { fImp->Show(); }

   ClassDef(TBrowser,0)  //ROOT Object Browser
};

#endif
