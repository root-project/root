// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   01/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWin32TreeViewCtrl
#define ROOT_TWin32TreeViewCtrl

////////////////////////////////////////////////////////
//                                                    //
// TWin32Control - class to use WIN32 common controls //
//                 for TGWin32WindowsObject objects   //
//                                                    //
//                                                    //
//                                                    //
//                                                    //
//                                                    //
////////////////////////////////////////////////////////

#include "TWin32CommCtrl.h"
#include "TWin32HookViaThread.h"

class TWin32TreeViewCtrl :  public TWin32CommCtrl {
private:
   TV_ITEM         fTVItem;               // TreeView definition of the present object
   Int_t           fNumOfChildren;        // The number of the children items
   HTREEITEM       fhExpandedItem;        // Handle of the last expanded item
   HTREEITEM       fhLastCreated;         // Handle of the last created item
   HTREEITEM       fhLastOpenedItem;      // Handle of the last "opened" item
   HIMAGELIST      fhImageList;           // Image list of the icon
   void            CloseItem();           // Close the current open folder
   void            AddFirst();            // Add the first tree view item

protected:
   virtual Int_t OnDeleteItem     (LPNM_TREEVIEW lParam);
   virtual Int_t OnItemExpanded   (LPNM_TREEVIEW lParam);
   virtual Int_t OnItemExpanding  (LPNM_TREEVIEW lParam);
   virtual void OnItemExpandingCB(LPNM_TREEVIEW lParam); // CB to call from the command thread
   virtual Int_t OnSelectChanged  (LPNM_TREEVIEW lParam);
   virtual Int_t OnSelectChanging (LPNM_TREEVIEW lParam);
   void ExecThreadCB(TWin32SendClass *code);

public:
   TWin32TreeViewCtrl(){fhwndWindow=0;} //default ctor
   TWin32TreeViewCtrl(TGWin32WindowsObject *winobj,const Text_t *title, Float_t x=0,Float_t y=0, Float_t width=1, Float_t  height=1,
                          const Text_t *type=WC_TREEVIEW,
                          UInt_t style=(DWORD) WS_VISIBLE |  TVS_HASLINES | TVS_HASBUTTONS | TVS_LINESATROOT );
  ~TWin32TreeViewCtrl();
  virtual void Add(TObject *obj, const char *name);
  virtual void CreateAccessories();                    // Create some extra things depends of the type of the control.
  virtual UInt_t           GetItemObject(RECT *rpc);   // returns the selected TObject item (lParam) and its position
  virtual HTREEITEM        GetCreatedItem() {return fhLastCreated;}
  virtual HTREEITEM        GetExpandedItem(){return fhExpandedItem;}
  virtual HTREEITEM        GetOpenedItem()  {return fhLastOpenedItem;}
  virtual LRESULT APIENTRY OnNotify(LPARAM lParam);    // CallBack function to manage the notify messages
  virtual void   OpenItem(HTREEITEM item);             // Open the new folder and close the old one if any
  virtual void   Shift(UInt_t lParam=0);               // Shift items to add a next one

  // ClassDef(TWin32TreeViewCtrl,0)  // TreeView Class is used to build TBrowserImp for Win32

};

#endif
