// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   01/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBrowser.h"
#include "TWin32TreeViewCtrl.h"
#include "TWin32BrowserImp.h"
#include "TWin32Application.h"

#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

#ifndef ROOT_TApplication
#include "TApplication.h"
#endif


// ClassImp(TWin32TreeViewCtrl)

//______________________________________________________________________________
TWin32TreeViewCtrl::TWin32TreeViewCtrl(TGWin32WindowsObject *winobj,const char *title,
                                       Float_t x,Float_t y, Float_t width, Float_t  height,
                                       const char *type,UInt_t style)
                                      : TWin32CommCtrl(winobj,title,x,y,width,height,type,style)
{
    fNumOfChildren = 0;
//*-*  Set the permananet attributes for Tree View
    fTVItem.mask   = TVIF_TEXT | TVIF_PARAM | TVIF_CHILDREN |
                     TVIF_IMAGE  | TVIF_SELECTEDIMAGE  ;

    fTVItem.cChildren = 1;

//    TreeView_SetImageList(fhwndWindow, GetWin32ApplicationImp()->GetSmallIconList(), TVSIL_NORMAL);
    TreeView_SetImageList(fhwndWindow, ((TWin32BrowserImp *)fMasterWindow)->GetSmallIconList(), TVSIL_NORMAL);

    fhLastOpenedItem = fhExpandedItem = fhLastCreated = TVI_ROOT;
    AddFirst();
}

//______________________________________________________________________________
TWin32TreeViewCtrl::~TWin32TreeViewCtrl()
{
}

//______________________________________________________________________________
void TWin32TreeViewCtrl::Add(TObject *obj, const char *name)
{
 TV_ITEM  tvi;
 TV_INSERTSTRUCT tvins;

// Set the text of the item.
 fTVItem.pszText = name ? (char *) name : (char *)(obj->GetName());
 fTVItem.cchTextMax = lstrlen(fTVItem.pszText);

// Assume the item is not a parent item, so give it a
// document image.
 fTVItem.iImage = kClosedFolderIcon;
 fTVItem.iSelectedImage = kClosedFolderIcon;

// Save the pointer the ROOT Object together with TreeView Item
 fTVItem.lParam = (LPARAM) obj;

 tvins.item         = fTVItem;
 tvins.hParent      = fhExpandedItem;
 tvins.hInsertAfter = TVI_LAST;


// Add the item to the tree-view control.
// printf(" AddToList %s after %x \n", name, fhSelectedItem);
 fhLastCreated = TreeView_InsertItem(fhwndWindow, &tvins);
 fNumOfChildren++;

 if (!fhLastCreated) {
         Int_t err = GetLastError();
         Printf(" Last Error was %d \n", err);
         Error("AddToList","Can't add the new item");
 }
}

//______________________________________________________________________________
void TWin32TreeViewCtrl::AddFirst()
{
 TV_ITEM  tvi;
 TV_INSERTSTRUCT tvins;

// Set the text of the item.
 fTVItem.pszText = (char *)(gROOT->GetName());
 fTVItem.cchTextMax = lstrlen(fTVItem.pszText);

// Assume the item is not a parent item, so give it a
// document image.

 fTVItem.iImage = kMainROOTIcon;
 fTVItem.iSelectedImage = kMainROOTIcon;

// Save the pointer the ROOT Object together with TreeView Item
 fTVItem.lParam = (LPARAM) gROOT;

 tvins.item         = fTVItem;
 tvins.hParent      = fhExpandedItem;
 tvins.hInsertAfter = TVI_FIRST;


// Add the item to the tree-view control.
 fhLastCreated = TreeView_InsertItem(fhwndWindow, &tvins);
 TreeView_SelectItem(fhwndWindow,fhLastCreated);
 TreeView_Expand(fhwndWindow,fhLastCreated,TVE_EXPAND);

 if (!fhLastCreated) {
         Int_t err = GetLastError();
         Printf(" Last Error was %d \n", err);
         Error("AddToList","Can't add the new item");
 }
}

//______________________________________________________________________________
void TWin32TreeViewCtrl::CreateAccessories()
{
           // Create some extra things depends of the type of the control.
}

#ifdef uuu
//______________________________________________________________________________
void TWin32TreeViewCtrl::ExecThreadCB(TWin32SendClass *command)
{
        // This class doesn't use this entry point
}
#endif

//______________________________________________________________________________
void TWin32TreeViewCtrl::ExecThreadCB(TWin32SendClass *command)
{
   UINT code     =   (UINT)  (command->GetData(0));
   LPNM_TREEVIEW Tree = (LPNM_TREEVIEW)(command->GetData(1));
   delete command;

   switch (code) {
     case TVN_ITEMEXPANDED:
                  OnItemExpanded(Tree);
          break;
     case TVN_ITEMEXPANDING:
                  OnItemExpandingCB(Tree);
          break;
     case TVN_SELCHANGED:
                  OnSelectChanged(Tree);
                  break;
     case TVN_SELCHANGING:
                  OnSelectChanging(Tree);
                  break;
     case TVN_DELETEITEM:
                  OnDeleteItem(Tree);
                   break;
     default:
                 break;
    }
   if (Tree) free(Tree);
   return ;
}

//______________________________________________________________________________
UInt_t TWin32TreeViewCtrl::GetItemObject(RECT *prc)
{
 //*-*  returns the selected TObject item and its position or ZERO;
        TV_ITEM tvi;
//              TV_HITTESTINFO tvihit;
//              tvihit.pt.x = prc->right;
//              tvihit.pt.y = prc->bottom;
//              tvihit.flags = TVHT_ONITEM | VHT_ONITEMBUTTON ;
        if (tvi.hItem = TreeView_GetSelection(fhwndWindow))
        {
                tvi.mask = TVIF_PARAM;
                if (TreeView_GetItem(fhwndWindow, &tvi))
                        if (TreeView_GetItemRect(fhwndWindow, tvi.hItem,prc,FALSE))
                                return tvi.lParam;
        }
        return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32TreeViewCtrl::OnNotify(LPARAM lParam)
{
// CallBack function to manage the notify messages
// We have to create our own copy of this message;

 UINT code     =  (UInt_t)(((LPNMHDR) lParam)->code);
 NM_TREEVIEW *Tree = (NM_TREEVIEW *)lParam;

 switch (code) {
     case NM_RCLICK:
                 return -1; // means mouse event has been detected
     case TVN_ITEMEXPANDED:
                 return OnItemExpanded(Tree);
     case TVN_ITEMEXPANDING:
                  return OnItemExpanding(Tree);
     case TVN_SELCHANGED:
                  return OnSelectChanged(Tree);
     case TVN_SELCHANGING:
                  return OnSelectChanging(Tree);
     case TVN_DELETEITEM:
                  return OnDeleteItem(Tree);
     default:
                 return 0;
    }
 return 0;
}

//______________________________________________________________________________
Int_t TWin32TreeViewCtrl::OnDeleteItem(LPNM_TREEVIEW lParam)
{
  // HTREEITEM item = lParam->itemNew.hItem;
  // printf(" TWin32TreeViewCtrl::OnSelectChanged \n");
  // TreeView_Expand(fhwndWindow,item,TVE_EXPAND);
        return 0;
}

//______________________________________________________________________________
Int_t TWin32TreeViewCtrl::OnItemExpanded(LPNM_TREEVIEW lParam)
{
//*-* Check whether this item has any kids and re-set its status accordanly

        return 0;
#if 0
        if (TreeView_GetChild(fhwndWindow,lParam->itemNew.hItem)) return 0;
#endif
}
//______________________________________________________________________________
Int_t TWin32TreeViewCtrl::OnSelectChanged(LPNM_TREEVIEW lParam)
{
  // HTREEITEM item = lParam->itemNew.hItem;
  // printf(" TWin32TreeViewCtrl::OnSelectChanged \n");
  // TreeView_Expand(fhwndWindow,item,TVE_EXPAND);
        return 0;
}

//______________________________________________________________________________
Int_t TWin32TreeViewCtrl::OnSelectChanging(LPNM_TREEVIEW lParam)
{
  // HTREEITEM item = lParam->itemNew.hItem;
  // printf(" TWin32TreeViewCtrl::OnSelectChanged \n");
  // TreeView_Expand(fhwndWindow,item,TVE_EXPAND);
        return 0;
}

//______________________________________________________________________________
Int_t TWin32TreeViewCtrl::OnItemExpanding(LPNM_TREEVIEW lParam)
{
  HTREEITEM item = lParam->itemNew.hItem;
  UINT     state = lParam->itemNew.state;
  UINT     mask  = lParam->itemNew.stateMask;
  TObject  *obj  = (TObject *)(lParam->itemNew.lParam);


//*-* action member indicates whether the list is to expand or collapse
  UINT    action = lParam->action;
//*-*  TVE_COLLAPSE        Collapses the list.
//*-*  TVE_COLLAPSERESET   Collapses the list and removes the child items. Note that TVE_COLLAPSE must also be specified.
//*-*  TVE_EXPAND          Expands the list.
//*-*  TVE_TOGGLE          Collapses the list if it is currently expanded or expands it if it is currently collapsed.

  if (action == TVE_COLLAPSE)
  {

      ((TWin32BrowserImp *)fMasterWindow)->SetViewFlag(kTreeOnly);
      TreeView_Expand(fhwndWindow,item,TVE_COLLAPSE | TVE_COLLAPSERESET );
      return 0;
  }
  else if (action == TVE_EXPAND)
  {
        ((TWin32BrowserImp *)fMasterWindow)->SetViewFlag(kTreeOnly);
         OnItemExpandingCB(lParam);
         return 0;

         NM_TREEVIEW *Tree = (NM_TREEVIEW *)memcpy(malloc(sizeof(NM_TREEVIEW)),
                                                   (void *)lParam,sizeof(NM_TREEVIEW));
         TWin32SendClass *CodeOp = new TWin32SendClass(this,(UInt_t)(((LPNMHDR) lParam)->code),
                                                                (UInt_t)Tree,0,0,0);
         ExecCommandThread(CodeOp);
         return 0; //Returns TRUE to prevent the list from expanding or collapsing.
  }
  return 0;
}

//______________________________________________________________________________
void TWin32TreeViewCtrl::OnItemExpandingCB(LPNM_TREEVIEW lParam)
{
  HTREEITEM item = lParam->itemNew.hItem;
  UINT     state = lParam->itemNew.state;
  UINT     mask  = lParam->itemNew.stateMask;
  TObject  *obj  = (TObject *)(lParam->itemNew.lParam);

//*-* action member indicates whether the list is to expand or collapse
  UINT    action = lParam->action;
//*-*  TVE_COLLAPSE        Collapses the list.
//*-*  TVE_COLLAPSERESET   Collapses the list and removes the child items. Note that TVE_COLLAPSE must also be specified.
//*-*  TVE_EXPAND          Expands the list.
//*-*  TVE_TOGGLE          Collapses the list if it is currently expanded or expands it if it is currently collapsed.

  if (action == TVE_EXPAND)
  {
      fhExpandedItem = item;
//      ((TWin32BrowserImp *)fMasterWindow)->SetSelectedItem(item);
//*-*  Check whether this expanded item is selected
      Bool_t iOpenFlag = kTRUE;
      if (!(state & TVIS_SELECTED))
      {
//*-*  List view would be erased and change as well
          ((TWin32BrowserImp *)fMasterWindow)->SetListBlocked(kTRUE);
          iOpenFlag = kFALSE;
      }
      ((TWin32BrowserImp *)fMasterWindow)->Shift();

      SetCursor(LoadCursor(NULL,IDC_WAIT));

      TBrowser *b = ((TWin32BrowserImp *)fMasterWindow)->Browser();
      if (b) {
         obj->Browse(b);
         b->SetSelected(obj);
      }
      ((TWin32BrowserImp *)fMasterWindow)->SetListBlocked(kFALSE);

      lParam->itemNew.mask = 0;
//*-* Change the "shape" of the folder  of this item
      if (iOpenFlag)
      {
          if (fhLastOpenedItem != TVI_ROOT && fhLastOpenedItem != item
              && lParam->itemNew.lParam != (LPARAM) gROOT ) CloseItem();

          fhLastOpenedItem = item;

          if (item != TVI_ROOT)
          {
              lParam->itemNew.mask = TVIF_IMAGE | TVIF_SELECTEDIMAGE ;
              if (lParam->itemNew.lParam != (LPARAM) gROOT)
              {
                lParam->itemNew.iSelectedImage = kOpenedFolderIcon;
                lParam->itemNew.iImage         = kOpenedFolderIcon;
              }
              else
              {
                lParam->itemNew.iSelectedImage = kMainROOTIcon;
                lParam->itemNew.iImage         = kMainROOTIcon;
              }
          }
      }
//*-*  Check whether this item born any children or it is the root item
      if (fNumOfChildren == 0)
      {
//*-* Destroy the "parents" status of this item"
        lParam->itemNew.mask |= TVIF_CHILDREN;
        lParam->itemNew.stateMask = 0;
        lParam->itemNew.cChildren = 0;
      }

      if (iOpenFlag || fNumOfChildren == 0)
                  TreeView_SetItem(fhwndWindow,&(lParam->itemNew));
      fNumOfChildren = 0;
  }
  else
          Error(" OnItemExpandingCB", "Wrong notify");
}

//______________________________________________________________________________
void TWin32TreeViewCtrl::OpenItem(HTREEITEM item)
{
  if (fhLastOpenedItem != TVI_ROOT && fhLastOpenedItem != item) CloseItem();

//*-*  "Open" the current folder
   TV_ITEM tvi;
   fhLastOpenedItem = item;
   tvi.hItem = item;

   tvi.mask = TVIF_IMAGE | TVIF_SELECTEDIMAGE ;
   tvi.iSelectedImage = kOpenedFolderIcon;
   tvi.iImage         = kOpenedFolderIcon;

   TreeView_SetItem(fhwndWindow,&tvi);
}

//______________________________________________________________________________
void TWin32TreeViewCtrl::CloseItem()
{
//*-*
//*-* TWin32TreeViewCtrl::CloseItem  closes the current opened item
//*-*  and prepared the current item structure to be present it with "opened folder
//*-*

//*-* Close the "old" folder
    TV_ITEM tvi;
    tvi.mask   = TVIF_IMAGE | TVIF_SELECTEDIMAGE ;
    tvi.iSelectedImage = tvi.iImage = kClosedFolderIcon;
    tvi.hItem  = fhLastOpenedItem;
    TreeView_SetItem(fhwndWindow,&tvi);
//      fhLastOpenedItem = TVI_ROOT;

}

//______________________________________________________________________________
void TWin32TreeViewCtrl::Shift(UInt_t lParam)
{
  if (lParam==0 ) return;
  if (!((TObject *)lParam)->IsFolder()) return;
//*-* Scan children of the last opened to find the next opened item ?
  HTREEITEM item = TreeView_GetChild(fhwndWindow,fhLastOpenedItem);
  if (item && (UInt_t) item != 0xfeeefeee)  // I didn't find what does 0xfeeefee mean
  {
      Bool_t ret;
      TV_ITEM tvi;
      tvi.mask = TVIF_PARAM ;
      do
      {
         tvi.hItem = item;
             ret = TreeView_GetItem(fhwndWindow,&tvi);
             item= TreeView_GetNextSibling(fhwndWindow,item);
      }
      while( tvi.lParam != lParam && ret && item);

      if (ret) fhExpandedItem = tvi.hItem;
  }
  else
      fhLastOpenedItem = fhExpandedItem = TVI_ROOT;
}
