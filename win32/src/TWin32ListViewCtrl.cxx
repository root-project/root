// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   04/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TBrowser.h"
#include "TWin32ListViewCtrl.h"
#include "TWin32BrowserImp.h"

#ifndef ROOT_Buttons
#include "Buttons.h"
#endif

#ifndef ROOT_TWin32Application
#include "TWin32Application.h"
#endif

#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

#ifndef ROOT_TApplication
#include "TApplication.h"
#endif

#ifndef ROOT_TLink
#include "TLink.h"
#endif

#ifndef ROOT_TClass
#include "TClass.h"
#endif

// ClassImp(TWin32ListViewCtrl)

//______________________________________________________________________________
TWin32ListViewCtrl::TWin32ListViewCtrl(TGWin32WindowsObject *winobj,const char *title, Float_t x,Float_t y, Float_t width, Float_t  height,
                          const char *type,UInt_t style) :
                          TWin32CommCtrl(winobj,title,x,y,width,height,type,style )
{
  ResetIndex();

//*-* Initialize LV_ITEM members that are common to all items.

  fLVItem.mask =  LVIF_TEXT | LVIF_PARAM | LVIF_STATE ;

  if (((TWin32BrowserImp *)fMasterWindow)->GetNormalIconList())
  {
         ListView_SetImageList(fhwndWindow,((TWin32BrowserImp *)fMasterWindow)->GetNormalIconList(),LVSIL_NORMAL);
     fLVItem.mask |= LVIF_IMAGE ;
  }
  if (((TWin32BrowserImp *)fMasterWindow)->GetSmallIconList())
  {
         ListView_SetImageList(fhwndWindow,((TWin32BrowserImp *)fMasterWindow)->GetSmallIconList(),LVSIL_SMALL);
     fLVItem.mask |= LVIF_IMAGE ;
  }

}

//______________________________________________________________________________
void TWin32ListViewCtrl::Add(TObject *obj, char const *name)
{
//*-*
//*-*  List View Item States
//*-*  ----------------------
//*-*  An item's state determines its appearance and functionality.
//*-*  The state can be zero, or one or more of the following values:

//*-*     LVIS_CUT          The item is marked for a cut and paste operation.
//*-*     LVIS_DROPHILITED  The item is highlighted as a drag-and-drop target.
//*-*     LVIS_FOCUSED      The item has the focus, so it is surrounded by a standard
//*-*                       focus rectangle. Although more than one item may be
//*-*                       selected, only one item can have the focus.
//*-*     LVIS_SELECTED     The item is selected. The appearance of a selected item
//*-*                       depends on whether it has the focus
//*-*

 fLVItem.state = 0;
 fLVItem.stateMask = 0;
 fLVItem.pszText = name ? (char *)name: (char *)(obj->GetName());
 fLVItem.cchTextMax = lstrlen(fLVItem.pszText);

 if (obj->IsFolder())
     fLVItem.iImage = kClosedFolderIcon;                   // image list index
 else
     fLVItem.iImage = kDocumentIcon;

 fLVItem.lParam = (LPARAM) obj;

//*-*  Specifies the one-based index of the subitem to which this structure refers,
//*-*  or zero if this structure refers to an item rather than a subitem.
 fLVItem.iSubItem = 0;

//*-* Specifies the zero-based index of the item to which this structure refers.
 fLVItem.iItem = ++fiLastCreated;

 //*-*  First Clean the list
  if (!fLVItem.iItem) ListView_DeleteAllItems(fhwndWindow);

//*-* Add the item
  Add();
//  printf(" add %s %d as %d \n", name,fiLastCreated, i );

}

//______________________________________________________________________________
void TWin32ListViewCtrl::Add(char const **row, Int_t cols, TObject *obj)
{
//*-*  TWin32InspectView::Add inserts one item to the listview
//*-*
//*-*  row  - one row to be isnerted in the this list view (charater array)
//*-*  cols - the number of the columns cvame with row (the length of the row, in fact)
//*-*
//*-*  List View Item States
//*-*  ----------------------
//*-*  An item's state determines its appearance and functionality.
//*-*  The state can be zero, or one or more of the following values:

//*-*     LVIS_CUT          The item is marked for a cut and paste operation.
//*-*     LVIS_DROPHILITED  The item is highlighted as a drag-and-drop target.
//*-*     LVIS_FOCUSED      The item has the focus, so it is surrounded by a standard
//*-*                       focus rectangle. Although more than one item may be
//*-*                       selected, only one item can have the focus.
//*-*     LVIS_SELECTED     The item is selected. The appearance of a selected item
//*-*                       depends on whether it has the focus
//*-*


 fLVItem.state = 0;
 fLVItem.stateMask = 0;

 if (obj) SetObject(obj);

 if (obj && obj->IsFolder())
     fLVItem.iImage = kClosedFolderIcon;                   // image list index
 else
     fLVItem.iImage = kDocumentIcon;

 fLVItem.lParam = (LPARAM) fObject;

//*-* Specifies the zero-based index of the item to which this structure refers.
 fLVItem.iItem = ++fiLastCreated;

//*-*  First Clean the list
 if (!fLVItem.iItem) ListView_DeleteAllItems(fhwndWindow);


 Int_t i;
 // ListView_SetItemCount(fhwndWindow,cols);
 for (i=0;i<cols;i++)
 {
    //*-*  Specifies the one-based index of the subitem to which this structure refers,
    //*-*  or zero if this structure refers to an item rather than a subitem.
    fLVItem.iSubItem = i;
    fLVItem.pszText = row[i] ? (char *)row[i]: " ";

//*-* Add the sub item
    if (i)
        ListView_SetItemText(fhwndWindow,fLVItem.iItem, i,row[i] ? (char *)row[i]: " ")
    else
    {
        fLVItem.pszText = row[i] ? (char *)row[i]: " ";
        fLVItem.iSubItem = 0;
        Add();
    }


//  printf(" add %s %d as %d \n", name,fiLastCreated, i );
 }

}

//______________________________________________________________________________
void TWin32ListViewCtrl::Add(char const *cell, Int_t row, Int_t col)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*  Inserts one element into the row,col cell.
//*-*
//*-*  cell - the null-terminated text to be set
//*-*  row - specifies the zero-based index of the item to which this structure refers.
//*-*
//*-*  col- Specifies the one-based index of the subitem to which this structure refers,
//*-*       or zero if this structure refers to an item rather than a subitem.
//*-*
//*-*  NB. It changes/sets the text attributes if the existen object only.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 ListView_SetItemText(fhwndWindow, row, col,cell ? (char *)cell : " ");

//  printf(" add %s %d as %d \n", name,fiLastCreated, i );
}


//______________________________________________________________________________
void TWin32ListViewCtrl::Add()
{
//*-* Add the item
 Int_t j = ListView_InsertItem(fhwndWindow, &fLVItem);

}

//______________________________________________________________________________
void TWin32ListViewCtrl::AddColumn(Int_t index)
{
//*-* Add the new column in the list view
//*-*  index = -1 means the column index is equal the subitem index
 Int_t indx = index;
 if (indx == -1) indx = fLVColumn.iSubItem;
 Int_t j = ListView_InsertColumn(fhwndWindow,indx,&fLVColumn);
 j = ListView_SetColumnWidth(fhwndWindow,indx,LVSCW_AUTOSIZE);
}

//______________________________________________________________________________
void TWin32ListViewCtrl::ClearList()
{
   // Delete all items from the list
  ListView_DeleteAllItems(fhwndWindow);
  ResetIndex();
}

//______________________________________________________________________________
Int_t TWin32ListViewCtrl::GetFormat(EListFmt fmt)
{
    const Int_t formats[] = {LVCFMT_LEFT, LVCFMT_LEFT, LVCFMT_CENTER, LVCFMT_RIGHT };
    return formats[fmt];
}

//______________________________________________________________________________
void TWin32ListViewCtrl::AddColumns(char const **headers, Int_t cols, Int_t *width, EListFmt *form)
{
// creates a header for all columns
    if (cols <= 0) return;

    Int_t i;
    for (i = 0; i < cols;i++)
        AddColumn(headers ? headers[i]:0, i+1, width ? width[i]:0, form ? form[i]:EListFmt(0));
}
//______________________________________________________________________________
void TWin32ListViewCtrl::AddColumn(char const *header, Int_t col, Int_t width, EListFmt form)
{
// set the header of the single column

    fLVColumn.mask = LVCF_SUBITEM;

    if (form)
    {
       fLVColumn.mask |=  LVCF_FMT;
       fLVColumn.fmt = GetFormat(form);
    }

    if (width)
    {
        fLVColumn.mask |=  LVCF_WIDTH;
        fLVColumn.cx = width;
    }

    if (header)
    {
        fLVColumn.mask |=  LVCF_TEXT;
        fLVColumn.pszText = (char *)header;
        fLVColumn.cchTextMax = lstrlen(fLVColumn.pszText);
    }

    fLVColumn.iSubItem = col;

    AddColumn();
}

//______________________________________________________________________________
void TWin32ListViewCtrl::CreateAccessories()
{
           // Create some extra things depends of the type of the control.
}

#ifdef uuuu
//______________________________________________________________________________
void TWin32ListViewCtrl::ExecThreadCB(TWin32SendClass *command)
{
   UINT code     =   (UINT)  (command->GetData(0));
   NM_LISTVIEW *List = (NM_LISTVIEW *)(command->GetData(1));
   delete (TWin32SendClass *)command;

   switch (code) {
     case LVN_ITEMCHANGED:
                  OnItemChanged(List);
                  break;
     case LVN_ITEMCHANGING:
                  OnItemChanging(List);
                  break;
     case LVN_DELETEITEM:
                  OnDeleteItem(List);
     case LVN_DELETEALLITEMS:
                  OnDeleteAllItems(List);
                   break;
     default:
                 break;
    }
   if (List) free(List);
   return ;

}
#endif

//______________________________________________________________________________
UInt_t TWin32ListViewCtrl::GetItemObject(RECT *prc)
{
 //*-*  returns the selected TObject item and its position or ZERO;
        LV_ITEM lvi;
        lvi.iItem =  ListView_GetNextItem(fhwndWindow,-1,LVNI_FOCUSED);
        if (lvi.iItem >= 0)
        {
                lvi.mask = LVIF_PARAM;
                if (ListView_GetItem(fhwndWindow,&lvi))
                        if (ListView_GetItemRect(fhwndWindow, lvi.iItem,prc,LVIR_SELECTBOUNDS))
                                return lvi.lParam;
        }
        return 0;
}

//______________________________________________________________________________
LRESULT APIENTRY TWin32ListViewCtrl::OnNotify(LPARAM lParam)
{
 // CallBack function to manage the notify messages
UINT code     =  ((LPNMHDR) lParam)->code;
NM_LISTVIEW *List = (NM_LISTVIEW *)lParam;

switch (code) {

      case NM_DBLCLK: {
//            printf(" NM_D_BLCLK\n");
            RECT prc;
            SetObject((TObject *)GetItemObject(&prc));
            if (fObject) ExecCommandThread();
            return 0;
          }
      case NM_RCLICK:
//            printf(" NM_RCLICK\n");
            return -1; // means mouse event has been detected
            break;
     case LVN_KEYDOWN:
                 {
                   DWORD wVKey = ((LV_KEYDOWN *)lParam)->wVKey;
                 }
     case NM_CLICK:
//          printf(" NM_CLICK\n");
                 break;
     case LVN_ITEMCHANGED:
                  OnItemChanged(List);
                  break;
     case LVN_ITEMCHANGING:
                  OnItemChanging(List);
                  break;
     case LVN_DELETEITEM:
                  OnDeleteItem(List);
                                  break;
     case LVN_DELETEALLITEMS:
                  OnDeleteAllItems(List);
                  break;
     default:
                 break;
    }

 return 0;
}

//______________________________________________________________________________
void TWin32ListViewCtrl::OnItemChanged(NM_LISTVIEW *lParam)
{
  int item      = lParam->iItem;
  int subItem   = lParam->iSubItem;
  UINT newstate = lParam->uNewState;
  UINT oldstate = lParam->uOldState;
  UINT changed  = lParam->uChanged;
  SetObject(0);
  TObject  *obj  = (TObject *)(lParam->lParam);
  if (!obj) return;  // The clicked item has no linked object to show

//*-*  LVIS_SELECTED     The item is selected
//*-*  LVIS_DROPHILITED  The item is highlighted as a drag-and-drop target.
//*-*  LVIS_FOCUSED      The item has the focus. Only one item can have the focus
//*-*  LVIS_CUT          The item is marked for a cut and paste operation.
  if (changed & LVIF_STATE && newstate & LVIS_SELECTED)
  {
      ResetIndex();
//*-*  Expand the  TreevView ctrl level
          if (obj->IsFolder())
          {
//*-*  Shift TreevView ctrl level
             Int_t view = ((TWin32BrowserImp *)fMasterWindow)->GetViewFlag() & kTreeOnly;
             TWin32TreeViewCtrl *tree =
                   (TWin32TreeViewCtrl *)(((TWin32BrowserImp *)fMasterWindow)->GetCommCtrl(kID_TREEVIEW));
             if (tree && view)
             {
                 ((TWin32BrowserImp *)fMasterWindow)->Shift((UInt_t)obj);
                 HTREEITEM htvi = tree->GetExpandedItem();
                 if (htvi != TVI_ROOT)
                 {
                      HWND hwnd = tree->GetWindow();
                      TreeView_Expand(hwnd,htvi,TVE_COLLAPSE | TVE_COLLAPSERESET );
                      TreeView_SelectItem(hwnd,htvi);
                      TreeView_Expand(hwnd,htvi,TVE_EXPAND);
                      return;
                 }
                 ((TWin32BrowserImp *)fMasterWindow)->ClearViewFlag(kTreeOnly);
              }
           }
      SetCursor(LoadCursor(NULL,IDC_WAIT));
      SetObject(obj);       // Save the pointer to call its method later.
  }
}
//______________________________________________________________________________
void TWin32ListViewCtrl::ExecThreadCB(TWin32SendClass *command)
{
 TObject *obj =  fObject;
 if (obj) {
    TClass *cl = obj->IsA();
    // For the TCollection the TIreator can be employed to show next TObject
    if (cl->InheritsFrom(TLink::Class()))
       obj->ExecuteEvent(kButton1Up,0,0);
    else
      obj->Browse(((TWin32BrowserImp *)fMasterWindow)->Browser());
 }
}

//______________________________________________________________________________
void TWin32ListViewCtrl::SetItemCount(Int_t count)
{
//*-* The SetItemCount prepares a list view control for adding a large number of items.
//*-*
//*-*  count - number of the expected items
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    if (count)
        ListView_SetItemCount(fhwndWindow,count);
}

//______________________________________________________________________________
void TWin32ListViewCtrl::SetObject(TObject *obj)
{
  fObject = obj;
  if (fObject) {
    TBrowser *b = ((TWin32BrowserImp *)fMasterWindow)->Browser();
    if (b) b->SetSelected(fObject);
  }
}
