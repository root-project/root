/*
 * $Header$
 * $Log$
 */

#ifndef __XSELEMENT_DLG_H
#define __XSELEMENT_DLG_H

#include <TGTab.h>
#include <TGFrame.h>
#include <TGButton.h>
#include <TGLayout.h>

#include "XSElementList.h"
#include "XSPeriodicTable.h"

/* =========== XSElementDlg ============== */
class XSElementDlg : public TGTransientFrame
{
private:
   UInt_t         *selectedZ;

   XSPeriodicTable      *pTable;
   TGTab         *tabMenu;
   TGButton      *okButton,
            *closeButton;
   TGCompositeFrame   *buttonFrame,
            *nameFrame,
            *mnemonicFrame,
            *zFrame;
   XSElementList      *nameListBox,
            *mnemonicListBox,
            *zListBox;
   TGLayoutHints      *buttonLayoutHints,
            *frameLayoutHints,
            *lHints,
            *lHints2;

public:
   XSElementDlg(const TGWindow *p, const TGWindow *main,
         UInt_t *retZ, UInt_t w=600, UInt_t h=350);
   ~XSElementDlg() override;

   void   CloseWindow() override;
   virtual Bool_t   ProcessButton(Longptr_t param);
   Bool_t   ProcessMessage(Longptr_t msg,
            Longptr_t param1, Longptr_t param2) override;

   //ClassDefOverride(XSElementDlg,1)
}; // XSElementDlg

#endif
