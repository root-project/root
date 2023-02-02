/*
 * $Header$
 * $Log$
 */

#ifndef __XSSTEPBUTTON_H
#define __XSSTEPBUTTON_H

#include <TGFrame.h>
#include <TGButton.h>

/* --- Messages that generate ---- */

#define XSSTEPBUTTON_UP      0
#define XSSTEPBUTTON_DOWN   1

/* =========== XSStepButton ============== */
class XSStepButton : public TGCompositeFrame
{
protected:
   const TGWindow   *fMsgWindow;   // window handling listbox messages

   Int_t      buttonId;

   TGLayoutHints   *lHints;

   TGPictureButton   *upButton,
         *downButton;

   UInt_t      width,
         height;

public:
   XSStepButton( const TGWindow *p, Int_t id );
   ~XSStepButton() override;

   virtual void   Associate(const TGWindow *w) { fMsgWindow = w; }

   Bool_t   ProcessMessage(Longptr_t msg,
            Longptr_t param1, Longptr_t param2) override;

   TGDimension   GetDefaultSize() const override
         { return TGDimension(width,height); }


   //ClassDefOverride(XSStepButton,1)
}; // XSStepButton

#endif
