#ifndef CANVSAVE_H
#define CANVSAVE_H

//-------------------------------------------------------
//
//   A small panel destined to provide graphical interface
//   for "Canvas Save" action
//
//   Author: Dmitry Vasiliev (LNS, Catania)
//
//-------------------------------------------------------

#include <TGFrame.h>
#include <TGLabel.h>
#include <TGComboBox.h>
#include <TGTextEntry.h>
#include <TGTextBuffer.h>
#include <TGButton.h>

class CanvSave : public TGTransientFrame {

private:
   TGCompositeFrame         *fFrame, *fF1, *fF2, *fF3;
   TGComboBox               *fCombo;
   TGTextEntry              *fText;
   TGTextBuffer             *fBuf;
   TGButton                 *fSave, *fCancel;

public:
   CanvSave(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
            UInt_t options = kMainFrame | kVerticalFrame);
   virtual ~CanvSave();
   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(CanvSave,0)
};

#endif
