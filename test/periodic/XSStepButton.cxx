/*
 * $Header$
 * $Log$
 */

#include "XSStepButton.h"

//ClassImp(XSStepButton)

/* ----- XSStepButton ----- */
XSStepButton::XSStepButton( const TGWindow *p, Int_t id )
   : TGCompositeFrame(p, 10, 10, kFixedSize)
{
   buttonId = id;
   fMsgWindow = NULL;

   lHints = new TGLayoutHints(kLHintsTop | kLHintsLeft);
   upButton = new TGPictureButton(this,
         fClient->GetPicture("arrow_up.xpm"),XSSTEPBUTTON_UP);
   upButton->Associate(this);
   downButton = new TGPictureButton(this,
         fClient->GetPicture("arrow_down.xpm"),XSSTEPBUTTON_DOWN);
   downButton->Associate(this);
   AddFrame(upButton, lHints);
   AddFrame(downButton, lHints);

   width = upButton->GetDefaultWidth() + 2*fBorderWidth;
   height = 2*upButton->GetDefaultHeight() + 2*fBorderWidth;

   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
} // XSStepButton

/* ----- ~XSStepButton ----- */
XSStepButton::~XSStepButton( )
{
   delete   lHints;
   delete   upButton;
   delete   downButton;
} // ~XSStepButton

/* ----- ProcessMessage ----- */
Bool_t
XSStepButton::ProcessMessage(Long_t msg, Long_t param1, Long_t /* param2 */)
{
   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               // Send a message
               if (fMsgWindow)
                  SendMessage(
                     fMsgWindow,
                     MK_MSG(kC_COMMAND,
                        kCM_BUTTON),
                     buttonId,
                     param1);
               break;
            default:
               break;
         }
      default:
         break;
   }
   return kTRUE;
} // ProcessMessage
