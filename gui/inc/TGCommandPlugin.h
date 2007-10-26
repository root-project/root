// @(#)root/gui:$Id$
// Author: Bertrand Bellenot   26/09/2007

#ifndef ROOT_TGCommandPlugin
#define ROOT_TGCommandPlugin

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLabel;
class TGComboBox;
class TGTextEntry;
class TGTextBuffer;
class TGTextView;
class TTimer;

class TGCommandPlugin : public TGMainFrame {

protected:
   Int_t              fPid;               // current process id
   TGHorizontalFrame *fHf;                // horizontal frame
   TGLabel           *fLabel;             // "command :" label
   TGComboBox        *fComboCmd;          // commands combobox
   TGTextEntry       *fCommand;           // command text entry widget
   TGTextBuffer      *fCommandBuf;        // command text buffer
   TGTextView        *fStatus;            // output capture view
   TTimer            *fTimer;             // for local/remote update

public:

   TGCommandPlugin(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~TGCommandPlugin();

   void           CheckRemote(const char * /*str*/);
   void           HandleCommand();

   virtual Bool_t HandleTimer(TTimer *t);

   ClassDef(TGCommandPlugin, 0) // Command (I/O redirection) plugin for the new ROOT Browser
};

#endif
