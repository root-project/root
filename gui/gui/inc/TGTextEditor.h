// @(#)root/gui:$Id$
// Author: Bertrand Bellenot   20/06/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextEditor
#define ROOT_TGTextEditor


#include "TGFrame.h"
#include "TGTextEdit.h"

class TGComboBox;
class TGLabel;
class TGLayoutHints;
class TGMenuBar;
class TGPopupMenu;
class TGStatusBar;
class TGText;
class TGTextBuffer;
class TGTextEntry;
class TGToolBar;
class TMacro;
class TString;
class TTimer;

class TGTextEditor : public TGMainFrame {

protected:

   TTimer           *fTimer;              ///< for statusbar and toolbar update
   TGStatusBar      *fStatusBar;          ///< for file name, line and col number
   TGToolBar        *fToolBar;            ///< toolbar with common tool buttons
   TGTextEdit       *fTextEdit;           ///< text edit widget
   TGLabel          *fLabel;              ///< "command" label
   TGComboBox       *fComboCmd;           ///< commands combobox
   TGTextEntry      *fCommand;            ///< command text entry widget
   TGTextBuffer     *fCommandBuf;         ///< command text buffer
   TGLayoutHints    *fMenuBarLayout;      ///< used for the menubar
   TGLayoutHints    *fMenuBarItemLayout;  ///< used for for menubar items
   TGMenuBar        *fMenuBar;            ///< editor's menu bar
   TGPopupMenu      *fMenuFile;           ///< "File" menu entry
   TGPopupMenu      *fMenuEdit;           ///< "Edit" menu entry
   TGPopupMenu      *fMenuSearch;         ///< "Search" menu entry
   TGPopupMenu      *fMenuTools;          ///< "Tools" menu entry
   TGPopupMenu      *fMenuHelp;           ///< "Help" menu entry
   Bool_t            fExiting;            ///< true if editor is closing
   Bool_t            fTextChanged;        ///< true if text has changed
   TString           fFilename;           ///< name of the opened file
   TMacro           *fMacro;              ///< pointer on the opened macro
   virtual void      Build();

public:
   TGTextEditor(const char *filename = 0, const TGWindow *p = 0,
                UInt_t w = 900, UInt_t h = 600);
   TGTextEditor(TMacro *macro, const TGWindow *p = 0, UInt_t w = 0,
                UInt_t h = 0);
   virtual ~TGTextEditor();

   void           ClearText();
   Bool_t         LoadBuffer(const char *buf) { return fTextEdit->LoadBuffer(buf); }
   void           LoadFile(const char *fname = 0);
   void           SaveFile(const char *fname);
   Bool_t         SaveFileAs();
   void           PrintText();
   void           Search(Bool_t ret);
   void           Goto();
   void           About();
   void           DataChanged();
   void           DataDropped(char *fname);
   Int_t          IsSaved();
   void           CompileMacro();
   void           ExecuteMacro();
   void           InterruptMacro();
   void           SetText(TGText *text) { fTextEdit->SetText(text); }
   void           AddText(TGText *text) { fTextEdit->AddText(text); }
   void           AddLine(const char *string) { fTextEdit->AddLine(string); }
   void           AddLineFast(const char *string) { fTextEdit->AddLineFast(string); }
   TGText        *GetText() const { return fTextEdit->GetText(); }

   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   CloseWindow();
   virtual void   DeleteWindow();

   ClassDef(TGTextEditor,0)  // Simple text editor using TGTextEdit widget
};

#endif
