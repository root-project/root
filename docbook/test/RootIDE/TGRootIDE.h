// @(#)root/test/RootIDE/:$Id$
// Author: Bertrand Bellenot   20/04/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGROOTIDE
#define ROOT_TGROOTIDE


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGRootIDE                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGTextEdit
#include "TGTextEdit.h"
#endif

class TGToolBar;
class TTimer;
class TGStatusBar;
class TGLayoutHints;
class TGMenuBar;
class TGPopupMenu;
class TString;
class TMacro;
class TGText;
class TGTextEntry;
class TGComboBox;
class TGTextBuffer;
class TGTab;
class TGFileContainer;
class TGLVEntry;
class TObjArray;
class TGTabElement;
class TGVerticalFrame;
class TGHorizontalFrame;
class TGPictureButton;
class THtml;
class TGLabel;
class TProcessID;
class TGHtml;

class TGDocument : public TNamed {

protected:
   Int_t              fTabId;       // Tab id associated to this document
   Bool_t             fModified;    // modified flag
   TGTextEdit        *fEditor;      // pointer on TGTextEdit widget
   TGTab             *fTab;         // pointer to main TGTab widget
   TGTabElement      *fTabEl;       // pointer to associated TGTabElement
   TObjArray         *fDocList;     // pointer to list of opened documents

public:
   TGDocument(const char *fname = "", const char *title = "", Int_t tabid = 0,
             TGTab *tab = 0, TGTabElement *tabel = 0, TGTextEdit *edit = 0,
             TObjArray *doclist = 0);
   virtual ~TGDocument() { }

   Bool_t         Open(const char *filename);
   Bool_t         Close();
   Bool_t         Save(const char *filename = "");
   Int_t          GetTabId() const { return fTabId; }
   Bool_t         IsModified() const { return fModified; }
   void           SetTabId(Int_t id) { fTabId = id; }
   void           Modified(Bool_t mod = kTRUE) { fModified = mod; }
   TGTextEdit    *GetTextEdit() const { return fEditor; }
   TGTab         *GetTab() const { return fTab; }
   TGTabElement  *GetTabEl() const { return fTabEl; }
   void           DataChanged();
   void           DataDropped(char *fname);

   ClassDef(TGDocument,0)  // Simple class describing document used in TGRootIDE
};

class TGRootIDE : public TGMainFrame {

protected:
   Int_t              fPid;               // current process id
   TTimer            *fTimer;             // for statusbar and toolbar update
   TGStatusBar       *fStatusBar;         // for file name, line and col number
   TGToolBar         *fToolBar;           // toolbar with common tool buttons
   TGTab             *fTab;               // Main tab
   TGTextEntry       *fDir;               // directory text entry
   TGTextBuffer      *fDirBuf;            // directory text buffer
   TGComboBox        *fDirCombo;          // directory history combobox
   TGFileContainer   *fContents;          // file navigation container
   TGTextEdit        *fTextEdit;          // active text edit widget
   TGTextView        *fTextView;          // command output widget
   TGComboBox        *fComboCmd;          // commands combobox
   TGComboBox        *fFileType;          // file type combobox
   TGTextEntry       *fFilter;            // file type text entry widget
   TGLabel           *fLabel;             // "command :" label
   TGTextEntry       *fCommand;           // command text entry widget
   TGTextBuffer      *fCommandBuf;        // command text buffer
   TGLayoutHints     *fMenuBarLayout;     // used for the menubar
   TGLayoutHints     *fMenuBarItemLayout; // used for for menubar items
   TGMenuBar         *fMenuBar;           // editor's menu bar
   TGPopupMenu       *fMenuFile;          // "File" menu entry
   TGPopupMenu       *fMenuEdit;          // "Edit" menu entry
   TGPopupMenu       *fMenuSearch;        // "Search" menu entry
   TGPopupMenu       *fMenuTools;         // "Tools" menu entry
   TGPopupMenu       *fMenuHelp;          // "Help" menu entry
   Bool_t             fExiting;           // true if editor is closing
   Bool_t             fTextChanged;       // true if text has changed
   TString            fFilename;          // name of the opened file
   TMacro            *fMacro;             // pointer on the opened macro
   TObjArray         *fDocList;           // list of open documents
   TGDocument        *fCurrentDoc;        // pointer on current document
   Int_t              fCurrent;           // index of current document
   Int_t              fNbDoc;             // number of documents

   TGVerticalFrame   *fVerticalFrame;     // vertical frame
   TGHorizontalFrame *fHorizontalFrame;   // horizontal frame
   TGPictureButton   *fBack;              // back button
   TGPictureButton   *fForward;           // forward button
   TGPictureButton   *fReload;            // reload button
   TGPictureButton   *fStop;              // stop loading button
   TGPictureButton   *fHome;              // home button
   TGComboBox        *fComboBox;          // url history combo box
   TGTextBuffer      *fURLBuf;            // url buffer
   TGTextEntry       *fURL;               // url text entry
   THtml             *fHtml;              // html doc
   TGHtml            *fGuiHtml;           // html widget

   virtual void       Build();

public:
   TGRootIDE(const char *filename = 0, const TGWindow *p = 0,
                UInt_t w = 900, UInt_t h = 600);
   TGRootIDE(TMacro *macro, const TGWindow *p = 0, UInt_t w = 0,
                UInt_t h = 0);
   virtual ~TGRootIDE();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   CloseWindow();

   void           ClearText();
   Bool_t         LoadBuffer(const char *buf) { return fTextEdit->LoadBuffer(buf); }
   void           LoadFile(char *fname = NULL);
   void           SaveFile(const char *fname);
   Bool_t         SaveFileAs();
   void           PrintText();
   void           Search(Bool_t ret);
   void           Goto();
   void           About();
   Int_t          IsSaved();
   void           CompileMacro();
   void           ExecuteMacro();
   void           InterruptMacro();
   void           SetText(TGText *text) { fTextEdit->SetText(text); }
   void           AddText(TGText *text) { fTextEdit->AddText(text); }
   void           AddLine(const char *string) { fTextEdit->AddLine(string); }
   void           AddLineFast(const char *string) { fTextEdit->AddLineFast(string); }
   TGText        *GetText() const { return fTextEdit->GetText(); }

   void           DisplayFile(const TString &fname);
   void           DisplayDirectory(const TString &fname);
   void           DisplayObject(const TString& fname,const TString& name);
   void           OnDoubleClick(TGLVEntry* f, Int_t btn);
   void           CloseTab(Int_t id);
   void           DoTab(Int_t id);
   void           ApplyFilter(Int_t id);
   
   void           CheckRemote(const char *str);
   void           DirSelected(const char *txt);
   void           DirChanged();
   void           Selected(const char *txt);
   void           URLChanged();
   void           Back();
   void           Forward();
   void           Reload();
   void           Stop();
   void           MouseOver(char *);
   void           MouseDown(char *);

   ClassDef(TGRootIDE,0)  // Simple IDE using TGTextEdit and TGHtml widgets
};

#endif
