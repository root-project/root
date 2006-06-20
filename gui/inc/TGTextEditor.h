// @(#)root/gui:$Name:$:$Id:$
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

/**************************************************************************

    This file is part of notepad, a simple text editor.
    Copyright (C) 1997-2001, Harald Radke, Hector Peraza.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextEditor                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGToolBar;
class TTimer;
class TGStatusBar;
class TGTextEdit;
class TGLayoutHints;
class TGMenuBar;
class TGPopupMenu;
class TString;
class TMacro;

class TGTextEditor : public TGMainFrame {

protected:

   TTimer           *fTimer;              // for statusbar and toolbar update
   TGStatusBar      *fStatusBar;          // for file name, line and col number
   TGToolBar        *fToolBar;            // toolbar with common tool buttons
   TGTextEdit       *fTextEdit;           // text edit widget
   TGComboBox       *fComboCmd;           // commands combobox
   TGTextEntry      *fCommand;            // command text entry widget
   TGTextBuffer     *fCommandBuf;         // command text buffer
   TGLayoutHints    *fMenuBarLayout;      // used for the menubar
   TGLayoutHints    *fMenuBarItemLayout;  // used for for menubar items
   TGMenuBar        *fMenuBar;            // editor's menu bar
   TGPopupMenu      *fMenuFile;           // "File" menu entry
   TGPopupMenu      *fMenuEdit;           // "Edit" menu entry
   TGPopupMenu      *fMenuSearch;         // "Search" menu entry
   TGPopupMenu      *fMenuTools;          // "Tools" menu entry
   TGPopupMenu      *fMenuHelp;           // "Help" menu entry
   Bool_t            fExiting;            // true if editor is closing
   Bool_t            fTextChanged;        // true if text has changed
   TString           fFilename;           // name of the opened file
   TMacro           *fMacro;              // pointer on the opened macro
   virtual void      Build();

public:
   TGTextEditor(const char *filename = 0, const TGWindow *p = 0, 
                UInt_t w = 900, UInt_t h = 600);
   TGTextEditor(TMacro *macro, const TGWindow *p = 0, UInt_t w = 0, 
                UInt_t h = 0);
   virtual ~TGTextEditor();

   void           ClearText();
   void           LoadFile(char *fname = NULL);
   void           SaveFile(const char *fname);
   Bool_t         SaveFileAs();
   void           PrintText();
   void           Search(Bool_t ret);
   void           Goto();
   void           About();
   void           DataChanged() { fTextChanged = kTRUE; }
   Int_t          IsSaved();
   void           CompileMacro();
   void           ExecuteMacro();
   void           InterruptMacro();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   CloseWindow();

   ClassDef(TGTextEditor,0)  // Simple text editor using TGTextEdit widget
};

#endif
