// @(#)root/test/rhtml/:$Id$
// Author: Bertrand Bellenot   09/05/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGHtmlBrowser
#define ROOT_TGHtmlBrowser

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGHtmlBrowser                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGFrame.h"

class TGMenuBar;
class TGPopupMenu;
class TGStatusBar;
class TGVerticalFrame;
class TGHorizontalFrame;
class TGComboBox;
class TGTextBuffer;
class TGTextEntry;
class TGPictureButton;
class TGHtml;

class TGHtmlBrowser : public TGMainFrame {

protected:

   TGMenuBar         *fMenuBar;           // menu bar
   TGPopupMenu       *fMenuFile;          // "File" menu entry
   TGPopupMenu       *fMenuFavorites;     // "Favorites" menu entry
   TGPopupMenu       *fMenuTools;         // "Tools" menu entry
   TGPopupMenu       *fMenuHelp;          // "Help" menu entry
   TGStatusBar       *fStatusBar;         // status bar
   TGVerticalFrame   *fVerticalFrame;     // main vertical frame
   TGHorizontalFrame *fHorizontalFrame;   // main horizontal frame
   TGPictureButton   *fBack;              // "Back" picture button
   TGPictureButton   *fForward;           // "Forward" picture button
   TGPictureButton   *fReload;            // "Reload Page" picture button
   TGPictureButton   *fStop;              // "Stop Loading" picture button
   TGPictureButton   *fHome;              // "Home" picture button
   TGComboBox        *fComboBox;          // combo box for URLs history
   TGTextBuffer      *fURLBuf;            // text buffer for current URL text entry
   TGTextEntry       *fURL;               // current URL text entry
   TGHtml            *fHtml;              // main TGHtml widget
   Int_t              fNbFavorites;       // number of favorites in the menu

public:
   TGHtmlBrowser(const char *filename = 0, const TGWindow *p = 0,
                 UInt_t w = 900, UInt_t h = 600);
   virtual ~TGHtmlBrowser() { ; }

   virtual void      CloseWindow();
   virtual Bool_t    ProcessMessage(Long_t msg, Long_t parm1, Long_t);
   void              Selected(const char *txt);
   void              URLChanged();
   void              Back();
   void              Forward();
   void              Reload();
   void              Stop();
   void              MouseOver(char *);
   void              MouseDown(char *);

   ClassDef(TGHtmlBrowser, 0) // very simple html browser
};

#endif

