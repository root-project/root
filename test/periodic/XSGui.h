/*
 * $Header$
 * $Log$
 */

#ifndef __XSGUI_H
#define __XSGUI_H

#include <TVirtualX.h>
#include <TGMenu.h>
#include <TGFrame.h>
#include <TGLabel.h>
#include <TGButton.h>
#include <TGCanvas.h>
#include <TGClient.h>
#include <TGMsgBox.h>
#include <TGStatusBar.h>
#include <TGTextEntry.h>
#include <TGFileDialog.h>

#define PRGNAME   "XSGui"
#define VERSION "1.0a"
#define AUTHOR   "V.Vlachoudis"
#define DATE   "Jun-1999"
#define EMAIL   "V.Vlachoudis@cern.ch"

#define ABOUT   PRGNAME" " VERSION"\n" \
      AUTHOR" " DATE \
      EMAIL

/* ---- Define all the available commands ----- */
enum TCommandIdentifiers {
   M_FILE_OPEN,
   M_FILE_SAVE,
   M_FILE_SAVEAS,
   M_FILE_CLOSE,
   M_FILE_PRINT,
   M_FILE_PRINT_SETUP,
   M_FILE_EXIT,

   M_ELEM_REACTION,
   M_ELEM_MODIFY,
   M_ELEM_CLEAR,

   M_OPTION_ZOOM,
   M_OPTION_SETUP,

   M_HELP_ABOUT,
};

/* --- Define the Help for each menu item --- */
// WARNING!!! MUST BE IN ALIGN WITH TCommandsIdentifiers
#ifdef __XSGUI_CXX
static const char   *menuTip[] = {
   "Open a file",
   "Save drawing",
   "Save drawing as",
   "Close current drawing",
   "Print drawing",
   "Printer Setup",
   "Exit from program",

   "Select a Reaction",
   "Edit/Modify/Delete the current graphs",
   "Clear Current Graph",

   "Zoom current image",
   "General graphic options",

   "About the program"
};
#endif

/* ================== XSGui ===================== */
class XSGui : public TGMainFrame
{
private:
   TGMenuBar      *menuBar;

   TGPopupMenu      *fileMenu,
            *elemMenu,
            *optMenu,
            *helpMenu;

   TGLayoutHints      *menuBarLayout,
            *menuBarItemLayout,
            *menuBarHelpLayout;

   TGStatusBar      *statusBar;

public:
   XSGui(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~XSGui();

   virtual void   CloseWindow();
   virtual Bool_t   ProcessMessage(Longptr_t msg, Longptr_t param, Longptr_t);

      Bool_t   ProcessMenuMessage( Longptr_t param );

   //ClassDef(XSGui,1)
}; // XSGui

#endif
