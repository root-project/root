// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This File contains the declaration of the RootShower-class           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOTSHOWER_H
#define ROOTSHOWER_H

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TDatime
#include "TDatime.h"
#endif
#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif

class TGMenuBar;
class TGPopupMenu;
class GTitleFrame;
class GButtonFrame;
class TGButton;
class TGListTree;
class TGListTreeItem;
class TRootEmbeddedCanvas;
class TGCanvas;
class TGStatusBar;
class TGTextEdit;
class TGTab;
class TCanvas;
class TPad;
class MyEvent;
class TEnv;
class TTimer;
class TH1F;
class TGToolBar;
class TContextMenu;


extern TGListTree       *gEventListTree;    // Event selection TGListTree
extern TGListTreeItem   *gBaseLTI;          // First ListTree item
extern TGListTreeItem   *gTmpLTI;           // Temporary ListTree item
extern TGListTreeItem   *gLTI[];            // Array of ListTree items (particles)

extern Int_t            gColIndex;          // Global gradient color table used 
                                            // for tracks color

class RootShower: public TGMainFrame {

friend class SettingsDialog;

private:
   // Statics
   static Int_t         fgDefaultXPosition; // default X position of top left corner
   static Int_t         fgDefaultYPosition; // default Y position of top left corner

   Bool_t               fOk;                // Return code from settings dialog
   Bool_t               fModified;          // kTRUE if setting mods not saved
   Bool_t               fSettingsModified;  // kTRUE if settings have been modified
   Bool_t               fIsRunning;         // Simulation running flag
   Bool_t               fInterrupted;       // Interrupts current simulation
   Bool_t               fShowProcess;       // Display process details
   Bool_t               fCreateGIFs;        // GIFs creation of current event

   ULong_t              fEventNr;           // Event number
   UInt_t               fNRun;              // Run number
   TDatime              fEventTime;         // Event generation date

   Int_t                fPicIndex;          // Index of animation images
   Int_t                 fPicNumber;         // Number of images used for animation
   Int_t                fPicDelay;          // Delay between animation images
   Int_t                fPicReset;          // kTRUE to display first anim picture

   TEnv                *fRootShowerEnv;     // RootShower environment variables
   // MenuBar Frame
   TGMenuBar           *fMenuBar;           // Main menu bar
   TGPopupMenu         *fMenuFile;          // "File" popup menu
   TGPopupMenu         *fMenuEvent;         // "Event" popup menu
   TGPopupMenu         *fMenuTools;         // "Tools" popup menu
   TGPopupMenu         *fMenuView;          // "View" popup menu
   TGPopupMenu         *fMenuHelp;          // "Help" popup menu
   TGLayoutHints       *fMenuBarLayout;
   TGLayoutHints       *fMenuBarItemLayout;
   TGLayoutHints       *fMenuBarHelpLayout;
   void                 MakeMenuBarFrame();
   void                 CloseMenuBarFrame();

   // ToolBar Frame
   TGToolBar           *fToolBar;
   void ShowToolBar(Bool_t show = kTRUE);

   // Layout hints
   TGLayoutHints       *fL1;
   TGLayoutHints       *fL2;
   TGLayoutHints       *fL3;
   TGLayoutHints       *fL4;
   TGLayoutHints       *fL5;
   TGLayoutHints       *fL6;
   TGLayoutHints       *fL7;
   TGLayoutHints       *fL8;

   // Title Frame
   GTitleFrame         *fTitleFrame;        // Title frame

   // Main Frame
   TGCompositeFrame    *fMainFrame;         // Main frame

   // Selection frame
   TGCompositeFrame    *fSelectionFrame;    // Frame containing list tree and button frame
   GButtonFrame        *fButtonFrame;       // Frame containing control buttons
   TGListTreeItem      *AddToTree(const char *name = 0);
   void                 BuildEventTree();
   TGCanvas            *fTreeView;          // Canvas containing event selection list tree
   TGListTree          *fEventListTree;     // Event selection TGListTree
   TGListTreeItem      *fCurListItem;       // Current TGlistTreeItem (level) in TGListTree
   TContextMenu        *fContextMenu;       // pointer to context menu

   // Display frame
   TGTab               *fDisplayFrame;      // TGTab for graphical and text display
   TRootEmbeddedCanvas *fEmbeddedCanvas;    // Events frame
   TRootEmbeddedCanvas *fEmbeddedCanvas2;   // Selected event frame
   TRootEmbeddedCanvas *fEmbeddedCanvas3;   // Statistics frame
   TGTextEdit          *fTextView;          // PDG infos frame

   // Zooming stuff...
   TGHorizontalFrame   *fHFrame,*fHFrame2;  // Frame containing zoom buttons
   TGLayoutHints       *fZoomButtonsLayout; // Layout of zoom buttons
   TGButton            *fZoomPlusButton,*fZoomMoinsButton;  // Zoom buttons
   TGButton            *fZoomPlusButton2,*fZoomMoinsButton2;// Zoom buttons

   // Statusbar
   TGStatusBar         *fStatusBar;         // Status bar reporting event info

   TTimer              *fTimer;             // Timer used for animation
   TCanvas             *fCA;                // Events view
   TCanvas             *fCB;                // Selected event view
   TCanvas             *fCC;                // Statistics
    
   MyEvent             *fEvent;             // Pointer on actual event
   TPad                *fPadC;              // TPad of statistics histo
    
   TH1F                *fHisto_dEdX;        // histogram of particle's energy loss

protected:
   Int_t                fFirstParticle;     // Primary particle type
   Double_t             fE0;                // Initial particle energy
   Double_t             fB;                 // Magnetic field

public:
   // statics
   static void         setDefaultPosition(Int_t x, Int_t y);

   // Constructors & destructor
   RootShower(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~RootShower();

   void                SetOk(Bool_t ok=true) { fOk = ok; }
   void                Modified(Bool_t modified=true) { fModified = modified; }
   void                SettingsModified(Bool_t modified=true) { fSettingsModified = modified; }
   void                Interrupt(Bool_t inter=true) { fInterrupted = inter; }
   Bool_t              IsInterrupted() { return fInterrupted; }
   virtual void        Initialize(Int_t first);
   virtual void        OnOpenFile(const Char_t *filename);
   virtual void        OnSaveFile(const Char_t *filename);
   virtual void        OnShowerProduce();
   virtual void        Produce();
   virtual void        ShowInfos();
   virtual void        HighLight(TGListTreeItem *item);
   virtual void        OnShowSelected(TGListTreeItem *item);
   virtual void        Layout();
   virtual void        CloseWindow();
   virtual Bool_t      HandleConfigureNotify(Event_t *event);
   virtual Bool_t      HandleKey(Event_t *event);
   virtual Bool_t      ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual Bool_t      HandleTimer(TTimer *);
   virtual Int_t       DistancetoPrimitive(Int_t px, Int_t py);
   void                Clicked(TGListTreeItem *item, Int_t x, Int_t y);
   void                UpdateDisplay() { fCA->Modified(); fCA->Update(); }
};

extern RootShower   *gRootShower;

#endif // EMSHOWER_H
