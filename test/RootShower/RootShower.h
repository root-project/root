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


extern TGListTree       *gEventListTree; // event selection TGListTree
extern TGListTreeItem   *gBaseLTI;
extern TGListTreeItem   *gTmpLTI;
extern TGListTreeItem   *gLTI[];

extern Int_t            gColIndex;

class RootShower: public TGMainFrame {

    friend class SettingsDialog;

private:

    // Statics
    static Int_t        fgDefaultXPosition; // default X position of top left corner
    static Int_t        fgDefaultYPosition; // default Y position of top left corner

    Bool_t              fOk;
    Bool_t              fModified;
    Bool_t              fSettingsModified;
    Bool_t              fIsRunning;
    Bool_t              fInterrupted;
    Bool_t              fShowProcess;
    Bool_t              fCreateGIFs;

    ULong_t             fEventNr;   // Event number
    UInt_t              fNRun;      // Run number
    TDatime             fEventTime; // Event generation date

    Int_t               fPicIndex;
    Int_t               fPicNumber;
    Int_t               fPicDelay;
    Int_t               fPicReset;

    TEnv               *fRootShowerEnv;
    // MenuBar Frame
    TGMenuBar          *fMenuBar;
    TGPopupMenu        *fMenuFile;
    TGPopupMenu        *fMenuTest;
    TGPopupMenu        *fMenuInspect;
    TGPopupMenu        *fMenuView;
    TGPopupMenu        *fMenuHelp;
    TGLayoutHints      *fMenuBarLayout;
    TGLayoutHints      *fMenuBarItemLayout;
    TGLayoutHints      *fMenuBarHelpLayout;
    void                MakeMenuBarFrame();
    void                CloseMenuBarFrame();

    // ToolBar Frame
    TGToolBar          *fToolBar;
    void ShowToolBar(Bool_t show = kTRUE);

    // Layout hints
    TGLayoutHints      *fL1;
    TGLayoutHints      *fL2;
    TGLayoutHints      *fL3;
    TGLayoutHints      *fL4;
    TGLayoutHints      *fL5;
    TGLayoutHints      *fL6;
    TGLayoutHints      *fL7;
    TGLayoutHints      *fL8;

    // Title Frame
    GTitleFrame        *fTitleFrame;

    // Main Frame
    TGCompositeFrame   *fMainFrame;

    // Selection frame
    TGCompositeFrame   *fSelectionFrame;
    GButtonFrame       *fButtonFrame; // button frame
    TGListTreeItem     *AddToTree(const Text_t *name = 0);
    void                BuildEventTree();
    TGCanvas           *fTreeView; // why do we need this?
    TGListTree         *fEventListTree; // event selection TGListTree
    TGListTreeItem     *fCurListItem; // current TGlistTreeItem (level) in TGListTree

    // Display frame
    TGTab              *fDisplayFrame; // TGTab for graphical and text display
    TRootEmbeddedCanvas *fEmbeddedCanvas; // the actual frame which displays event
    TRootEmbeddedCanvas *fEmbeddedCanvas2; // the actual frame which displays event
    TRootEmbeddedCanvas *fEmbeddedCanvas3; // the actual frame which displays histo
    TGTextEdit         *fTextView;

    // Zooming stuff...
    TGHorizontalFrame  *fHFrame,*fHFrame2;
    TGLayoutHints      *fZoomButtonsLayout;
    TGButton           *fZoomPlusButton,*fZoomMoinsButton;
    TGButton           *fZoomPlusButton2,*fZoomMoinsButton2;

    // Statusbar
    TGStatusBar        *fStatusBar;        // status bar reporting event info

    TTimer             *fTimer;
    TCanvas            *cA;
    TCanvas            *cB;
    TCanvas            *cC;
    
    MyEvent            *fEvent;
    TPad               *padC;
    
    TH1F               *fHisto_dEdX;       // histogram of particle's energy loss

protected:

    Int_t               fFirstParticle;
    Double_t            fE0;
    Double_t            fB;

public:
    // statics
    static void        setDefaultPosition(Int_t x, Int_t y);

    // Constructors & destructor
    RootShower(const TGWindow *p, UInt_t w, UInt_t h);
    virtual ~RootShower();

    void               SetOk(Bool_t ok=true) { fOk = ok; }
    void               Modified(Bool_t modified=true) { fModified = modified; }
    void               SettingsModified(Bool_t modified=true) { fSettingsModified = modified; }
    void               Interrupt(Bool_t inter=true) { fInterrupted = inter; }
    Bool_t             IsInterrupted() { return fInterrupted; }
    virtual void       Initialize(Int_t first);
    virtual void       OnOpenFile(const Char_t *filename);
    virtual void       OnSaveFile(const Char_t *filename);
    virtual void       OnShowerProduce();
    virtual void       produce();
    virtual void       ShowInfos();
    virtual void       HighLight(TGListTreeItem *item);
    virtual void       OnShowSelected(TGListTreeItem *item);
    virtual void       Layout();
    virtual void       CloseWindow();
    virtual Bool_t     HandleConfigureNotify(Event_t *event);
    virtual Bool_t     ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
    virtual Bool_t     HandleTimer(TTimer *);
    virtual Int_t      DistancetoPrimitive(Int_t px, Int_t py);

};

extern RootShower   *gRootShower;

#endif // EMSHOWER_H
