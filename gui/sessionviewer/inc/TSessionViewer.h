// @(#)root/sessionviewer:$Id$
// Author: Marek Biskup, Jakub Madejczyk, Bertrand Bellenot 10/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TSessionViewer
#define ROOT_TSessionViewer

#include "TGFrame.h"

#include "TString.h"

#include "TGTextEntry.h"

#include "TGNumberEntry.h"

#include "TGTab.h"

#include "TGListView.h"

#include "TTime.h"

#include <stdio.h>
#include <time.h>

class TList;
class TChain;
class TDSet;
class TGNumberEntry;
class TGTextEntry;
class TGTextButton;
class TGCheckButton;
class TGTextBuffer;
class TGTableLayout;
class TGIcon;
class TGLabel;
class TGHProgressBar;
class TGPopupMenu;
class TGLVContainer;
class TGListView;
class TGLVEntry;
class TGCanvas;
class TGListTree;
class TGListTreeItem;
class TGStatusBar;
class TGPicture;
class TGMenuBar;
class TGPopupMenu;
class TGToolBar;
class TGTextView;
class TGTab;
class TRootEmbeddedCanvas;
class TGListBox;
class TCanvas;
class TEnv;
struct UserGroup_t;

class TProofMgr;
class TProof;
class TSessionViewer;
class TSessionLogView;
class TQueryResult;
class TContextMenu;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionViewer - A GUI for ROOT / PROOF Sessions                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// TQueryDescription class : Description of queries
//////////////////////////////////////////////////////////////////////////

class TQueryDescription : public TObject {

public:
   enum ESessionQueryStatus {
      kSessionQueryAborted = 0,
      kSessionQuerySubmitted,
      kSessionQueryRunning,
      kSessionQueryStopped,
      kSessionQueryCompleted,
      kSessionQueryFinalized,
      kSessionQueryCreated,
      kSessionQueryFromProof
   };

   ESessionQueryStatus fStatus;     // query status
   TString        fReference;       // query reference string (unique identifier)
   TString        fQueryName;       // query name
   TString        fSelectorString;  // selector name
   TString        fTDSetString;     // dataset name
   TString        fOptions;         // query processing options
   TString        fEventList;       // event list
   Int_t          fNbFiles;         // number of files to process
   Long64_t       fNoEntries;       // number of events/entries to process
   Long64_t       fFirstEntry;      // first event/entry to process
   TTime          fStartTime;       // start time of the query
   TTime          fEndTime;         // end time of the query
   TObject       *fChain;           // dataset on which to process selector
   TQueryResult  *fResult;          // query result received back

   const char    *GetName() const { return fQueryName; }

   ClassDef(TQueryDescription, 1)  // Query description
};


enum EMenuIdentification {
   kMenuAddToFeedback,
   kMenuShow,
   kMenuRemoveFromFeedback
};

//////////////////////////////////////////////////////////////////////////
// TSessionDescription class : Description of Session
//////////////////////////////////////////////////////////////////////////

class TSessionDescription : public TObject {

public:
   TString            fTag;         // session unique identifier
   TString            fName;        // session name
   TString            fAddress;     // server address
   Int_t              fPort;        // communication port
   TString            fConfigFile;  // configuration file name
   Int_t              fLogLevel;    // log (debug) level
   TString            fUserName;    // user name (on server)
   Bool_t             fConnected;   // kTRUE if connected
   Bool_t             fAttached;    // kTRUE if attached
   Bool_t             fLocal;       // kTRUE if session is local
   Bool_t             fSync;        // kTRUE if in sync mode
   Bool_t             fAutoEnable;  // enable packages at session startup time
   TList             *fQueries;     // list of queries in this session
   TList             *fPackages;    // list of packages
   TQueryDescription *fActQuery;    // current (actual) query
   TProof            *fProof;       // pointer on TProof used by this session
   TProofMgr         *fProofMgr;    // Proof sessions manager
   Int_t              fNbHistos;    // number of feedback histos

   const char        *GetName() const { return fName; }

   ClassDef(TSessionDescription, 1) // Session description
};

//////////////////////////////////////////////////////////////////////////
// TPackageDescription class : Description of Package
//////////////////////////////////////////////////////////////////////////

class TPackageDescription : public TObject {

public:
   TString        fName;         // package name
   TString        fPathName;     // full path name of package
   Int_t          fId;           // package id
   Bool_t         fUploaded;     // package has been uploaded
   Bool_t         fEnabled;      // package has been enabled

   const char    *GetName() const { return fName; }

   ClassDef(TPackageDescription, 1) // Package description
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionServerFrame                                                  //
// A composite Frame used in the right part of the Session Viewer GUI   //
// for any information relative to server side : address, port, user... //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSessionServerFrame : public TGCompositeFrame {

private:
   TGCompositeFrame  *fFrmNewServer;   // main group frame
   TGTextEntry       *fTxtName;        // connection name text entry
   TGTextEntry       *fTxtAddress;     // server address text entry
   TGNumberEntry     *fNumPort;        // port number selector
   TGNumberEntry     *fLogLevel;       // log (debug) level selector
   TGTextEntry       *fTxtConfig;      // configuration file text entry
   TGTextEntry       *fTxtUsrName;     // user name text entry
   TGCheckButton     *fSync;           // sync / async flag selector
   TSessionViewer    *fViewer;         // pointer on the main viewer
   TGTextButton      *fBtnAdd;         // "Add" button
   TGTextButton      *fBtnConnect;     // "Connect" button

public:
   TSessionServerFrame(TGWindow *parent, Int_t w, Int_t h);
   virtual ~TSessionServerFrame();

   void        Build(TSessionViewer *gui);

   const char *GetName() const { return fTxtName->GetText(); }
   const char *GetAddress() const { return fTxtAddress->GetText(); }
   Int_t       GetPortNumber() const { return fNumPort->GetIntNumber(); }
   Int_t       GetLogLevel() const { return fLogLevel->GetIntNumber(); }
   const char *GetConfigText() const { return fTxtConfig->GetText(); }
   const char *GetUserName() const { return fTxtUsrName->GetText(); }
   Bool_t      IsSync() const { return (Bool_t)(fSync->GetState() == kButtonDown); }

   void        SetAddEnabled(Bool_t on = kTRUE) {
               on == kTRUE ? ShowFrame(fBtnAdd) : HideFrame(fBtnAdd); }
   void        SetConnectEnabled(Bool_t on = kTRUE) {
               on == kTRUE ? ShowFrame(fBtnConnect) : HideFrame(fBtnConnect); }
   void        SetName(const char *str) { fTxtName->SetText(str); }
   void        SetAddress(const char *str) { fTxtAddress->SetText(str); }
   void        SetPortNumber(Int_t port) { fNumPort->SetIntNumber(port); }
   void        SetLogLevel(Int_t log) { fLogLevel->SetIntNumber(log); }
   void        SetConfigText(const char *str) { fTxtConfig->SetText(str); }
   void        SetUserName(const char *str) { fTxtUsrName->SetText(str); }
   void        SetSync(Bool_t sync) {
               fSync->SetState(sync ? kButtonDown : kButtonUp); }

   void        SettingsChanged();

   void        OnBtnConnectClicked();
   void        OnBtnNewServerClicked();
   void        OnBtnDeleteClicked();
   void        OnBtnAddClicked();
   void        OnConfigFileClicked();
   void        Update(TSessionDescription* desc);
   virtual Bool_t HandleExpose(Event_t *event);
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);

   ClassDef(TSessionServerFrame, 0) // Server frame
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionFrame                                                        //
// A composite Frame used in the right part of the Session Viewer GUI   //
// for any information, settings or controls relative to the current    //
// session.                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSessionFrame : public TGCompositeFrame {

private:

   TGTab             *fTab;                  // main tab frame
   TGCompositeFrame  *fFA, *fFB, *fFC;
   TGCompositeFrame  *fFD, *fFE;             // five tabs element
   TGTextEntry       *fCommandTxt;           // Command line text entry
   TGTextBuffer      *fCommandBuf;           // Command line text buffer
   TGTextView        *fInfoTextView;         // summary on current query
   TGCheckButton     *fClearCheck;           // clear text view after each command
   TGTextButton      *fBtnShowLog;           // show log button
   TGTextButton      *fBtnNewQuery;          // new query button
   TGTextButton      *fBtnGetQueries;        // get entries button
   // Packages tab related items
   TGListBox         *fLBPackages;           // packages listbox
   TGTextButton      *fBtnAdd;               // add package button
   TGTextButton      *fBtnRemove;            // remove package button
   TGTextButton      *fBtnUp;                // move package up button
   TGTextButton      *fBtnDown;              // move package down button
   TGTextButton      *fBtnShow;              // show packages button
   TGTextButton      *fBtnShowEnabled;       // show enabled packages button
   TGCheckButton     *fChkMulti;             // multiple selection check
   TGCheckButton     *fChkEnable;            // enable at session startup check
   TGTextButton      *fBtnUpload;            // upload packages button
   TGTextButton      *fBtnEnable;            // enable packages button
   TGTextButton      *fBtnClear;             // clear all packages button
   TGTextButton      *fBtnDisable;           // disable packages button
   // Datasets tab related items
   TGCanvas          *fDSetView;             // dataset tree view
   TGListTree        *fDataSetTree;          // dataset list tree
   TGTextButton      *fBtnUploadDSet;        // upload dataset button
   TGTextButton      *fBtnRemoveDSet;        // remove dataset button
   TGTextButton      *fBtnVerifyDSet;        // verify dataset button
   TGTextButton      *fBtnRefresh;           // refresh list button
   // Options tab related items
   TGTextEntry       *fTxtParallel;          // parallel nodes text entry
   TGNumberEntry     *fLogLevel;             // log level number entry
   TGTextButton      *fApplyLogLevel;        // apply log level button
   TGTextButton      *fApplyParallel;        // apply parallel nodes button

   TSessionViewer    *fViewer;               // pointer on main viewer
   TGLabel           *fInfoLine[19];         // infos on session

public:
   TSessionFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionFrame();

   void     Build(TSessionViewer *gui);
   void     CheckAutoEnPack(Bool_t checked = kTRUE) {
            fChkEnable->SetState(checked ? kButtonDown : kButtonUp); }
   Int_t    GetLogLevel() const { return fLogLevel->GetIntNumber(); }
   void     SetLogLevel(Int_t log) { fLogLevel->SetIntNumber(log); }
   TGTab   *GetTab() const { return fTab; }

   //Function that handle input from user:
   void     OnApplyLogLevel();
   void     OnApplyParallel();
   void     OnBtnAddClicked();
   void     OnBtnRemoveClicked();
   void     OnBtnUpClicked();
   void     OnBtnDownClicked();
   void     OnBtnShowLogClicked();
   void     OnBtnNewQueryClicked();
   void     OnBtnGetQueriesClicked();
   void     OnBtnDisconnectClicked();
   void     OnCommandLine();
   void     OnUploadPackages();
   void     OnEnablePackages();
   void     OnDisablePackages();
   void     OnClearPackages();
   void     OnMultipleSelection(Bool_t on);
   void     OnStartupEnable(Bool_t on);
   void     ProofInfos();
   void     SetLocal(Bool_t local = kTRUE);
   void     ShutdownSession();
   void     UpdatePackages();
   void     OnBtnUploadDSet();
   void     OnBtnRemoveDSet();
   void     OnBtnVerifyDSet();
   void     UpdateListOfDataSets();

   ClassDef(TSessionFrame, 0) // Session frame
};

//////////////////////////////////////////////////////////////////////////
// New Query Dialog
//////////////////////////////////////////////////////////////////////////

class TEditQueryFrame : public TGCompositeFrame {

private:
   TGCompositeFrame  *fFrmMore;        // options frame
   TGTextButton      *fBtnMore;        // "more >>" / "less <<" button

   TGTextEntry       *fTxtQueryName;   // query name text entry
   TGTextEntry       *fTxtChain;       // chain name text entry
   TGTextEntry       *fTxtSelector;    // selector name text entry
   TGTextEntry       *fTxtOptions;     // options text entry
   TGNumberEntry     *fNumEntries;     // number of entries selector
   TGNumberEntry     *fNumFirstEntry;  // first entry selector
   TGTextEntry       *fTxtParFile;     // parameter file name text entry
   TGTextEntry       *fTxtEventList;   // event list text entry
   TSessionViewer    *fViewer;         // pointer on main viewer
   TQueryDescription *fQuery;          // query description class
   TObject           *fChain;          // actual TChain

public:
   TEditQueryFrame(TGWindow* p, Int_t w, Int_t h);
   virtual ~TEditQueryFrame();
   void     Build(TSessionViewer *gui);
   void     OnNewQueryMore();
   void     OnBrowseChain();
   void     OnBrowseSelector();
   void     OnBrowseEventList();
   void     OnBtnSave();
   void     OnElementSelected(TObject *obj);
   void     SettingsChanged();
   void     UpdateFields(TQueryDescription *desc);

   ClassDef(TEditQueryFrame, 0) // Edit query frame
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionQueryFrame                                                   //
// A composite Frame used in the right part of the Session Viewer GUI   //
// for any information, settings or controls relative to queries.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSessionQueryFrame : public TGCompositeFrame {

private:

   enum EQueryStatus { kRunning = 0, kDone, kStopped, kAborted };

   TGTextButton         *fBtnSubmit;         // submit query button
   TGTextButton         *fBtnFinalize;       // finalize query button
   TGTextButton         *fBtnStop;           // stop process button
   TGTextButton         *fBtnAbort;          // abort process button
   TGTextButton         *fBtnShowLog;        // show log button
   TGTextButton         *fBtnRetrieve;       // retrieve query button
   TGTextButton         *fBtnSave;           // save query button
   TGTextView           *fInfoTextView;      // summary on current query

   Bool_t                fModified;          // kTRUE if settings have changed
   Int_t                 fFiles;             // number of files processed
   Long64_t              fFirst;             // first event/entry to process
   Long64_t              fEntries;           // number of events/entries to process
   Long64_t              fPrevTotal;         // used for progress bar
   Long64_t              fPrevProcessed;     // used for progress bar
   TGLabel              *fLabInfos;          // infos on current process
   TGLabel              *fLabStatus;         // actual process status
   TGLabel              *fTotal;             // total progress info
   TGLabel              *fRate;              // rate of process in events/sec
   EQueryStatus          fStatus;            // status of actual query
   TGTab                *fTab;               // main tab frame
   TGCompositeFrame     *fFA, *fFB, *fFC;    // three tabs element
   TEditQueryFrame      *fFD;                // fourth tab element (edit query frame)
   TGHProgressBar       *frmProg;            // current process progress bar
   TRootEmbeddedCanvas  *fECanvas;           // node statistics embeded canvas
   TCanvas              *fStatsCanvas;       // node statistics canvas
   TSessionViewer       *fViewer;            // pointer on main viewer
   TQueryDescription    *fDesc;              // query description

public:
   TSessionQueryFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionQueryFrame();

   void     Build(TSessionViewer *gui);

   TCanvas *GetStatsCanvas() const { return fStatsCanvas; }
   TEditQueryFrame *GetQueryEditFrame() const { return fFD; }
   TGTab   *GetTab() const { return fTab; }

   void     Feedback(TList *objs);
   void     Modified(Bool_t mod = kTRUE);
   void     Progress(Long64_t total, Long64_t processed);
   void     Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                     Float_t initTime, Float_t procTime,
                     Float_t evtrti, Float_t mbrti, Int_t actw, Int_t tses, Float_t eses);
   void     Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                     Float_t initTime, Float_t procTime,
                     Float_t evtrti, Float_t mbrti) {
                     Progress(total, processed, bytesread, initTime, procTime,
                              evtrti, mbrti, -1, -1, -1.); }
   void     ProgressLocal(Long64_t total, Long64_t processed);
   void     IndicateStop(Bool_t aborted);
   void     ResetProgressDialog(const char *selec, Int_t files, Long64_t first, Long64_t entries);

   //Function that handle input from user:
   void     OnBtnSubmit();
   void     OnBtnFinalize();
   void     OnBtnStop();
   void     OnBtnAbort();
   void     OnBtnShowLog();
   void     OnBtnRetrieve();
   void     UpdateInfos();
   void     UpdateButtons(TQueryDescription *desc);
   void     UpdateHistos(TList *objs);

   ClassDef(TSessionQueryFrame, 0) // Query frame
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionOutputFrame                                                  //
// A composite Frame used in the right part of the Session Viewer GUI   //
// displaying output list objects coming from query result.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSessionOutputFrame : public TGCompositeFrame {

private:
   TGLVEntry              *fEntryTmp;      // used to transfer to feedback
   TGLVContainer          *fLVContainer;   // output list view
   TSessionViewer         *fViewer;        // pointer on the main viewer

public:
   TSessionOutputFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionOutputFrame();

   void           AddObject(TObject *obj);
   void           Build(TSessionViewer *gui);
   TGLVContainer  *GetLVContainer() { return fLVContainer; }
   void           OnElementClicked(TGLVEntry* entry, Int_t btn, Int_t x, Int_t y);
   void           OnElementDblClicked(TGLVEntry *entry ,Int_t btn, Int_t x, Int_t y);
   void           RemoveAll() { fLVContainer->RemoveAll(); }

   ClassDef(TSessionOutputFrame, 0) // Output frame
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionInputFrame                                                   //
// A composite Frame used in the right part of the Session Viewer GUI   //
// displaying input list objects coming from query result.              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSessionInputFrame : public TGCompositeFrame {

private:
   TSessionViewer   *fViewer;       // pointer on the main viewer
   TGLVContainer    *fLVContainer;  // container for the input list view

public:
   TSessionInputFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionInputFrame();

   void           AddObject(TObject *obj);
   void           Build(TSessionViewer *gui);
   void           RemoveAll() { fLVContainer->RemoveAll(); }
   TGLVContainer  *GetLVContainer() { return fLVContainer; }

   ClassDef(TSessionInputFrame, 0) // Input frame
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionViewer                                                       //
// This is the main widget, mother of all the previous classes          //
// Used to manage sessions, servers, queries...                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSessionViewer : public TGMainFrame {

private:
   time_t                  fStart, fElapsed;    // time of connection
   Bool_t                  fChangePic;          // KTRUE if animation active
   Bool_t                  fBusy;               // KTRUE if busy i.e : connecting
   TGHorizontalFrame      *fHf;                 //
   TGVerticalFrame        *fV1;                 //
   TGVerticalFrame        *fV2;                 //
   TSessionServerFrame    *fServerFrame;        // right side server frame
   TSessionFrame          *fSessionFrame;       // right side session frame
   TSessionQueryFrame     *fQueryFrame;         // right side query frame
   TSessionOutputFrame    *fOutputFrame;        // output frame
   TSessionInputFrame     *fInputFrame;         // input frame
   TSessionLogView        *fLogWindow;          // external log window
   TSessionDescription    *fActDesc;            // actual session description
   TList                  *fSessions;           // list of sessions
   const TGPicture        *fLocal;              // local session icon picture
   const TGPicture        *fProofCon;           // connected server icon picture
   const TGPicture        *fProofDiscon;        // disconnected server icon picture
   const TGPicture        *fQueryCon;           // connected(?) query icon picture
   const TGPicture        *fQueryDiscon;        // disconnected(?) query icon picture
   const TGPicture        *fBaseIcon;           // base list tree icon picture

   TGFrame                *fActFrame;           // actual (displayed) frame
   TGToolBar              *fToolBar;            // application tool bar
   TGMenuBar              *fMenuBar;            // application main menu bar
   TGPopupMenu            *fFileMenu;           // file menu entry
   TGPopupMenu            *fSessionMenu;        // session menu entry
   TGPopupMenu            *fQueryMenu;          // query menu entry
   TGPopupMenu            *fOptionsMenu;        // options menu entry
   TGPopupMenu            *fCascadeMenu;        // options menu entry
   TGPopupMenu            *fHelpMenu;           // help menu entry

   TGPopupMenu            *fPopupSrv;           // server related popup menu
   TGPopupMenu            *fPopupQry;           // query related popup menu
   TContextMenu           *fContextMenu;        // input/output objects context menu

   TGHProgressBar         *fConnectProg;        // connection progress bar
   TGCanvas               *fTreeView;           // main right sessions/queries tree view
   TGListTree             *fSessionHierarchy;   // main sessions/queries hierarchy list tree
   TGListTreeItem         *fSessionItem;        // base (main) session list tree item
   TGStatusBar            *fStatusBar;          // bottom status bar
   TGPicture              *fRightIconPicture;   // lower bottom left icon used to show connection status
   TGIcon                 *fRightIcon;          // associated picture
   TTimer                 *fTimer;              // timer used to change icon picture
   UserGroup_t            *fUserGroup;          // user connected to session
   Bool_t                  fAutoSave;           // kTRUE if config is to be saved on exit
   TString                 fConfigFile;         // configuration file name
   TEnv                   *fViewerEnv;          // viewer's configuration

public:

   TSessionViewer(const char *title = "ROOT Session Viewer", UInt_t w = 550, UInt_t h = 320);
   TSessionViewer(const char *title, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual ~TSessionViewer();
   virtual void Build();
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t);

   TSessionServerFrame    *GetServerFrame() const { return fServerFrame; }
   TSessionFrame          *GetSessionFrame() const { return fSessionFrame; }
   TSessionQueryFrame     *GetQueryFrame() const { return fQueryFrame; }
   TSessionOutputFrame    *GetOutputFrame() const { return fOutputFrame; }
   TSessionInputFrame     *GetInputFrame() const { return fInputFrame; }
   TSessionDescription    *GetActDesc() const { return fActDesc; }
   TList                  *GetSessions() const { return fSessions; }
   TGListTree             *GetSessionHierarchy() const { return fSessionHierarchy; }
   TGListTreeItem         *GetSessionItem() const { return fSessionItem; }
   const TGPicture        *GetLocalPict() const { return fLocal; }
   const TGPicture        *GetProofConPict() const { return fProofCon; }
   const TGPicture        *GetProofDisconPict() const { return fProofDiscon; }
   const TGPicture        *GetQueryConPict() const { return fQueryCon; }
   const TGPicture        *GetQueryDisconPict() const { return fQueryDiscon; }
   const TGPicture        *GetBasePict() const { return fBaseIcon; }
   TGPopupMenu            *GetPopupSrv() const { return fPopupSrv; }
   TGPopupMenu            *GetPopupQry() const { return fPopupQry; }
   TContextMenu           *GetContextMenu() const { return fContextMenu; }
   TGStatusBar            *GetStatusBar() const { return fStatusBar; }
   TGHProgressBar         *GetConnectProg() const { return fConnectProg; }
   TGPopupMenu            *GetCascadeMenu() const { return fCascadeMenu; }
   TGPopupMenu            *GetOptionsMenu() const { return fOptionsMenu; }

   void     ChangeRightLogo(const char *name);
   void     CleanupSession();
   void     CloseWindow();
   void     DisableTimer();
   void     EditQuery();
   void     EnableTimer();
   Bool_t   HandleTimer(TTimer *);
   Bool_t   IsBusy() const { return fBusy; }
   Bool_t   IsAutoSave() const { return fAutoSave; }
   void     LogMessage(const char *msg, Bool_t all);
   void     MyHandleMenu(Int_t);
   void     OnCascadeMenu();
   void     OnListTreeClicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y);
   void     OnListTreeDoubleClicked(TGListTreeItem *entry, Int_t btn);
   void     QueryResultReady(char *query);
   void     DeleteQuery();
   void     ReadConfiguration(const char *filename = 0);
   void     ResetSession();
   void     UpdateListOfProofs();
   void     UpdateListOfSessions();
   void     UpdateListOfPackages();
   void     WriteConfiguration(const char *filename = 0);
   void     SetBusy(Bool_t busy = kTRUE) { fBusy = busy; }
   void     SetChangePic(Bool_t change) { fChangePic = change;}
   void     SetLogWindow(TSessionLogView *log) { fLogWindow = log; }
   void     ShowEnabledPackages();
   void     ShowPackages();
   void     ShowInfo(const char *txt);
   void     ShowLog(const char *queryref);
   void     ShowStatus();
   void     StartupMessage(char *msg, Bool_t stat, Int_t curr, Int_t total);
   void     StartViewer();
   void     Terminate();

   ClassDef(TSessionViewer, 0) // Session Viewer
};

R__EXTERN TSessionViewer *gSessionViewer;

#endif
