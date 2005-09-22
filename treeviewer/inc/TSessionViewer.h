// @(#)root/treeviewer:$Name:  $:$Id: TSessionViewer.h
// Author: Marek Biskup, Jakub Madejczyk, Bertrand Bellenot 10/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionViewer                                                       //
//                                                                      //
// Widget used to manage Proof or local sessions, proof connections,    //
// queries construction and results handling.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSessionViewer
#define ROOT_TSessionViewer

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TGTextEntry
#include "TGTextEntry.h"
#endif

#ifndef ROOT_TGNumberEntry
#include "TGNumberEntry.h"
#endif

#ifndef ROOT_TGTab
#include "TGTab.h"
#endif

#ifndef ROOT_TGListView
#include "TGListView.h"
#endif

#ifndef ROOT_TTime
#include "TTime.h"
#endif

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

class TVirtualProof;
class TProofServer;
class TSessionViewer;
class TSessionLogView;
class TQueryResult;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionViewer - A GUI for ROOT / Proof Sessions                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// TQueryDescription class : Descrition of queries
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
   TString        fParFile;         // parameter file name
   Int_t          fNoEntries;       // number of events/entries to process
   Int_t          fFirstEntry;      // first event/entry to process
   TObject        *fChain;          // dataset on which to process selector
   TQueryResult   *fResult;         // query result received back

public:
   ClassDef(TQueryDescription,0)
};


enum EMenuIdentification {
   kMenuAddToFeedback,
   kMenuShow,
   kMenuRemoveFromFeedback
};

//////////////////////////////////////////////////////////////////////////
// TSessionDescription class : Descrition of Session
//////////////////////////////////////////////////////////////////////////

class TSessionDescription : public TObject {

public:
   TString                    fName;         // session name
   TString                    fAddress;      // server address
   Int_t                      fPort;         // communication port
   TString                    fConfigFile;   // configuration file name
   Int_t                      fLogLevel;     // log (debug) level
   TString                    fUserName;     // user name (on server)
   Bool_t                     fConnected;    // kTRUE if connected
   Bool_t                     fLocal;        // kTRUE if session is local
   Bool_t                     fSync;         // kTRUE if in sync mode
   TList                      *fQueries;     // list of queries in this session
   TQueryDescription          *fActQuery;    // current (actual) query
   TVirtualProof              *fProof;       // pointer on TVirtualProof used by this session

   ClassDef(TSessionDescription,0)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionFeedbackFrame                                                //
// A composite Frame used in the right part of the Session Viewer GUI   //
// displaying feedback coming from server on actual query process.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSessionFeedbackFrame : public TGCompositeFrame {

private:
   TGTextButton      *fBtnAdd;            // Add button
   TGTextButton      *fBtnSet;            // Set button
   TGNumberEntry     *fNumEntFrequency;   // update frequency selector
   TGLabel           *fLabFrequency;      // update frequency label
   TGTextEntry       *fTexEntAdd;         // item to add text entry
   TGTextBuffer      *fTBAdd;             // related text buffer
   TGLVContainer     *fLVContainer;       // container for the list of items in feedback
   TGListView        *fListView;          // list of items in feedback

   TGLVEntry         *fDelEntry;          // pointer to the element chosen to be removed
   TGPopupMenu       *fPopupMenu;         // popup menu to use with items in the listview
   TSessionViewer    *fViewer;            // pointer on the main viewer

public:
   TSessionFeedbackFrame(TGWindow *parent, Int_t w, Int_t h);
   virtual ~TSessionFeedbackFrame();

   void     Build(TSessionViewer *gui);
   void     AddToFeedback(TObject *obj);
   void     AddToFeedback(TGLVEntry *entry);

   //Function that handle input from user:
   void     OnChBtnPressed();
   void     OnChBtnReleased();
   void     OnBtnAddClicked();

   void     OnElementClicked(TGLVEntry *entry ,Int_t btn, Int_t x, Int_t y);
   void     MyHandleMenu(Int_t id);

   ClassDef(TSessionFeedbackFrame,0)
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

public:
   TSessionServerFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionServerFrame();

   void        Build(TSessionViewer *gui);

   const char *GetName() const { return fTxtName->GetText(); }
   const char *GetAddress() const { return fTxtAddress->GetText(); }
   Int_t       GetPortNumber() { return fNumPort->GetIntNumber(); }
   Int_t       GetLogLevel() { return fLogLevel->GetIntNumber(); }
   const char *GetConfigText() const { return fTxtConfig->GetText(); }
   const char *GetUserName() const { return fTxtUsrName->GetText(); }
   Bool_t      IsSync() { return (Bool_t)(fSync->GetState() == kButtonDown); }

   void        SetName(const char *str) { fTxtName->SetText(str); }
   void        SetAddress(const char *str) { fTxtAddress->SetText(str); }
   void        SetPortNumber(Int_t port) { fNumPort->SetIntNumber(port); }
   void        SetLogLevel(Int_t log) { fLogLevel->SetIntNumber(log); }
   void        SetConfigText(const char *str) { fTxtConfig->SetText(str); }
   void        SetUserName(const char *str) { fTxtUsrName->SetText(str); }
   void        SetSync(Bool_t sync) { fSync->SetState(sync ? kButtonDown : kButtonUp); }

   void        OnBtnConnectClicked();
   void        OnBtnNewServerClicked();
   void        OnBtnDeleteClicked();
   void        OnBtnAddClicked();
   void        OnConfigFileClicked();
   void        Update(TSessionDescription* desc);

   Bool_t      WriteConfigFile(const TString &filePath, TList *vec);
   TList      *ReadConfigFile(const TString &filePath);

   ClassDef(TSessionServerFrame,0)
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

   enum EQueryStatus { kRunning = 0, kDone, kStopped, kAborted };

   Int_t             fFiles;                 // number of files processed
   TTime             fStartTime, fEndTime;   // start and end time of the process
   Long64_t          fFirst;                 // first event/entry to process
   Long64_t          fEntries;               // number of events/entries to process
   Long64_t          fPrevTotal;             // used for progress bar
   TGLabel           *fLabStatus;            // actual process status
   TGLabel           *fProcessed;            // actual progress informations
   TGLabel           *fTotal;                // total progress info
   TGLabel           *fRate;                 // rate of process in events/sec
   EQueryStatus      fStatus;                // status of actual query
   TGTab             *fTab;                  // main tab frame
   TGCompositeFrame  *fFA, *fFB;             // two tabs element
   TSessionFeedbackFrame *fFeedbackFrame;    // current session feedback
   TGTextEntry       *fTexEntResultsURL;     // results URL text entry
   TGTextBuffer      *fTexBufResultsURL;     // results URL text buffer
   TGTextButton      *fBtnDisconnect;        // disconnect button
   TGTextButton      *fBtnShowLog;           // show log button
   TGTextButton      *fBtnNewQuery;          // new query button
   TGTextButton      *fBtnGetQueries;        // get entries button
   TGHProgressBar    *frmProg;               // current process progress bar
   TSessionViewer    *fViewer;               // pointer on main viewer

public:
   TSessionFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionFrame();

   void     Build(TSessionViewer *gui);

   TTime    GetStartTime() { return fStartTime; }
   TTime    GetEndTime()   { return fEndTime; }
   Int_t    GetNumberOfFiles() { return fFiles; }
   Long64_t GetFirstEntry() { return fFirst; }
   Long64_t GetEntries() { return fEntries; }

   void     SetStartTime(TTime time) { fStartTime = time; }
   void     SetEndTime(TTime time) { fEndTime = time; }
   void     SetNumberOfFiles(Int_t number) { fFiles = number; }
   void     SetFirstEntry(Long64_t entry) { fFirst = entry; }
   void     SetEntries(Long64_t entries) { fEntries = entries; }

   //Function that handle input from user:
   void     OnBtnShowLogClicked();
   void     OnBtnNewQueryClicked();
   void     OnBtnGetQueriesClicked();
   void     OnBtnDisconnectClicked();

   void     Feedback(TList *objs);
   void     Progress(Long64_t total, Long64_t processed);
   void     IndicateStop(Bool_t aborted);
   void     ResetProgressDialog(const char *selec, Int_t files, Long64_t first, Long64_t entries);

   ClassDef(TSessionFrame,0)
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

   TGTextEntry       *fTexEntResultsURL;     // results URL text entry
   TGTextBuffer      *fTexBufResultsURL;     // results URL text buffer
   TGTab             *fTab;                  // main tab frame
   TGCompositeFrame  *fFA, *fFB;             // two tabs element
   TRootEmbeddedCanvas *fECanvas;            // node statistics embeded canvas
   TCanvas           *fStatsCanvas;          // node statistics canvas
   TGTextButton      *fBtnSubmit;            // submit query button
   TGTextButton      *fBtnFinalize;          // finalize query button
   TGTextButton      *fBtnStop;              // stop process button
   TGTextButton      *fBtnAbort;             // abort process button
   TGTextButton      *fBtnShowLog;           // show log button
   TGTextButton      *fBtnRetrieve;          // retrieve query button
   TGCheckButton     *fFeedbackChk;          // Feedback check button
   TGTextView        *fInfoTextView;         // summary on current query
   TSessionLogView   *fLogWindow;            // external log window
   TSessionViewer    *fViewer;               // pointer on main viewer

   TQueryDescription *fDesc;

public:
   TSessionQueryFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionQueryFrame();

   void     Build(TSessionViewer *gui);
   void     Feedback(TList *objs);

   //Function that handle input from user:
   void     OnBtnSubmit();
   void     OnBtnFinalize();
   void     OnBtnStop();
   void     OnBtnAbort();
   void     OnBtnShowLog();
   void     OnBtnRetrieve();
   void     UpdateInfos();
   void     UpdateButtons(TQueryDescription *desc);

   void     SetTab(Int_t tab) { fTab->SetTab(tab); }
   void     SetTabEnabled(Int_t tab, Bool_t en) { fTab->SetEnabled(tab, en); }

   ClassDef(TSessionQueryFrame,0)
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
   TGLVEntry        *fEntryTmp;     // used to transfer to feedback
   TGLVContainer    *fLVContainer;  // output list view
   TGPopupMenu      *fPopupMenu;    // popup menu to use with items in the listview
   TSessionViewer   *fViewer;       // pointer on the main viewer
   TSessionFeedbackFrame  *fFeedbackFrame;

public:
   TSessionOutputFrame(TGWindow* parent, Int_t w, Int_t h);
   virtual ~TSessionOutputFrame();

   void           AddObject(TObject *obj);
   void           Build(TSessionViewer *gui);
   TGLVContainer  *GetLVContainer() { return fLVContainer; }
   void           MyHandleMenu(Int_t id);
   void           OnElementClicked(TGLVEntry* entry, Int_t btn, Int_t x, Int_t y);
   void           OnElementDblClicked(TGLVEntry *entry ,Int_t btn, Int_t x, Int_t y);
   void           RemoveAll() { fLVContainer->RemoveAll(); }
   void           SetFeedbackFrame(TSessionFeedbackFrame* feedbackFrame)
                     { fFeedbackFrame = feedbackFrame; }

   ClassDef(TSessionOutputFrame,0)
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

   ClassDef(TSessionInputFrame,0)
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
   TGHorizontalFrame      *fHf;                 //
   TGVerticalFrame        *fV1;                 //
   TGVerticalFrame        *fV2;                 //
   TSessionServerFrame    *fServerFrame;        // right side server frame
   TSessionFrame          *fSessionFrame;       // right side session frame
   TSessionQueryFrame     *fQueryFrame;         // right side query frame
   TSessionFeedbackFrame  *fFeedbackFrame;      // feedback frame
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

public:

   TSessionViewer(const char *title = "ROOT Session Viewer", UInt_t w = 550, UInt_t h = 320);
   TSessionViewer(const char *title, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual ~TSessionViewer();
   virtual void Build();
   virtual void BuildSessionHierarchy(TList *vec);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t);

   TSessionServerFrame    *GetServerFrame() { return fServerFrame; }
   TSessionFrame          *GetSessionFrame() { return fSessionFrame; }
   TSessionQueryFrame     *GetQueryFrame() { return fQueryFrame; }
   TSessionFeedbackFrame  *GetFeedbackFrame() { return fFeedbackFrame; }
   TSessionOutputFrame    *GetOutputFrame() { return fOutputFrame; }
   TSessionInputFrame     *GetInputFrame() { return fInputFrame; }
   TSessionDescription    *GetActDesc() { return fActDesc; }
   TList                  *GetSessions() { return fSessions; }
   TGListTree             *GetSessionHierarchy() { return fSessionHierarchy; }
   TGListTreeItem         *GetSessionItem() { return fSessionItem; }
   const TGPicture        *GetLocalPict() { return fLocal; }
   const TGPicture        *GetProofConPict() { return fProofCon; }
   const TGPicture        *GetProofDisconPict() { return fProofDiscon; }
   const TGPicture        *GetQueryConPict() { return fQueryCon; }
   const TGPicture        *GetQueryDisconPict() { return fQueryDiscon; }
   const TGPicture        *GetBasePict() { return fBaseIcon; }
   TGPopupMenu            *GetPopupSrv() { return fPopupSrv; }
   TGPopupMenu            *GetPopupQry() { return fPopupQry; }
   TContextMenu           *GetContextMenu() { return fContextMenu; }
   TGStatusBar            *GetStatusBar() { return fStatusBar; }
   TGHProgressBar         *GetConnectProg() { return fConnectProg; }

   void     ChangeRightLogo(const char *name);
   void     CleanupSession();
   void     CloseWindow();
   void     DeleteQuery();
   void     DisableTimer();
   void     EditQuery();
   void     EnableTimer();
   Bool_t   HandleTimer(TTimer *);
   void     LogMessage(const char *msg, Bool_t all);
   void     MyHandleMenu(Int_t);
   void     OnListTreeClicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y);
   void     QueryResultReady(char *query);
   void     RemoveQuery();
   void     SetChangePic(Bool_t change) { fChangePic = change;}
   void     SetLogWindow(TSessionLogView *log) { fLogWindow = log; }
   void     ShowInfo(const char *txt);
   void     ShowLog(const char *queryref);
   void     StartupMessage(char *msg, Bool_t stat, Int_t curr, Int_t total);
   void     StartViewer();

   ClassDef(TSessionViewer,0)
};

#endif
