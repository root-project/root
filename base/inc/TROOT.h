// @(#)root/base:$Name:  $:$Id: TROOT.h,v 1.10 2000/11/27 10:42:27 rdm Exp $
// Author: Rene Brun   08/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TROOT
#define ROOT_TROOT


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TROOT                                                                //
//                                                                      //
// The TROOT object is the entry point to the system.                   //
// The single instance of TROOT is accessable via the global gROOT.     //
// Using the gROOT pointer one has access to basically every object     //
// created in a ROOT based program. The TROOT object is essentially a   //
// "dispatcher" with several lists pointing to the ROOT main objects.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDirectory
#include "TDirectory.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif

class TClass;
class TColor;
class TDataType;
class TFile;
class TStyle;
class TVirtualPad;
class TApplication;
class TInterpreter;
class TBrowser;
class TGlobal;
class TFunction;
class TFolder;

class TROOT : public TDirectory {

friend class TCint;

private:
   static Int_t    fgDirLevel;            //indentation level for ls()
   static Bool_t   fgRootInit;            //Singleton initialization flag
   Int_t           fLineIsProcessing;     //To synchronize multi-threads

protected:
   TString         fVersion;              //ROOT version (from CMZ VERSQQ) ex 0.05/01
   Int_t           fVersionInt;           //ROOT version in integer format (501)
   Int_t           fVersionDate;          //Date of ROOT version (ex 951226)
   Int_t           fVersionTime;          //Time of ROOT version (ex 1152)
   Int_t           fTimer;                //Timer flag
   TApplication    *fApplication;         //Pointer to current application
   TInterpreter    *fInterpreter;         //Command interpreter
   TFile           *fCurrentFile;         //Current file
   TDirectory      *fCurrentDirectory;    //Current directory
   TVirtualPad     *fCurrentCanvas;       //Current graphics canvas
   TVirtualPad     *fCurrentPad;          //Current graphics pad
   TStyle          *fCurrentStyle;        //Current graphics style
   Bool_t          fBatch;                //True if session without graphics
   Bool_t          fEditHistograms;       //True if histograms can be edited with the mouse
   Bool_t          fFromPopUp;            //True if command executed from a popup menu
   Bool_t          fMustClean;            //True if object destructor scans canvases
   Bool_t          fReadingObject;        //True while reading an object
   Bool_t          fForceStyle;           //Force setting of current style when reading objects
   Bool_t          fInterrupt;            //True if macro should be interrupted
   Int_t           fEditorMode;           //Current Editor mode
   TObject         *fPrimitive;           //Currently selected primitive
   TVirtualPad     *fSelectPad;           //Currently selected pad
   TSeqCollection  *fClasses;             //List of classes definition
   TSeqCollection  *fTypes;               //List of data types definition
   TSeqCollection  *fGlobals;             //List of global variables
   TSeqCollection  *fGlobalFunctions;     //List of global functions
   TSeqCollection  *fFiles;               //List of files
   TSeqCollection  *fMappedFiles;         //List of memory mapped files
   TSeqCollection  *fSockets;             //List of network sockets
   TSeqCollection  *fCanvases;            //List of canvases
   TSeqCollection  *fStyles;              //List of styles
   TSeqCollection  *fFunctions;           //List of analytic functions
   TSeqCollection  *fTasks;               //List of tasks
   TSeqCollection  *fColors;              //List of colors
   TSeqCollection  *fGeometries;          //List of geometries
   TSeqCollection  *fBrowsers;            //List of browsers
   TSeqCollection  *fSpecials;            //List of special objects
   TSeqCollection  *fCleanups;            //List of recursiveRemove collections
   TSeqCollection  *fMessageHandlers;     //List of message handlers
   TSeqCollection  *fStreamerInfo;        //List of active StreamerInfo classes
   TFolder         *fRootFolder;          //top level folder //root
   TList           *fBrowsables;          //List of browsables
   TString         fCutClassName;         //Name of default CutG class in graphics editor
   TString         fDefCanvasName;        //Name of default canvas

   static VoidFuncPtr_t fgMakeDefCanvas;  //Pointer to default canvas constructor

                   TROOT();               //Only used by Dictionary
   void            InitSystem();          //Operating System interface
   void            InitThreads();         //Initialize threads library

   void           *operator new(size_t l) { return TObject::operator new(l); }

public:
                     TROOT(const char *name, const char *title, VoidFuncPtr_t *initfunc = 0);
   virtual           ~TROOT();
   void              Browse(TBrowser *b);
   Bool_t            ClassSaved(TClass *cl);
   virtual TObject  *FindObject(const char *name) const;
   virtual TObject  *FindObject(TObject *obj) const;
   virtual TObject  *FindObjectAny(const char *name) const;
   TObject          *FindSpecialObject(const char *name, void *&where);
   const char       *FindObjectClassName(const char *name) const;
   const char       *FindObjectPathName(TObject *obj) const;
   void              ForceStyle(Bool_t force=kTRUE) {fForceStyle = force;}
   Bool_t            FromPopUp() {return fFromPopUp;}
   TApplication     *GetApplication() {return fApplication;}
   TClass           *GetClass(const char *name, Bool_t load=kTRUE);
   TColor           *GetColor(Int_t color);
   const char       *GetCutClassName() const {return fCutClassName.Data();}
   const char       *GetDefCanvasName() const {return fDefCanvasName.Data();}
   Bool_t            GetEditHistograms() {return fEditHistograms;}
   Int_t             GetEditorMode() {return fEditorMode;}
   Bool_t            GetForceStyle() {return fForceStyle;}
   VoidFuncPtr_t     GetMakeDefCanvas();
   Int_t             GetVersionDate() {return fVersionDate;}
   Int_t             GetVersionTime() {return fVersionTime;}
   Int_t             GetVersionInt() {return fVersionInt;}
   const char       *GetVersion() const {return fVersion.Data();}
   TSeqCollection   *GetListOfClasses()   {return fClasses;}
   TSeqCollection   *GetListOfColors()    {return fColors;}
   TSeqCollection   *GetListOfTypes(Bool_t load = kFALSE);
   TSeqCollection   *GetListOfGlobals(Bool_t load = kFALSE);
   TSeqCollection   *GetListOfGlobalFunctions(Bool_t load = kFALSE);
   TSeqCollection   *GetListOfFiles()      {return fFiles;}
   TSeqCollection   *GetListOfMappedFiles(){return fMappedFiles;}
   TSeqCollection   *GetListOfSockets()    {return fSockets;}
   TSeqCollection   *GetListOfCanvases()   {return fCanvases;}
   TSeqCollection   *GetListOfStyles()     {return fStyles;}
   TSeqCollection   *GetListOfFunctions()  {return fFunctions;}
   TSeqCollection   *GetListOfGeometries() {return fGeometries;}
   TSeqCollection   *GetListOfBrowsers()   {return fBrowsers;}
   TSeqCollection   *GetListOfSpecials()   {return fSpecials;}
   TSeqCollection   *GetListOfTasks()      {return fTasks;}
   TSeqCollection   *GetListOfCleanups()   {return fCleanups;}
   TSeqCollection   *GetListOfStreamerInfo()    {return fStreamerInfo;}
   TSeqCollection   *GetListOfMessageHandlers() {return fMessageHandlers;}
   TList            *GetListOfBrowsables() {return fBrowsables;}
   TDataType        *GetType(const char *name, Bool_t load = kFALSE);
   TFile            *GetFile() {return fFile;}
   TFile            *GetFile(const char *name);
   TStyle           *GetStyle(const char *name);
   TObject          *GetFunction(const char *name);
   TGlobal          *GetGlobal(const char *name, Bool_t load = kFALSE);
   TGlobal          *GetGlobal(TObject *obj, Bool_t load = kFALSE);
   TFunction        *GetGlobalFunction(const char *name, const char *params = 0, Bool_t load = kFALSE);
   TFunction        *GetGlobalFunctionWithPrototype(const char *name, const char *proto = 0, Bool_t load = kFALSE);
   TObject          *GetGeometry(const char *name);
   TObject          *GetSelectedPrimitive() {return fPrimitive;}
   TVirtualPad      *GetSelectedPad() {return fSelectPad;}
   Int_t             GetNclasses() {return fClasses->GetSize();}
   Int_t             GetNtypes() {return fTypes->GetSize();}
   TFolder          *GetRootFolder() {return fRootFolder;}
   void              Idle(UInt_t idleTimeInSec, const char *command=0);
   Int_t             IgnoreInclude(const char *fname, const char *expandedfname);
   Bool_t            IsBatch() const { return fBatch; }
   Bool_t            IsFolder() const {return kTRUE;}
   Bool_t            IsInterrupted() const { return fInterrupt; }
   Bool_t            IsLineProcessing() const { return fLineIsProcessing; }
   Bool_t            IsProofServ() const { return fName == "Proofserv" ? kTRUE : kFALSE; }
   void              ls(Option_t *option="");
   Int_t             LoadClass(const char *classname, const char *libname);
   void              LoadMacro(const char *filename);
   Int_t             Macro(const char *filename);
   void              Message(Int_t id, TObject *obj);
   Bool_t            MustClean() {return fMustClean;}
   void              ProcessLine(const char *line);
   void              ProcessLineSync(const char *line);
   Long_t            ProcessLineFast(const char *line);
   void              Proof(const char *cluster = "proof://localhost");
   Bool_t            ReadingObject() {return fReadingObject;}
   void              Reset(Option_t *option="");
   void              SaveContext();
   void              SetApplication(TApplication *app) { fApplication = app; }
   void              SetBatch(Bool_t batch=kTRUE) { fBatch = batch; }
   void              SetCutClassName(const char *name="TCutG");
   void              SetDefCanvasName(const char *name="c1") {fDefCanvasName = name;}
   void              SetEditHistograms(Bool_t flag=kTRUE) {fEditHistograms=flag;}
   void              SetEditorMode(const char *mode="");
   void              SetFromPopUp(Bool_t flag=kTRUE) { fFromPopUp = flag; }
   void              SetInterrupt(Bool_t flag=kTRUE) { fInterrupt = flag; }
   void              SetLineIsProcessing() { fLineIsProcessing++; }
   void              SetLineHasBeenProcessed() {if (fLineIsProcessing) fLineIsProcessing--;}
   void              SetReadingObject(Bool_t flag=kTRUE) {fReadingObject = flag;}
   void              SetMustClean(Bool_t flag=kTRUE) { fMustClean=flag; }
   void              SetSelectedPrimitive(TObject *obj) { fPrimitive = obj; }
   void              SetSelectedPad(TVirtualPad *pad) { fSelectPad = pad; }
   void              SetStyle(const char *stylename="Default");
   void              Time(Int_t casetime=1) { fTimer = casetime; }
   Int_t             Timer() { return fTimer; }

   //---- static functions
   static Int_t       DecreaseDirLevel();
   static Int_t       GetDirLevel();
   static const char *GetMacroPath();
   static Int_t       IncreaseDirLevel();
   static void        IndentLevel();
   static Bool_t      Initialized();
   static void        SetDirLevel(Int_t level=0);
   static void        SetMakeDefCanvas(VoidFuncPtr_t makecanvas);

   ClassDef(TROOT,0)  //Top level (or root) structure for all classes
};


R__EXTERN TROOT  *gROOT;

#endif

