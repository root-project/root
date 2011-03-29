// @(#)root/base:$Id$
// Author: Rene Brun   08/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//                R O O T top level object description
//
//    The TROOT object is the entry point to the ROOT system.
//    The single instance of TROOT is accessible via the global gROOT.
//    Using the gROOT pointer one has access to basically every object
//    created in a ROOT based program. The TROOT object is essentially a
//    container of several lists pointing to the main ROOT objects.
//
//    The following lists are accessible from gROOT object:
//       gROOT->GetListOfClasses
//       gROOT->GetListOfColors
//       gROOT->GetListOfTypes
//       gROOT->GetListOfGlobals
//       gROOT->GetListOfGlobalFunctions
//       gROOT->GetListOfFiles
//       gROOT->GetListOfMappedFiles
//       gROOT->GetListOfSockets
//       gROOT->GetListOfSecContexts
//       gROOT->GetListOfCanvases
//       gROOT->GetListOfStyles
//       gROOT->GetListOfFunctions
//       gROOT->GetListOfSpecials (for example graphical cuts)
//       gROOT->GetListOfGeometries
//       gROOT->GetListOfBrowsers
//       gROOT->GetListOfCleanups
//       gROOT->GetListOfMessageHandlers
//
//   The TROOT class provides also many useful services:
//     - Get pointer to an object in any of the lists above
//     - Time utilities TROOT::Time
//
//   The ROOT object must be created as a static object. An example
//   of a main program creating an interactive version is shown below:
//
//---------------------Example of a main program--------------------------------
//
//       #include "TRint.h"
//
//       int main(int argc, char **argv)
//       {
//          TRint *theApp = new TRint("ROOT example", &argc, argv);
//
//          // Init Intrinsics, build all windows, and enter event loop
//          theApp->Run();
//
//          return(0);
//       }
//-----------------------End of Main program--------------------------------
////////////////////////////////////////////////////////////////////////////////

#include "RConfig.h"
#include "RConfigure.h"
#include "RConfigOptions.h"

#include <string>
#include <map>
#include <stdlib.h>
#ifdef WIN32
#include <io.h>
#endif

#include "Riostream.h"
#include "Gtypes.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassGenerator.h"
#include "TDataType.h"
#include "TDatime.h"
#include "TStyle.h"
#include "TObjectTable.h"
#include "TClassTable.h"
#include "TSystem.h"
#include "THashList.h"
#include "TObjArray.h"
#include "TEnv.h"
#include "TError.h"
#include "TColor.h"
#include "TGlobal.h"
#include "TFunction.h"
#include "TVirtualPad.h"
#include "TBrowser.h"
#include "TSystemDirectory.h"
#include "TApplication.h"
#include "TInterpreter.h"
#include "TGuiFactory.h"
#include "TMessageHandler.h"
#include "TFolder.h"
#include "TQObject.h"
#include "TProcessUUID.h"
#include "TPluginManager.h"
#include "TMap.h"
#include "TObjString.h"
#include "TVirtualMutex.h"
#ifdef R__HAS_CLING
# include "TCling.h"
#else
# include "TCint.h"
#endif

#include <string>
namespace std {} using namespace std;

#if defined(R__UNIX)
#include "TUnixSystem.h"
#elif defined(R__WIN32)
#include "TWinNTSystem.h"
#endif

extern "C" void R__SetZipMode(int);

// Mutex for protection of concurrent gROOT access
TVirtualMutex* gROOTMutex = 0;

//-------- Names of next three routines are a small homage to CMZ --------------
//______________________________________________________________________________
static Int_t IVERSQ()
{
   // Return version id as an integer, i.e. "2.22/04" -> 22204.

   Int_t maj, min, cycle;
   sscanf(ROOT_RELEASE, "%d.%d/%d", &maj, &min, &cycle);
   return 10000*maj + 100*min + cycle;
}

//______________________________________________________________________________
static Int_t IDATQQ(const char *date)
{
   // Return built date as integer, i.e. "Apr 28 2000" -> 20000428.

   static const char *months[] = {"Jan","Feb","Mar","Apr","May",
                                  "Jun","Jul","Aug","Sep","Oct",
                                  "Nov","Dec"};

   char  sm[12];
   Int_t yy, mm=0, dd;
   sscanf(date, "%s %d %d", sm, &dd, &yy);
   for (int i = 0; i < 12; i++)
      if (!strncmp(sm, months[i], 3)) {
         mm = i+1;
         break;
      }
   return 10000*yy + 100*mm + dd;
}

//______________________________________________________________________________
static Int_t ITIMQQ(const char *time)
{
   // Return built time as integer (with min precision), i.e.
   // "17:32:37" -> 1732.

   Int_t hh, mm, ss;
   sscanf(time, "%d:%d:%d", &hh, &mm, &ss);
   return 100*hh + mm;
}
//______________________________________________________________________________
static void CleanUpROOTAtExit()
{
   // Clean up at program termination before global objects go out of scope.

   if (gROOT) {
      R__LOCKGUARD(gROOTMutex);

      if (gROOT->GetListOfFiles())
         gROOT->GetListOfFiles()->Delete("slow");
      if (gROOT->GetListOfSockets())
         gROOT->GetListOfSockets()->Delete();
      if (gROOT->GetListOfMappedFiles())
         gROOT->GetListOfMappedFiles()->Delete("slow");
   }
}



Int_t  TROOT::fgDirLevel = 0;
Bool_t TROOT::fgRootInit = kFALSE;
Bool_t TROOT::fgMemCheck = kFALSE;

// This local static object initializes the ROOT system
namespace ROOT {
   TROOT *GetROOT() {
      static TROOT root("root", "The ROOT of EVERYTHING");
      return &root;
   }
   TString &GetMacroPath() {
      static TString macroPath;
      return macroPath;
   }
}

TROOT *gROOT = ROOT::GetROOT();     // The ROOT of EVERYTHING

// Global debug flag (set to > 0 to get debug output).
// Can be set either via the interpreter (gDebug is exported to CINT),
// via the rootrc resouce "Root.Debug", via the shell environment variable
// ROOTDEBUG, or via the debugger.
Int_t gDebug;


ClassImp(TROOT)

//______________________________________________________________________________
TROOT::TROOT() : TDirectory(),
     fLineIsProcessing(0), fVersion(0), fVersionInt(0), fVersionCode(0),
     fVersionDate(0), fVersionTime(0), fBuiltDate(0), fBuiltTime(0), fSvnRevision(0),
     fTimer(0), fApplication(0), fInterpreter(0), fBatch(kTRUE), fEditHistograms(kTRUE),
     fFromPopUp(kTRUE),fMustClean(kTRUE),fReadingObject(kFALSE),fForceStyle(kFALSE),
     fInterrupt(kFALSE),fEscape(kFALSE),fExecutingMacro(kFALSE),fEditorMode(0),
     fPrimitive(0),fSelectPad(0),fClasses(0),fTypes(0),fGlobals(0),fGlobalFunctions(0),
     fFiles(0),fMappedFiles(0),fSockets(0),fCanvases(0),fStyles(0),fFunctions(0),
     fTasks(0),fColors(0),fGeometries(0),fBrowsers(0),fSpecials(0),fCleanups(0),
     fMessageHandlers(0),fStreamerInfo(0),fClassGenerators(0),fSecContexts(0),
     fProofs(0),fClipboard(0),fDataSets(0),fUUIDs(0),fRootFolder(0),fBrowsables(0),
     fPluginManager(0)
{
   // Default ctor.
}

//______________________________________________________________________________
TROOT::TROOT(const char *name, const char *title, VoidFuncPtr_t *initfunc)
   : TDirectory(), fLineIsProcessing(0), fVersion(0), fVersionInt(0), fVersionCode(0),
     fVersionDate(0), fVersionTime(0), fBuiltDate(0), fBuiltTime(0), fSvnRevision(0),
     fTimer(0), fApplication(0), fInterpreter(0), fBatch(kTRUE), fEditHistograms(kTRUE),
     fFromPopUp(kTRUE),fMustClean(kTRUE),fReadingObject(kFALSE),fForceStyle(kFALSE),
     fInterrupt(kFALSE),fEscape(kFALSE),fExecutingMacro(kFALSE),fEditorMode(0),
     fPrimitive(0),fSelectPad(0),fClasses(0),fTypes(0),fGlobals(0),fGlobalFunctions(0),
     fFiles(0),fMappedFiles(0),fSockets(0),fCanvases(0),fStyles(0),fFunctions(0),
     fTasks(0),fColors(0),fGeometries(0),fBrowsers(0),fSpecials(0),fCleanups(0),
     fMessageHandlers(0),fStreamerInfo(0),fClassGenerators(0),fSecContexts(0),
     fProofs(0),fClipboard(0),fDataSets(0),fUUIDs(0),fRootFolder(0),fBrowsables(0),
     fPluginManager(0)
{
   // Initialize the ROOT system. The creation of the TROOT object initializes
   // the ROOT system. It must be the first ROOT related action that is
   // performed by a program. The TROOT object must be created on the stack
   // (can not be called via new since "operator new" is protected). The
   // TROOT object is either created as a global object (outside the main()
   // program), or it is one of the first objects created in main().
   // Make sure that the TROOT object stays in scope for as long as ROOT
   // related actions are performed. TROOT is a so called singleton so
   // only one instance of it can be created. The single TROOT object can
   // always be accessed via the global pointer gROOT.
   // The name and title arguments can be used to identify the running
   // application. The initfunc argument can contain an array of
   // function pointers (last element must be 0). These functions are
   // executed at the end of the constructor. This way one can easily
   // extend the ROOT system without adding permanent dependencies
   // (e.g. the graphics system is initialized via such a function).

   if (fgRootInit) {
      //Warning("TROOT", "only one instance of TROOT allowed");
      return;
   }

   R__LOCKGUARD2(gROOTMutex);

   gROOT      = this;
   gDirectory = 0;
   SetName(name);
   SetTitle(title);
   TDirectory::Build();

   // will be used by global "operator delete" so make sure it is set
   // before anything is deleted
   fMappedFiles = 0;

   // create already here, but only initialize it after gEnv has been created
   gPluginMgr = fPluginManager = new TPluginManager;

   // Initialize Operating System interface
   InitSystem();

#ifndef ROOTPREFIX
   if (!gSystem->Getenv("ROOTSYS")) {
      fprintf(stderr, "Fatal in <TROOT::TROOT>: ROOTSYS not set. Set it before trying to run.\n");
      exit(1);
   }
#endif

   // Initialize interface to CINT C++ interpreter
   fVersionInt      = 0;  // check in TROOT dtor in case TCint fails
   fClasses         = 0;  // might be checked via TCint ctor

   fInterpreter     = new TCint("C/C++", "CINT C/C++ Interpreter");

   fConfigOptions   = R__CONFIGUREOPTIONS;
   fConfigFeatures  = R__CONFIGUREFEATURES;
   fVersion         = ROOT_RELEASE;
   fVersionCode     = ROOT_VERSION_CODE;
   fVersionInt      = IVERSQ();
   fVersionDate     = IDATQQ(ROOT_RELEASE_DATE);
   fVersionTime     = ITIMQQ(ROOT_RELEASE_TIME);
   fBuiltDate       = IDATQQ(__DATE__);
   fBuiltTime       = ITIMQQ(__TIME__);

   ReadSvnInfo();

   fClasses         = new THashTable(800,3);
   //fIdMap           = new IdMap_t;
   fStreamerInfo    = new TObjArray(100);
   fClassGenerators = new TList;

   // initialize plugin manager early
   fPluginManager->LoadHandlersFromEnv(gEnv);

   TSystemDirectory *workdir = new TSystemDirectory("workdir", gSystem->WorkingDirectory());

   fTimer       = 0;
   fApplication = 0;
   fColors      = new TObjArray(1000); fColors->SetName("ListOfColors");
   fTypes       = 0;
   fGlobals     = 0;
   fGlobalFunctions = 0;
   // fList was created in TDirectory::Build but with different sizing.
   delete fList;
   fList        = new THashList(1000,3);
   fFiles       = new TList; fFiles->SetName("Files");
   fMappedFiles = new TList; fMappedFiles->SetName("MappedFiles");
   fSockets     = new TList; fSockets->SetName("Sockets");
   fCanvases    = new TList; fCanvases->SetName("Canvases");
   fStyles      = new TList; fStyles->SetName("Styles");
   fFunctions   = new TList; fFunctions->SetName("Functions");
   fTasks       = new TList; fTasks->SetName("Tasks");
   fGeometries  = new TList; fGeometries->SetName("Geometries");
   fBrowsers    = new TList; fBrowsers->SetName("Browsers");
   fSpecials    = new TList; fSpecials->SetName("Specials");
   fBrowsables  = new TList; fBrowsables->SetName("Browsables");
   fCleanups    = new THashList; fCleanups->SetName("Cleanups");
   fMessageHandlers = new TList; fMessageHandlers->SetName("MessageHandlers");
   fSecContexts = new TList; fSecContexts->SetName("SecContexts");
   fProofs      = new TList; fProofs->SetName("Proofs");
   fClipboard   = new TList; fClipboard->SetName("Clipboard");
   fDataSets    = new TList; fDataSets->SetName("DataSets");

   TProcessID::AddProcessID();
   fUUIDs = new TProcessUUID();

   fRootFolder = new TFolder();
   fRootFolder->SetName("root");
   fRootFolder->SetTitle("root of all folders");
   fRootFolder->AddFolder("Classes",   "List of Active Classes",fClasses);
   fRootFolder->AddFolder("Colors",    "List of Active Colors",fColors);
   fRootFolder->AddFolder("MapFiles",  "List of MapFiles",fMappedFiles);
   fRootFolder->AddFolder("Sockets",   "List of Socket Connections",fSockets);
   fRootFolder->AddFolder("Canvases",  "List of Canvases",fCanvases);
   fRootFolder->AddFolder("Styles",    "List of Styles",fStyles);
   fRootFolder->AddFolder("Functions", "List of Functions",fFunctions);
   fRootFolder->AddFolder("Tasks",     "List of Tasks",fTasks);
   fRootFolder->AddFolder("Geometries","List of Geometries",fGeometries);
   fRootFolder->AddFolder("Browsers",  "List of Browsers",fBrowsers);
   fRootFolder->AddFolder("Specials",  "List of Special Objects",fSpecials);
   fRootFolder->AddFolder("Handlers",  "List of Message Handlers",fMessageHandlers);
   fRootFolder->AddFolder("Cleanups",  "List of RecursiveRemove Collections",fCleanups);
   fRootFolder->AddFolder("StreamerInfo","List of Active StreamerInfo Classes",fStreamerInfo);
   fRootFolder->AddFolder("SecContexts","List of Security Contexts",fSecContexts);
   fRootFolder->AddFolder("PROOF Sessions", "List of PROOF sessions",fProofs);
   fRootFolder->AddFolder("ROOT Memory","List of Objects in the gROOT Directory",fList);
   fRootFolder->AddFolder("ROOT Files","List of Connected ROOT Files",fFiles);

   // by default, add the list of files, tasks, canvases and browsers in the Cleanups list
   fCleanups->Add(fCanvases); fCanvases->SetBit(kMustCleanup);
   fCleanups->Add(fBrowsers); fBrowsers->SetBit(kMustCleanup);
   fCleanups->Add(fTasks);    fTasks->SetBit(kMustCleanup);
   fCleanups->Add(fFiles);    fFiles->SetBit(kMustCleanup);
   fCleanups->Add(fInterpreter);

   fExecutingMacro= kFALSE;
   fForceStyle    = kFALSE;
   fFromPopUp     = kFALSE;
   fReadingObject = kFALSE;
   fInterrupt     = kFALSE;
   fEscape        = kFALSE;
   fMustClean     = kTRUE;
   fPrimitive     = 0;
   fSelectPad     = 0;
   fEditorMode    = 0;
   fDefCanvasName = "c1";
   fEditHistograms= kFALSE;
   fLineIsProcessing = 1;   // This prevents WIN32 "Windows" thread to pick ROOT objects with mouse
   gDirectory     = this;
   gPad           = 0;

   //set name of graphical cut class for the graphics editor
   //cannot call SetCutClassName at this point because the TClass of TCutG
   //is not yet build
   fCutClassName = "TCutG";

   // Create a default MessageHandler
   new TMessageHandler((TClass*)0);

   // Create some styles
   gStyle = 0;
   TStyle::BuildStyles();
   SetStyle("Default");

   // Setup default (batch) graphics and GUI environment
   gBatchGuiFactory = new TGuiFactory;
   gGuiFactory      = gBatchGuiFactory;
   gGXBatch         = new TVirtualX("Batch", "ROOT Interface to batch graphics");
   gVirtualX        = gGXBatch;

#ifdef R__WIN32
   fBatch = kFALSE;
#else
   if (gSystem->Getenv("DISPLAY"))
      fBatch = kFALSE;
   else
      fBatch = kTRUE;
#endif

   int i = 0;
   while (initfunc && initfunc[i]) {
      (initfunc[i])();
      fBatch = kFALSE;  // put system in graphics mode (backward compatible)
      i++;
   }

   // Load and init threads library
   InitThreads();

   // Load RQ_OBJECT.h in interpreter (allows signal/slot programming, like Qt)
   TQObject::LoadRQ_OBJECT();

   // Set initial/default list of browsable objects
   fBrowsables->Add(fRootFolder, "root");
   fBrowsables->Add(fProofs, "PROOF Sessions");
   fBrowsables->Add(workdir, gSystem->WorkingDirectory());
   fBrowsables->Add(fFiles, "ROOT Files");

   atexit(CleanUpROOTAtExit);

   fgRootInit = kTRUE;

   TClass::ReadRules(); // Read the default customization rules ...
}

//______________________________________________________________________________
TROOT::~TROOT()
{
   // Clean up and free resources used by ROOT (files, network sockets,
   // shared memory segments, etc.).

   if (gROOT == this) {

      // Mark the object are invalid, so that we can vero some actions
      // (like autoloading) while we are in the destructor.
      SetBit(TObject::kInvalidObject);
      
      // Turn-off the global mutex to avoid recreating mutexes that have
      // already been deleted during the destruction phase
      gGlobalMutex = 0;

      // Return when error occured in TCint, i.e. when setup file(s) are
      // out of date
      if (!fVersionInt) return;

      // ATTENTION!!! Order is important!

#ifdef R__COMPLETE_MEM_TERMINATION
      SafeDelete(fBrowsables);
      SafeDelete(fRootFolder);
      fSpecials->Delete();   SafeDelete(fSpecials);    // delete special objects : PostScript, Minuit, Html
#endif
      fFiles->Delete("slow"); SafeDelete(fFiles);       // and files
      fSecContexts->Delete("slow"); SafeDelete(fSecContexts); // and security contexts
      fSockets->Delete();     SafeDelete(fSockets);     // and sockets
      fMappedFiles->Delete("slow");                     // and mapped files
      delete fUUIDs;
      TProcessID::Cleanup();                            // and list of ProcessIDs
      TSeqCollection *tl = fMappedFiles; fMappedFiles = 0; delete tl;

      fFunctions->Delete();  SafeDelete(fFunctions);   // etc..
      fColors->Delete();     SafeDelete(fColors);
      fStyles->Delete();     SafeDelete(fStyles);
      fGeometries->Delete(); SafeDelete(fGeometries);
      fBrowsers->Delete();   SafeDelete(fBrowsers);
      
#ifdef R__COMPLETE_MEM_TERMINATION
      if (gGuiFactory != gBatchGuiFactory) SafeDelete(gGuiFactory);
      SafeDelete(gBatchGuiFactory);
      if (gGXBatch != gVirtualX) SafeDelete(gGXBatch);
      SafeDelete(gVirtualX);
#endif
      
      // Stop emitting signals
      TQObject::BlockAllSignals(kTRUE);

      fMessageHandlers->Delete(); SafeDelete(fMessageHandlers);

#ifdef R__COMPLETE_MEM_TERMINATION
      SafeDelete(fCanvases);
      SafeDelete(fTasks);
      SafeDelete(fProofs);
      SafeDelete(fDataSets);
      SafeDelete(fClipboard);
      fCleanups->Clear();
      delete fPluginManager; gPluginMgr = fPluginManager = 0;
      delete gClassTable;  gClassTable = 0;
      delete gEnv; gEnv = 0;

      if (fTypes) fTypes->Delete();
      SafeDelete(fTypes);
      if (fGlobals) fGlobals->Delete();
      SafeDelete(fGlobals);
      if (fGlobalFunctions) fGlobalFunctions->Delete();
      SafeDelete(fGlobalFunctions);
      fClasses->Delete();    SafeDelete(fClasses);     // TClass'es must be deleted last
#endif

      // Remove shared libraries produced by the TSystem::CompileMacro() call
      gSystem->CleanCompiledMacros();

      // Cleanup system class
      delete gSystem;

      // Problem deleting the interpreter. Want's to delete objects already
      // deleted in the dtor's above. Crash.
      // It should only close the files and NOT delete.
      SafeDelete(fInterpreter);

#ifdef R__COMPLETE_MEM_TERMINATION
      SafeDelete(fCleanups);
#endif

      // Prints memory stats
      TStorage::PrintStatistics();

      gROOT = 0;
      fgRootInit = kFALSE;
   }
}

//______________________________________________________________________________
void TROOT::AddClass(TClass *cl)
{
   // Add a class to the list and map of classes.

   //if (!cl) return;
   //GetListOfClasses()->Add(cl);
   //if (cl->GetTypeInfo()) {
   //   fIdMap->Add(cl->GetTypeInfo()->name(),cl);
   //}
   TClass::AddClass(cl);
}

//______________________________________________________________________________
void TROOT::AddClassGenerator(TClassGenerator *generator)
{
   // Add a class generator.  This generator will be called by TClass::GetClass
   // in case its does not find a loaded rootcint dictionary to request the
   // creation of a TClass object.

   if (!generator) return;
   fClassGenerators->Add(generator);
}

//______________________________________________________________________________
void TROOT::Browse(TBrowser *b)
{
   // Add browsable objects to TBrowser.

   TObject *obj;
   TIter next(fBrowsables);

   while ((obj = (TObject *) next())) {
      const char *opt = next.GetOption();
      if (opt && strlen(opt))
         b->Add(obj, opt);
      else
         b->Add(obj, obj->GetName());
   }
}

//______________________________________________________________________________
Bool_t TROOT::ClassSaved(TClass *cl)
{
// return class status bit kClassSaved for class cl
// This function is called by the SavePrimitive functions writing
// the C++ code for an object.

   if (cl == 0) return kFALSE;
   if (cl->TestBit(TClass::kClassSaved)) return kTRUE;
   cl->SetBit(TClass::kClassSaved);
   return kFALSE;
}

//______________________________________________________________________________
void TROOT::CloseFiles()
{
   // Close any files and sockets that gROOT knows about.
   // Delete the corresponding TFile and TSockets objects.
   // This can be used to insures that the files and sockets are closed before any library is unloaded!

   if (fFiles) fFiles->Delete("slow");
   if (fSockets) fSockets->Delete();
   if (fMappedFiles) fMappedFiles->Delete("slow");
}

//______________________________________________________________________________
TObject *TROOT::FindObject(const TObject *) const
{
// Find an object in one Root folder

   Error("FindObject","Not yet implemented");
   return 0;
}

//______________________________________________________________________________
TObject *TROOT::FindObject(const char *name) const
{
   // Returns address of a ROOT object if it exists
   //
   // If name contains at least one "/" the function calls FindObjectany
   // else
   // This function looks in the following order in the ROOT lists:
   //     - List of files
   //     - List of memory mapped files
   //     - List of functions
   //     - List of geometries
   //     - List of canvases
   //     - List of styles
   //     - List of specials
   //     - List of materials in current geometry
   //     - List of shapes in current geometry
   //     - List of matrices in current geometry
   //     - List of Nodes in current geometry
   //     - Current Directory in memory
   //     - Current Directory on file

   if (name && strstr(name,"/")) return FindObjectAny(name);

   TObject *temp = 0;

   temp   = fFiles->FindObject(name);       if (temp) return temp;
   temp   = fMappedFiles->FindObject(name); if (temp) return temp;
   temp   = fFunctions->FindObject(name);   if (temp) return temp;
   temp   = fGeometries->FindObject(name);  if (temp) return temp;
   temp   = fCanvases->FindObject(name);    if (temp) return temp;
   temp   = fStyles->FindObject(name);      if (temp) return temp;
   temp   = fSpecials->FindObject(name);    if (temp) return temp;
   TIter next(fGeometries);
   TObject *obj;
   while ((obj=next())) {
      temp = obj->FindObject(name);         if (temp) return temp;
   }
   if (gDirectory) temp = gDirectory->Get(name); if (temp) return temp;
   if (gPad) {
      TVirtualPad *canvas = gPad->GetVirtCanvas();
      if (fCanvases->FindObject(canvas)) {  //this check in case call from TCanvas ctor
         temp = canvas->FindObject(name);
         if (!temp && canvas != gPad) temp  = gPad->FindObject(name);
      }
   }
   return temp;
}

//______________________________________________________________________________
TObject *TROOT::FindSpecialObject(const char *name, void *&where)
{
   // Returns address and folder of a ROOT object if it exists
   //
   // This function looks in the following order in the ROOT lists:
   //     - List of files
   //     - List of memory mapped files
   //     - List of functions
   //     - List of geometries
   //     - List of canvases
   //     - List of styles
   //     - List of specials
   //     - List of materials in current geometry
   //     - List of shapes in current geometry
   //     - List of matrices in current geometry
   //     - List of Nodes in current geometry
   //     - Current Directory in memory
   //     - Current Directory on file

   TObject *temp = 0;
   where = 0;

   if (!temp && !strcmp(name, "gPad")) {
      temp = gPad;
      if (gPad) {
         TVirtualPad *canvas = gPad->GetVirtCanvas();
         //this check in case call from TCanvas ctor
         if (fCanvases->FindObject(canvas))
            where = canvas;
      }
   }
   if (!temp && !strcmp(name, "gVirtualX")) {
      return gVirtualX;
   }
   if (!temp && !strcmp(name, "gInterpreter")) {
      return gInterpreter;
   }
   if (!temp) {
      temp  = fFiles->FindObject(name);
      where = fFiles;
   }
   if (!temp) {
      temp  = fMappedFiles->FindObject(name);
      where = fMappedFiles;
   }
   if (!temp) {
      temp  = fFunctions->FindObject(name);
      where = fFunctions;
   }
   if (!temp) {
      temp  = fCanvases->FindObject(name);
      where = fCanvases;
   }
   if (!temp) {
      temp  = fStyles->FindObject(name);
      where = fStyles;
   }
   if (!temp) {
      temp  = fSpecials->FindObject(name);
      where = fSpecials;
   }
   if (!temp) {
      TObject *glast = fGeometries->Last();
      if (glast) {where = glast; temp = glast->FindObject(name);}
   }
   if (!temp && gDirectory) {
      temp  = gDirectory->Get(name);
      where = gDirectory;
   }
   if (!temp && gPad) {
      TVirtualPad *canvas = gPad->GetVirtCanvas();
      if (fCanvases->FindObject(canvas)) {  //this check in case call from TCanvas ctor
         temp  = canvas->FindObject(name);
         where = canvas;
         if (!temp && canvas != gPad) {
            temp  = gPad->FindObject(name);
            where = gPad;
         }
      }
   }
   if (!temp) return 0;
   if (temp->TestBit(kNotDeleted)) return temp;
   return 0;
}

//______________________________________________________________________________
TObject *TROOT::FindObjectAny(const char *name) const
{
   // Return a pointer to the first object with name starting at //root.
   // This function scans the list of all folders.
   // if no object found in folders, it scans the memory list of all files.

   TObject *obj = fRootFolder->FindObjectAny(name);
   if (obj) return obj;
   return gDirectory->FindObjectAnyFile(name);
}

//______________________________________________________________________________
const char *TROOT::FindObjectClassName(const char *name) const
{
   // Returns class name of a ROOT object including CINT globals.

   // Search first in the list of "standard" objects
   TObject *obj = FindObject(name);
   if (obj) return obj->ClassName();

   // Is it a global variable?
   TGlobal *g = GetGlobal(name);
   if (g) return g->GetTypeName();

   return 0;
}

//______________________________________________________________________________
const char *TROOT::FindObjectPathName(const TObject *) const
{
   // Return path name of obj somewhere in the //root/... path.
   // The function returns the first occurence of the object in the list
   // of folders. The returned string points to a static char array in TROOT.
   // If this function is called in a loop or recursively, it is the
   // user's responsability to copy this string in his area.

   Error("FindObjectPathName","Not yet implemented");
   return "??";
}

//______________________________________________________________________________
static TClass *R__FindSTLClass(const char *name, Bool_t load, Bool_t silent, const char *outername)
{
   // return a TClass object corresponding to 'name' assuming it is an STL container.
   // In particular we looking for possible alternative name (default template
   // parameter, typedefs template arguments, typedefed name).

   TClass *cl = 0;

   // We have not found the STL container yet.
   // First we are going to look for a similar name but different 'default' template
   // parameter (differences due to different STL implementation)

   string defaultname( TClassEdit::ShortType( name, TClassEdit::kDropStlDefault ) ) ;

   if (defaultname != name) {
      cl = (TClass*)gROOT->GetListOfClasses()->FindObject(defaultname.c_str());
      if (load && !cl) cl = gROOT->LoadClass(defaultname.c_str(), silent);
   }

   if (cl==0) {

      // now look for a typedef
      // well for now the typedefing in CINT has some issues
      // for examples if we generated the dictionary for
      //    set<string,someclass> then set<string> is typedef to it (instead of set<string,less<string> >)

      TDataType *objType = gROOT->GetType(name, load);
      if (objType) {
         const char *typedfName = objType->GetTypeName();
         if (typedfName) {
            string defaultTypedefName(TClassEdit::ShortType(typedfName, TClassEdit::kDropStlDefault));

            if (strcmp(typedfName, name) && defaultTypedefName == name) {
               cl = (TClass*)gROOT->GetListOfClasses()->FindObject(typedfName);
               if (load && !cl) cl = gROOT->LoadClass(typedfName, silent);
            }
         }
      }
   }
   if (cl==0) {
      // Try the alternate name where all the typedefs are resolved:

      const char *altname = gInterpreter->GetInterpreterTypeName(name);
      if (altname && strcmp(altname,name)!=0 && strcmp(altname,outername)!=0) {
         cl = TClass::GetClass(altname,load,silent);
      }
   }
   if (cl==0) {
      // Try with Long64_t instead of long long
      string long64name = TClassEdit::GetLong64_Name( name );
      if ( long64name != name && long64name != outername ) return R__FindSTLClass( long64name.c_str(), load, silent, outername);
   }
   if (cl == 0) {
      TString resolvedName = TClassEdit::ResolveTypedef(name,kFALSE).c_str();
      if (resolvedName != name && resolvedName != outername) cl = TClass::GetClass(resolvedName,load,silent);
   }
   if (cl == 0 && (strncmp(name,"std::",5)==0)) {
      // CINT sometime ignores the std namespace for stl containers,
      // so let's try without it.
      if (strlen(name+5)) cl = TClass::GetClass(name+5,load,silent);
   }

   if (load && cl==0) {
      // Create an Emulated class for this container.
      cl = new TClass(defaultname.c_str(), TClass::GetClass("TVirtualStreamerInfo")->GetClassVersion(), 0, 0, -1, -1, silent );
      cl->SetBit(TClass::kIsEmulation);
   }

   return cl;
}

//______________________________________________________________________________
TClass *TROOT::FindSTLClass(const char *name, Bool_t load, Bool_t silent) const
{
   // return a TClass object corresponding to 'name' assuming it is an STL container.
   // In particular we looking for possible alternative name (default template
   // parameter, typedefs template arguments, typedefed name).

   return R__FindSTLClass(name,load,silent,name);
}

//______________________________________________________________________________
TClass *TROOT::GetClass(const char *name, Bool_t load, Bool_t silent) const
{
   // Return pointer to class with name. Obsolete, use TClass::GetClass directly

   return TClass::GetClass(name,load,silent);
}


//______________________________________________________________________________
TClass *TROOT::GetClass(const type_info& typeinfo, Bool_t load, Bool_t silent) const
{
   // Return pointer to class from its name. Obsolete, use TClass::GetClass directly
   // See TClass::GetClass

   return TClass::GetClass(typeinfo,load,silent);
}

//______________________________________________________________________________
TColor *TROOT::GetColor(Int_t color) const
{
   // Return address of color with index color.

   TColor::InitializeColors();
   TObjArray *lcolors = (TObjArray*) GetListOfColors();
   if (color < 0 || color >= lcolors->GetSize()) return 0;
   TColor *col = (TColor*)lcolors->At(color);
   if (col && col->GetNumber() == color) return col;
   TIter   next(lcolors);
   while ((col = (TColor *) next()))
      if (col->GetNumber() == color) return col;

   return 0;
}

//______________________________________________________________________________
TCanvas *TROOT::MakeDefCanvas() const
{
   // Return a default canvas.

   return (TCanvas*)gROOT->ProcessLine("TCanvas::MakeDefCanvas();");
}

//______________________________________________________________________________
TDataType *TROOT::GetType(const char *name, Bool_t load) const
{
   // Return pointer to type with name.

   // First try without loading.  We can do that because nothing is
   // ever removed from the list of types. (See TCint::UpdateListOfTypes).
   TDataType* type = (TDataType*)gROOT->GetListOfTypes(kFALSE)->FindObject(name);
   if (type || !load)
      return type;
   else
      return (TDataType*)gROOT->GetListOfTypes(load)->FindObject(name);
}

//______________________________________________________________________________
TFile *TROOT::GetFile(const char *name) const
{
   // Return pointer to file with name.

   return (TFile*)GetListOfFiles()->FindObject(name);
}

//______________________________________________________________________________
TStyle *TROOT::GetStyle(const char *name) const
{
   // Return pointer to style with name

   return (TStyle*)GetListOfStyles()->FindObject(name);
}

//______________________________________________________________________________
TObject *TROOT::GetFunction(const char *name) const
{
   // Return pointer to function with name.

   if (name == 0 || name[0] == 0) {
      return 0;
   }

   TObject *f1 = fFunctions->FindObject(name);
   if (f1) return f1;

   gROOT->ProcessLine("TF1::InitStandardFunctions();");

   return fFunctions->FindObject(name);
}

//______________________________________________________________________________
TGlobal *TROOT::GetGlobal(const char *name, Bool_t load) const
{
   // Return pointer to global variable by name. If load is true force
   // reading of all currently defined globals from CINT (more expensive).

   return (TGlobal *)gROOT->GetListOfGlobals(load)->FindObject(name);
}

//______________________________________________________________________________
TGlobal *TROOT::GetGlobal(const TObject *addr, Bool_t load) const
{
   // Return pointer to global variable with address addr. If load is true
   // force reading of all currently defined globals from CINT (more
   // expensive).

   TIter next(gROOT->GetListOfGlobals(load));

   TGlobal *g;
   while ((g = (TGlobal*) next())) {
      const char *t = g->GetFullTypeName();
      if (!strncmp(t, "class", 5) || !strncmp(t, "struct", 6)) {
         int ptr = 0;
         if (t[strlen(t)-1] == '*') ptr = 1;
         if (ptr) {
            if (*(Long_t *)g->GetAddress() == (Long_t)addr) return g;
         } else {
            if ((Long_t)g->GetAddress() == (Long_t)addr) return g;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
TFunction *TROOT::GetGlobalFunction(const char *function, const char *params,
                                    Bool_t load)
{
   // Return pointer to global function by name. If params != 0
   // it will also resolve overloading. If load is true force reading
   // of all currently defined global functions from CINT (more expensive).
   // The param string must be of the form: "3189,\"aap\",1.3".

   if (!params)
      return (TFunction *)GetListOfGlobalFunctions(load)->FindObject(function);
   else {
      if (!fInterpreter)
         Fatal("GetGlobalFunction", "fInterpreter not initialized");

      TFunction *f;
      TIter      next(GetListOfGlobalFunctions(load));

      TString mangled = gInterpreter->GetMangledName(0, function, params);
      while ((f = (TFunction *) next())) {
         if (mangled == f->GetMangledName()) return f;
      }

      return 0;
   }
}

//______________________________________________________________________________
TFunction *TROOT::GetGlobalFunctionWithPrototype(const char *function,
                                               const char *proto, Bool_t load)
{
   // Return pointer to global function by name. If proto != 0
   // it will also resolve overloading. If load is true force reading
   // of all currently defined global functions from CINT (more expensive).
   // The proto string must be of the form: "int, char*, float".

   if (!proto)
      return (TFunction *)GetListOfGlobalFunctions(load)->FindObject(function);
   else {
      if (!fInterpreter)
         Fatal("GetGlobalFunctionWithPrototype", "fInterpreter not initialized");

      TFunction *f;
      TIter      next(GetListOfGlobalFunctions(load));

      TString mangled = gInterpreter->GetMangledNameWithPrototype(0,
                                                                     function,
                                                                     proto);
      while ((f = (TFunction *) next())) {
         if (mangled == f->GetMangledName()) return f;
      }
      return 0;
   }
}

//______________________________________________________________________________
TObject *TROOT::GetGeometry(const char *name) const
{
   // Return pointer to Geometry with name

   return GetListOfGeometries()->FindObject(name);
}

//______________________________________________________________________________
TCollection *TROOT::GetListOfGlobals(Bool_t load)
{
   // Return list containing the TGlobals currently defined.
   // Since globals are created and deleted during execution of the
   // program, we need to update the list of globals every time we
   // execute this method. However, when calling this function in
   // a (tight) loop where no interpreter symbols will be created
   // you can set load=kFALSE (default).

   if (!fGlobals) {
      fGlobals = new THashTable(100, 3);
      load = kTRUE;
   }

   if (!fInterpreter)
      Fatal("GetListOfGlobals", "fInterpreter not initialized");

   if (load)
      gInterpreter->UpdateListOfGlobals();

   return fGlobals;
}

//______________________________________________________________________________
TCollection *TROOT::GetListOfGlobalFunctions(Bool_t load)
{
   // Return list containing the TFunctions currently defined.
   // Since functions are created and deleted during execution of the
   // program, we need to update the list of functions every time we
   // execute this method. However, when calling this function in
   // a (tight) loop where no interpreter symbols will be created
   // you can set load=kFALSE (default).

   if (!fGlobalFunctions) {
      fGlobalFunctions = new THashTable(100, 3);
      load = kTRUE;
   }

   if (!fInterpreter)
      Fatal("GetListOfGlobalFunctions", "fInterpreter not initialized");

   if (load)
      gInterpreter->UpdateListOfGlobalFunctions();

   return fGlobalFunctions;
}

//______________________________________________________________________________
TCollection *TROOT::GetListOfTypes(Bool_t load)
{
   // Return list containing all TDataTypes (typedefs) currently defined.
   // Since types can be added and removed during execution of the
   // program, we need to update the list of types every time we
   // execute this method. However, when calling this function in
   // a (tight) loop where no new types will be created
   // you can set load=kFALSE (default).

   if (!fTypes) {
      fTypes = new THashTable(100, 3);
      load = kTRUE;

      // Add also basic types (like a identity typedef "typedef int int")
      fTypes->Add(new TDataType("char"));
      fTypes->Add(new TDataType("unsigned char"));
      fTypes->Add(new TDataType("short"));
      fTypes->Add(new TDataType("unsigned short"));
      fTypes->Add(new TDataType("int"));
      fTypes->Add(new TDataType("unsigned int"));
      fTypes->Add(new TDataType("unsigned"));
      fTypes->Add(new TDataType("long"));
      fTypes->Add(new TDataType("unsigned long"));
      fTypes->Add(new TDataType("long long"));
      fTypes->Add(new TDataType("unsigned long long"));
      fTypes->Add(new TDataType("float"));
      fTypes->Add(new TDataType("double"));
      fTypes->Add(new TDataType("void"));
      fTypes->Add(new TDataType("bool"));
      fTypes->Add(new TDataType("char*"));
   }

   if (!fInterpreter)
      Fatal("GetListOfTypes", "fInterpreter not initialized");

   if (load) {
///      printf("calling Update ListOfTypes\n");
      gInterpreter->UpdateListOfTypes();
///      printf("after calling Update ListOfTypes\n");
   }

   return fTypes;
}


//______________________________________________________________________________
void TROOT::Idle(UInt_t idleTimeInSec, const char *command)
{
   // Execute command when system has been idle for idleTimeInSec seconds.

   if (!fApplication)
      TApplication::CreateApplication();

   if (idleTimeInSec <= 0)
      fApplication->RemoveIdleTimer();
   else
      fApplication->SetIdleTimer(idleTimeInSec, command);
}

//______________________________________________________________________________
Int_t TROOT::IgnoreInclude(const char *fname, const char * /*expandedfname*/)
{
   // Return 1 if the given include file correspond to a class that has
   // been loaded through a compiled dictionnary.

   if (fname == 0) return 0;

   TString stem(fname);
   // Remove extension if any, ignore files with extension not being .h*
   Int_t where = stem.Last('.');
   if (where != kNPOS) {
      if (stem.EndsWith(".so") || stem.EndsWith(".sl") ||
          stem.EndsWith(".dl") || stem.EndsWith(".a")  ||
          stem.EndsWith(".dll", TString::kIgnoreCase))
         return 0;
      stem.Remove(where);
   }

   TString className = gSystem->BaseName(stem);
   TClass *cla = TClass::GetClass(className);

   if (!cla) {
      className = stem;
      className.ReplaceAll("/", "::");
      className.ReplaceAll("\\", "::");
      if (className.Contains(":::")) {
         // "C:\dir" becomes "C:::dir".
         // fname corresponds to whatever is stated after #include and
         // a full path name usually means that it's not a regular #include
         // but e.g. a ".L", so we can assume that this is not a header of
         // a class in a namespace (a global-namespace class would have been
         // detected already before).
         return 0;
      }
      cla = TClass::GetClass(className);
   }
   if ( cla ) {
      if (cla->GetDeclFileLine() <= 0) return 0; // to a void an error with VisualC++
      TString decfile = gSystem->BaseName(cla->GetDeclFileName());
      if (decfile == gSystem->BaseName(fname)) {
         return 1;
      }
   }
   return 0;
}

//______________________________________________________________________________
void TROOT::InitSystem()
{
   // Initialize operating system interface.

   if (gSystem == 0) {
#if defined(R__UNIX)
      gSystem = new TUnixSystem;
#elif defined(R__WIN32)
      gSystem = new TWinNTSystem;
#else
      gSystem = new TSystem;
#endif

      if (gSystem->Init())
         fprintf(stderr, "Fatal in <TROOT::InitSystem>: can't init operating system layer\n");

      if (!gSystem->HomeDirectory())
         fprintf(stderr, "Fatal in <TROOT::InitSystem>: HOME directory not set\n");

      // read default files
      gEnv = new TEnv(".rootrc");

      gDebug = gEnv->GetValue("Root.Debug", 0);

      //By default the zipmode is 1 (see Bits.h)
      Int_t zipmode = gEnv->GetValue("Root.ZipMode",1);
      if (zipmode !=1) R__SetZipMode(zipmode);

      const char *sdeb;
      if ((sdeb = gSystem->Getenv("ROOTDEBUG")))
         gDebug = atoi(sdeb);

      if (gDebug > 0 && isatty(2))
         fprintf(stderr, "Info in <TROOT::InitSystem>: running with gDebug = %d\n", gDebug);

      if (gEnv->GetValue("Root.MemStat", 0))
         TStorage::EnableStatistics();
      int msize = gEnv->GetValue("Root.MemStat.size", -1);
      int mcnt  = gEnv->GetValue("Root.MemStat.cnt", -1);
      if (msize != -1 || mcnt != -1)
         TStorage::EnableStatistics(msize, mcnt);

      fgMemCheck = gEnv->GetValue("Root.MemCheck", 0);

      TObject::SetObjectStat(gEnv->GetValue("Root.ObjectStat", 0));

   }
}

//______________________________________________________________________________
void TROOT::InitThreads()
{
   // Load and initialize thread library.

   if (gEnv->GetValue("Root.UseThreads", 0)) {
      char *path;
      if ((path = gSystem->DynamicPathName("libThread", kTRUE))) {
         delete [] path;
         LoadClass("TThread", "Thread");
      }
   }
}

//______________________________________________________________________________
TClass *TROOT::LoadClass(const char *requestedname, Bool_t silent) const
{
   // Helper function used by TClass::GetClass().
   // This function attempts to load the dictionary for 'classname'
   // either from the TClassTable or from the list of generator.
   // If silent is 'true', do not warn about missing dictionary for the class.
   // (typically used for class that are used only for transient members)

   // This function does not (and should not) attempt to check in the
   // list of loaded classes or in the typedef.


   // We need to cache the requested name as in some case this function is
   // called with gROOT->LoadClass(cl->GetName()) and the loading of a library,
   // for example via the autoloader, can result in our argument becoming invalid.
   // In addition the call to the dictionary function (dict()) might also have
   // the same effect (change/delete requestedname).
   TString classname(requestedname);

   VoidFuncPtr_t dict = TClassTable::GetDict(classname);

   TString resolved;

   if (!dict) {
      // Try to remove the ROOT typedefs
      resolved = TClassEdit::ResolveTypedef(classname,kTRUE);
      if (resolved != classname) {
         dict = TClassTable::GetDict(resolved.Data());
      } else {
         resolved.Clear();
      }
   }
   if (!dict) {
      if (gInterpreter->AutoLoad(classname)) {
         dict = TClassTable::GetDict(classname);
         if (!dict) {
            // Try the typedefs again.
            if (resolved.Length()) {
               dict = TClassTable::GetDict(resolved.Data());
            }
         }
      }
   }

   if (dict) {
      (dict)();
      TClass *ncl = TClass::GetClass(classname, kFALSE, silent);
      if (ncl) ncl->PostLoadCheck();
      return ncl;
   }

   TIter next(fClassGenerators);
   TClassGenerator *gen;
   while ((gen = (TClassGenerator*) next())) {
      TClass *cl = gen->GetClass(classname, kTRUE, silent);
      if (cl) {
         cl->PostLoadCheck();
         return cl;
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TROOT::LoadClass(const char * /*classname*/, const char *libname,
                       Bool_t check)
{
   // Check if class "classname" is known to the interpreter (in fact,
   // this check is not needed anymore, so classname is ignored). If
   // not it will load library "libname". If the library name does
   // not start with "lib", "lib" will be prepended and a search will
   // be made in the DynamicPath (see .rootrc). If not found a search
   // will be made on libname (without "lib" prepended) and if not found
   // a direct try of libname will be made (in case it contained an
   // absolute path).
   // If check is true it will only check if libname exists and is
   // readable.
   // Returns 0 on successful loading, -1 in case libname does not
   // exist or in case of error and -2 in case of version mismatch.

   Int_t err = -1;

   char *path;
   TString lib = libname;
   if (!lib.BeginsWith("lib"))
      lib = "lib" + lib;
   if ((path = gSystem->DynamicPathName(lib, kTRUE))) {
      if (check)
         err = 0;
      else {
         err = gSystem->Load(path, 0, kTRUE);
      }
      delete [] path;
   } else {
      if (check) {
         FileStat_t stat;
         if (!gSystem->GetPathInfo(libname, stat)) {
            if (R_ISREG(stat.fMode) &&
                !gSystem->AccessPathName(libname, kReadPermission))
               err = 0;
            else
               err = -1;
         } else
            err = -1;
      } else {
         err = gSystem->Load(libname, 0, kTRUE);
      }
   }

   if (err == 0 && !check) {
      GetListOfTypes(kTRUE);
   }

   if (err == -1) {
      //Error("LoadClass", "library %s could not be loaded", libname);
   }

   if (err == 1) {
      //Error("LoadClass", "library %s already loaded, but class %s unknown",
      //      libname, classname);
      err = 0;
   }

   return err;
}

//______________________________________________________________________________
void TROOT::ls(Option_t *option) const
{
   // To list all objects of the application.
   // Loop on all objects created in the ROOT linked lists.
   // Objects may be files and windows or any other object directly
   // attached to the ROOT linked list.

//   TObject::SetDirLevel();
//   GetList()->R__FOR_EACH(TObject,ls)(option);
   TDirectory::ls(option);
}

//______________________________________________________________________________
Int_t TROOT::LoadMacro(const char *filename, int *error, Bool_t check)
{
   // Load a macro in the interpreter's memory. Equivalent to the command line
   // command ".L filename". If the filename has "+" or "++" appended
   // the macro will be compiled by ACLiC. The filename must have the format:
   // [path/]macro.C[+|++[g|O]].
   // The possible error codes are defined by TInterpreter::EErrorCode.
   // If check is true it will only check if filename exists and is
   // readable.
   // Returns 0 on successful loading and -1 in case filename does not
   // exist or in case of error.

   Int_t err = -1;
   Int_t lerr, *terr;
   if (error)
      terr = error;
   else
      terr = &lerr;

   if (fInterpreter) {
      TString aclicMode;
      TString arguments;
      TString io;
      TString fname = gSystem->SplitAclicMode(filename, aclicMode, arguments, io);

      if (arguments.Length()) {
         Warning("LoadMacro", "argument(%s) ignored in %s", arguments.Data(), GetMacroPath());
      }
      char *mac = gSystem->Which(GetMacroPath(), fname, kReadPermission);
      if (!mac) {
         if (!check)
            Error("LoadMacro", "macro %s not found in path %s", fname.Data(), GetMacroPath());
         *terr = TInterpreter::kFatal;
      } else {
         err = 0;
         if (!check) {
            fname = mac;
            fname += aclicMode;
            fname += io;
            gInterpreter->LoadMacro(fname.Data(), (TInterpreter::EErrorCode*)terr);
            if (*terr)
               err = -1;
            //else   // maybe not needed (RDM)
            //   GetListOfTypes(kTRUE);
         }
      }
      delete [] mac;
   }
   return err;
}

//______________________________________________________________________________
Long_t TROOT::Macro(const char *filename, Int_t *error, Bool_t padUpdate)
{
   // Execute a macro in the interpreter. Equivalent to the command line
   // command ".x filename". If the filename has "+" or "++" appended
   // the macro will be compiled by ACLiC. The filename must have the format:
   // [path/]macro.C[+|++[g|O]][(args)].
   // The possible error codes are defined by TInterpreter::EErrorCode.
   // If padUpdate is true (default) update the current pad.
   // Returns the macro return value.

   Long_t result = 0;

   if (fInterpreter) {
      TString aclicMode;
      TString arguments;
      TString io;
      TString fname = gSystem->SplitAclicMode(filename, aclicMode, arguments, io);

      char *mac = gSystem->Which(GetMacroPath(), fname, kReadPermission);
      if (!mac) {
         Error("Macro", "macro %s not found in path %s", fname.Data(), GetMacroPath());
         if (error)
            *error = TInterpreter::kFatal;
      } else {
         fname = mac;
         fname += aclicMode;
         fname += arguments;
         fname += io;
         result = gInterpreter->ExecuteMacro(fname, (TInterpreter::EErrorCode*)error);
      }
      delete [] mac;

      if (padUpdate && gPad)
         gPad->Update();
   }

   return result;
}

//______________________________________________________________________________
void  TROOT::Message(Int_t id, const TObject *obj)
{
   // Process message id called by obj.

   TIter next(fMessageHandlers);
   TMessageHandler *mh;
   while ((mh = (TMessageHandler*)next())) {
      mh->HandleMessage(id,obj);
   }
}

//______________________________________________________________________________
Long_t TROOT::ProcessLine(const char *line, Int_t *error)
{
   // Process interpreter command via TApplication::ProcessLine().
   // On Win32 the line will be processed asynchronously by sending
   // it to the CINT interpreter thread. For explicit synchronous processing
   // use ProcessLineSync(). On non-Win32 platforms there is no difference
   // between ProcessLine() and ProcessLineSync().
   // The possible error codes are defined by TInterpreter::EErrorCode. In
   // particular, error will equal to TInterpreter::kProcessing until the
   // CINT interpreted thread has finished executing the line.
   // Returns the result of the command, cast to a Long_t.

   TString sline = line;
   sline = sline.Strip(TString::kBoth);

   if (!fApplication)
      TApplication::CreateApplication();

   return fApplication->ProcessLine(sline, kFALSE, error);
}

//______________________________________________________________________________
Long_t TROOT::ProcessLineSync(const char *line, Int_t *error)
{
   // Process interpreter command via TApplication::ProcessLine().
   // On Win32 the line will be processed synchronously (i.e. it will
   // only return when the CINT interpreter thread has finished executing
   // the line). On non-Win32 platforms there is no difference between
   // ProcessLine() and ProcessLineSync().
   // The possible error codes are defined by TInterpreter::EErrorCode.
   // Returns the result of the command, cast to a Long_t.

   TString sline = line;
   sline = sline.Strip(TString::kBoth);

   if (!fApplication)
      TApplication::CreateApplication();

   return fApplication->ProcessLine(sline, kTRUE, error);
}

//______________________________________________________________________________
Long_t TROOT::ProcessLineFast(const char *line, Int_t *error)
{
   // Process interpreter command directly via CINT interpreter.
   // Only executable statements are allowed (no variable declarations),
   // In all other cases use TROOT::ProcessLine().
   // The possible error codes are defined by TInterpreter::EErrorCode.

   TString sline = line;
   sline = sline.Strip(TString::kBoth);

   if (!fApplication)
      TApplication::CreateApplication();

   Long_t result = 0;

   if (fInterpreter) {
      TInterpreter::EErrorCode *code = (TInterpreter::EErrorCode*)error;
      result = gInterpreter->Calc(sline, code);
   }

   return result;
}

//______________________________________________________________________________
void TROOT::ReadSvnInfo()
{
   // Read Subversion revision information and branch name from the
   // etc/svnrev.txt file.

   fSvnRevision = 0;
#ifdef ROOT_SVN_REVISION
   fSvnRevision = ROOT_SVN_REVISION;
#endif
#ifdef ROOT_SVN_BRANCH
   fSvnBranch = ROOT_SVN_BRANCH;
#endif

   TString svninfo = "svninfo.txt";
   char *filename = 0;
#ifdef ROOTETCDIR
   filename = gSystem->ConcatFileName(ROOTETCDIR, svninfo);
#else
   TString etc = gRootDir;
#ifdef WIN32
   etc += "\\etc";
#else
   etc += "/etc";
#endif
#if defined(R__MACOSX) && (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR)
   // on iOS etc does not exist and svninfo resides in $ROOTSYS
   etc = gRootDir;
#endif
   filename = gSystem->ConcatFileName(etc, svninfo);
#endif

   FILE *fp = fopen(filename, "r");
   if (fp) {
      TString s;
      // read branch name
      s.Gets(fp);
      fSvnBranch = s;
      // read revision number
      s.Gets(fp);
      Int_t r = s.Atoi();
      if (r > 0)
         fSvnRevision = r;
      // read date/time make was run
      s.Gets(fp);
      fSvnDate = s;
      fclose(fp);
   }
   delete [] filename;
}

//______________________________________________________________________________
const char *TROOT::GetSvnDate()
{
   // Return date/time make was run.

   if (fSvnDate == "") {
      Int_t iday,imonth,iyear, ihour, imin;
      static const char *months[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                      "Jul", "Aug", "Sep", "Oct", "Nov", "De" };
      Int_t idate = gROOT->GetBuiltDate();
      Int_t itime = gROOT->GetBuiltTime();
      iday   = idate%100;
      imonth = (idate/100)%100;
      iyear  = idate/10000;
      ihour  = itime/100;
      imin   = itime%100;
      fSvnDate.Form("%s %02d %4d, %02d:%02d:00", months[imonth-1], iday, iyear, ihour, imin);
   }
   return fSvnDate;
}

//______________________________________________________________________________
void TROOT::RefreshBrowsers()
{
   // Refresh all browsers. Call this method when some command line
   // command or script has changed the browser contents. Not needed
   // for objects that have the kMustCleanup bit set. Most useful to
   // update browsers that show the file system or other objects external
   // to the running ROOT session.

   TIter next(GetListOfBrowsers());
   TBrowser *b;
   while ((b = (TBrowser*) next()))
      b->SetRefreshFlag(kTRUE);
}

//______________________________________________________________________________
void TROOT::RemoveClass(TClass *oldcl)
{
   // Remove a class from the list and map of classes

   //if (!oldcl) return;
   //GetListOfClasses()->Remove(oldcl);
   //if (oldcl->GetTypeInfo()) {
   //   fIdMap->Remove(oldcl->GetTypeInfo()->name());
   //}
   TClass::RemoveClass(oldcl);
}

//______________________________________________________________________________
void TROOT::Reset(Option_t *option)
{
   // Delete all global interpreter objects created since the last call to Reset
   //
   // If option="a" is set reset to startup context (i.e. unload also
   // all loaded files, classes, structs, typedefs, etc.).
   //
   // This function is typically used at the beginning (or end) of an unnamed macro
   // to clean the environment.
   //
   // IMPORTANT WARNING:
   // Do not use this call from within any function (neither compiled nor
   // interpreted.  This should only be used from a unnamed macro
   // (which starts with a { (curly braces)  ).  For example, using TROOT::Reset
   // from within an interpreted function will lead to the unloading of the
   // dictionary and source file, including the one defining the function being
   // executed.
   //

   if (IsExecutingMacro()) return;  //True when TMacro::Exec runs
   if (fInterpreter) {
      if (!strncmp(option, "a", 1)) {
         fInterpreter->Reset();
         fInterpreter->SaveContext();
      } else
         gInterpreter->ResetGlobals();

      if (fGlobals) fGlobals->Delete();
      if (fGlobalFunctions) fGlobalFunctions->Delete();

      SaveContext();
   }
}

//______________________________________________________________________________
void TROOT::SaveContext()
{
   // Save the current interpreter context.

   if (fInterpreter)
      gInterpreter->SaveGlobalsContext();
}

//______________________________________________________________________________
void TROOT::SetCutClassName(const char *name)
{
   // Set the default graphical cut class name for the graphics editor
   // By default the graphics editor creates an instance of a class TCutG.
   // This function may be called to specify a different class that MUST
   // derive from TCutG

   if (!name) {
      Error("SetCutClassName","Invalid class name");
      return;
   }
   TClass *cl = TClass::GetClass(name);
   if (!cl) {
      Error("SetCutClassName","Unknown class:%s",name);
      return;
   }
   if (!cl->InheritsFrom("TCutG")) {
      Error("SetCutClassName","Class:%s does not derive from TCutG",name);
      return;
   }
   fCutClassName = name;
}

//______________________________________________________________________________
void TROOT::SetEditorMode(const char *mode)
{
   // Set editor mode

   fEditorMode = 0;
   if (strlen(mode) == 0) return;
   if (!strcmp(mode,"Arc"))      {fEditorMode = kArc;        return;}
   if (!strcmp(mode,"Line"))     {fEditorMode = kLine;       return;}
   if (!strcmp(mode,"Arrow"))    {fEditorMode = kArrow;      return;}
   if (!strcmp(mode,"Button"))   {fEditorMode = kButton;     return;}
   if (!strcmp(mode,"Diamond"))  {fEditorMode = kDiamond;    return;}
   if (!strcmp(mode,"Ellipse"))  {fEditorMode = kEllipse;    return;}
   if (!strcmp(mode,"Pad"))      {fEditorMode = kPad;        return;}
   if (!strcmp(mode,"Pave"))     {fEditorMode = kPave;       return;}
   if (!strcmp(mode,"PaveLabel")){fEditorMode = kPaveLabel;  return;}
   if (!strcmp(mode,"PaveText")) {fEditorMode = kPaveText;   return;}
   if (!strcmp(mode,"PavesText")){fEditorMode = kPavesText;  return;}
   if (!strcmp(mode,"PolyLine")) {fEditorMode = kPolyLine;   return;}
   if (!strcmp(mode,"CurlyLine")){fEditorMode = kCurlyLine;  return;}
   if (!strcmp(mode,"CurlyArc")) {fEditorMode = kCurlyArc;   return;}
   if (!strcmp(mode,"Text"))     {fEditorMode = kText;       return;}
   if (!strcmp(mode,"Marker"))   {fEditorMode = kMarker;     return;}
   if (!strcmp(mode,"CutG"))     {fEditorMode = kCutG;       return;}
}

//______________________________________________________________________________
void TROOT::SetStyle(const char *stylename)
{
   // Change current style to style with name stylename

   TStyle *style = GetStyle(stylename);
   if (style) style->cd();
   else       Error("SetStyle","Unknown style:%s",stylename);
}


//-------- Static Member Functions ---------------------------------------------


//______________________________________________________________________________
Int_t TROOT::DecreaseDirLevel()
{
   // Decrease the indentation level for ls().
   return --fgDirLevel;
}

//______________________________________________________________________________
Int_t TROOT::GetDirLevel()
{
   //return directory level
   return fgDirLevel;
}

//______________________________________________________________________________
const char *TROOT::GetMacroPath()
{
   // Get macro search path. Static utility function.

   TString &macroPath = ROOT::GetMacroPath();

   if (macroPath.Length() == 0) {
      macroPath = gEnv->GetValue("Root.MacroPath", (char*)0);
#if defined(R__WIN32)
      macroPath.ReplaceAll("; ", ";");
#else
      macroPath.ReplaceAll(": ", ":");
#endif
      if (macroPath.Length() == 0)
#if !defined(R__WIN32)
   #ifdef ROOTMACRODIR
         macroPath = ".:" ROOTMACRODIR;
   #else
         macroPath = TString(".:") + gRootDir + "/macros";
   #endif
#else
   #ifdef ROOTMACRODIR
         macroPath = ".;" ROOTMACRODIR;
   #else
         macroPath = TString(".;") + gRootDir + "/macros";
   #endif
#endif
   }

   return macroPath;
}

//______________________________________________________________________________
void TROOT::SetMacroPath(const char *newpath)
{
   // Set or extend the macro search path. Static utility function.
   // If newpath=0 or "" reset to value specified in the rootrc file.

   TString &macroPath = ROOT::GetMacroPath();

   if (!newpath || !*newpath)
      macroPath = "";
   else
      macroPath = newpath;
}

//______________________________________________________________________________
Int_t TROOT::IncreaseDirLevel()
{
   // Increase the indentation level for ls().
   return ++fgDirLevel;
}

//______________________________________________________________________________
void TROOT::IndentLevel()
{
   // Functions used by ls() to indent an object hierarchy.

   for (int i = 0; i < fgDirLevel; i++) cout.put(' ');
}

//______________________________________________________________________________
Bool_t TROOT::Initialized()
{
   // Return kTRUE if the TROOT object has been initialized.
   return fgRootInit;
}

//______________________________________________________________________________
Bool_t TROOT::MemCheck()
{
   // Return kTRUE if the memory leak checker is on.
   return fgMemCheck;
}

//______________________________________________________________________________
void TROOT::SetDirLevel(Int_t level)
{
   // Return Indentation level for ls().
   fgDirLevel = level;
}

//______________________________________________________________________________
Int_t TROOT::ConvertVersionCode2Int(Int_t code)
{
   // Convert version code to an integer, i.e. 331527 -> 51507.

   return 10000*(code>>16) + 100*((code&65280)>>8) + (code&255);
}

//______________________________________________________________________________
Int_t TROOT::ConvertVersionInt2Code(Int_t v)
{
   // Convert version as an integer to version code as used in RVersion.h.

   int a = v/10000;
   int b = (v - a*10000)/100;
   int c = v - a*10000 - b*100;
   return (a << 16) + (b << 8) + c;
}

//______________________________________________________________________________
Int_t TROOT::RootVersionCode()
{
   // Return ROOT version code as defined in RVersion.h.

   return ROOT_VERSION_CODE;
}
