// @(#)root/base:$Name:  $:$Id: TROOT.cxx,v 1.7 2000/08/18 13:43:46 brun Exp $
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
//       gROOT->GetListOfCanvases
//       gROOT->GetListOfStyles
//       gROOT->GetListOfFunctions
//       gROOT->GetListOfSpecials (for example graphical cuts)
//       gROOT->GetListOfGeometries
//       gROOT->GetListOfBrowsers
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
//       #include "TROOT.h"
//       #include "TRint.h"
//
//       TROOT root("Rint", "The ROOT Interactive Interface");
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

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <string.h>
#include <iostream.h>

#include "Gtypes.h"
#include "TROOT.h"
#include "TClass.h"
#include "TDataType.h"
#include "TFile.h"
#include "TMapFile.h"
#include "TDatime.h"
#include "TStyle.h"
#include "TObjectTable.h"
#include "TClassTable.h"
#include "TSystem.h"
#include "THashList.h"
#include "TObjArray.h"
#include "TEnv.h"
#include "TColor.h"
#include "TGlobal.h"
#include "TFunction.h"
#include "TVirtualPad.h"
#include "TBrowser.h"
#include "TSystemDirectory.h"
#include "TApplication.h"
#include "TCint.h"
#include "TGuiFactory.h"
#include "TRandom.h"
#include "TMessageHandler.h"
#include "TVirtualGL.h"

#if defined(R__UNIX)
#include "TUnixSystem.h"
#elif defined(R__MAC)
#include "TMacSystem.h"
#elif defined(WIN32)
#include "TWinNTSystem.h"
#elif defined(R__VMS)
#include "TVmsSystem.h"
#endif


//*-*x18.6 macros/layout_root


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
static Int_t IDATQQ()
{
   // Return built date as integer, i.e. "Apr 28 2000" -> 20000428.

   static const char *months[] = {"Jan","Feb","Mar","Apr","May",
                                  "Jun","Jul","Aug","Sep","Oct",
                                  "Nov","Dec"};

   char  sm[12];
   Int_t yy, mm=0, dd;
   sscanf(__DATE__, "%s %d %d", sm, &dd, &yy);
   for (int i = 0; i < 12; i++)
      if (!strncmp(sm, months[i], 3)) {
         mm = i+1;
         break;
      }
   return 10000*yy + 100*mm + dd;
}

//______________________________________________________________________________
static Int_t ITIMQQ()
{
   // Return built time as integer (with min precision), i.e.
   // "17:32:37" -> 1732.

   Int_t hh, mm, ss;
   sscanf(__TIME__, "%d:%d:%d", &hh, &mm, &ss);
   return 100*hh + mm;
}
//------------------------------------------------------------------------------


TROOT      *gROOT;         // The ROOT of EVERYTHING
TRandom    *gRandom;       // Global pointer to random generator

// Global debug flag (set to != 0 to get debug output).
// Can be set either via interpreter (gDebug is exported to CINT),
// or via debugger. If set to > 4 X11 graphics will be in synchronous mode.
Int_t       gDebug;


Bool_t        TROOT::fgRootInit = kFALSE;
VoidFuncPtr_t TROOT::fgMakeDefCanvas = 0;


ClassImp(TROOT)

//______________________________________________________________________________
TROOT::TROOT() : TDirectory()
{
   // Default ctor.
}

//______________________________________________________________________________
TROOT::TROOT(const char *name, const char *title, VoidFuncPtr_t *initfunc)
           : TDirectory()
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
      Warning("TROOT", "only one instance of TROOT allowed");
      return;
   }

   gROOT      = this;
   gDirectory = 0;
   SetName(name);
   SetTitle(title);
   TDirectory::Build();

   // will be used by global "operator delete" so make sure it is set
   // before anything is deleted
   fMappedFiles = 0;

   // Initialize Operating System interface
   InitSystem();

   // Initialize interface to CINT C++ interpreter
   fVersionInt  = 0;  // check in TROOT dtor in case TCint fails
   fClasses     = 0;  // might be checked via TCint ctor
   fInterpreter = new TCint("C/C++", "CINT C/C++ Interpreter");

   // Add the root include directory to list search by default by
   // the interpreter (should this be here or somewhere else?)
#ifndef ROOTINCDIR
   TString include = gSystem->Getenv("ROOTSYS");
   include.Append("/include");
   fInterpreter->AddIncludePath(include);
#else
   fInterpreter->AddIncludePath(ROOTINCDIR);
#endif

   TSystemDirectory *workdir = new TSystemDirectory("workdir",gSystem->WorkingDirectory());

   fVersion     = ROOT_RELEASE;
   fVersionInt  = IVERSQ();
   fVersionDate = IDATQQ();
   fVersionTime = ITIMQQ();
   fApplication = 0;
   fClasses     = new THashList(this,300,2);
   fColors      = new TObjArray(1000);
   fTypes       = 0;
   fGlobals     = 0;
   fGlobalFunctions = 0;
   fFiles       = new TList(this);
   fMappedFiles = new TList(this);
   fSockets     = new TList(this);
   fCanvases    = new TList(this);
   fStyles      = new TList(this);
   fFunctions   = new TList(this);
   fProcesses   = new TList(this);
   fGeometries  = new TList(this);
   fBrowsers    = new TList(this);
   fSpecials    = new TList(this);
   fBrowsables  = new TList(this);
   fMessageHandlers = new TList(this);
   fForceStyle    = kFALSE;
   fFromPopUp     = kFALSE;
   fReadingBasket = kFALSE;
   fInterrupt     = kFALSE;
   fMustClean     = kTRUE;
   fPrimitive     = 0;
   fSelectPad     = 0;
   fEditorMode    = 0;
   fDefCanvasName = "c1";
   fEditHistograms= kFALSE;
   fLineIsProcessing  = 1;   // This prevents WIN32 "Windows" thread to pick ROOT objects with mouse
   gDirectory     = this;
   gPad           = 0;
   gRandom        = new TRandom;

   //set name of graphical cut class for the graphics editor
   //cannot call SetCutClassName at this point because the TClass of TCutG 
   //is not yet build
   fCutClassName = "TCutG";
   
   // Create a default MessageHandler
   new TMessageHandler((TClass*)0);

   // Create some styles
   TStyle::BuildStyles();

   // Setup default (batch) graphics and GUI environment
   gBatchGuiFactory = new TGuiFactory;
   gGuiFactory      = gBatchGuiFactory;
   gGXBatch         = new TVirtualX("Batch", "ROOT Interface to batch graphics");
   gVirtualX        = gGXBatch;
   gVirtualGL       = new TVirtualGL;

#ifdef WIN32
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

   // Set initial/default list of browsable objects
   fBrowsables->Add(fClasses, "Classes");
   fBrowsables->Add(GetListOfGlobals(), "Global Variables");
// fBrowsables->Add(GetListOfGlobalFunctions(), "Global Functions");
   fBrowsables->Add(fCanvases, "Canvases");
   fBrowsables->Add(fGeometries, "Geometries");
   fBrowsables->Add(fColors, "Colors");
   fBrowsables->Add(fStyles, "Styles");
   fBrowsables->Add(fFunctions, "Functions");
   fBrowsables->Add(fSockets, "Network Connections");
   fBrowsables->Add(fMappedFiles, "Memory Mapped Files");
   fBrowsables->Add(workdir, gSystem->WorkingDirectory());
   fBrowsables->Add(fFiles, "ROOT Files");

   fgRootInit = kTRUE;
}

//______________________________________________________________________________
TROOT::~TROOT()
{
   // Clean up and free resources used by ROOT (files, network sockets,
   // shared memory segments, etc.).

   if (gROOT == this) {

      fgRootInit = kFALSE;

      // Return when error occured in TCint, i.e. when setup file(s) are
      // out of date
      if (!fVersionInt) return;

      // ATTENTION!!! Order is important!

//      fSpecials->Delete();   SafeDelete(fSpecials);    // delete special objects : PostScript, Minuit, Html
#ifdef WIN32
//  Under Windows, one has to restore the color palettes created by individual canvases
      fCanvases->Delete();    SafeDelete(fCanvases);    // first close canvases
#endif
      fFiles->Delete();       SafeDelete(fFiles);       // and files
      fSockets->Delete();     SafeDelete(fSockets);     // and sockets
      fMappedFiles->Delete("slow");                     // and mapped files
      TSeqCollection *tl = fMappedFiles; fMappedFiles = 0; delete tl;

//      fProcesses->Delete();  SafeDelete(fProcesses);   // then terminate processes
//      fFunctions->Delete();  SafeDelete(fFunctions);   // etc..
//      fListHead->Delete();   SafeDelete(fListHead);    // delete objects in current directory
//      fColors->Delete();     SafeDelete(fColors);
//      fStyles->Delete();     SafeDelete(fStyles);
//      fGeometries->Delete(); SafeDelete(fGeometries);
//      fBrowsers->Delete();   SafeDelete(fBrowsers);
//      fBrowsables->Delete(); SafeDelete(fBrowsables);
//      fMessageHandlers->Delete(); SafeDelete(fMessageHandlers);
//      if (fTypes) fTypes->Delete();
//      SafeDelete(fTypes);
//      if (fGlobals) fGlobals->Delete();
//      SafeDelete(fGlobals);
//      if (fGlobalFunctions) fGlobalFunctions->Delete();
//      SafeDelete(fGlobalFunctions);
//      fClasses->Delete();    SafeDelete(fClasses);     // TClass'es must be deleted last

      // Problem deleting the interpreter. Want's to delete objects already
      // deleted in the dtor's above. Crash.
      //SafeDelete(fInterpreter);

      // Remove shared libraries produced by the TSystem::CompileMacro() call
      gSystem->CleanCompiledMacros();

      // Cleanup system class
      delete gSystem;

      // Prints memory stats
      TStorage::PrintStatistics();

      gROOT = 0;
   }
}

//______________________________________________________________________________
void TROOT::Browse(TBrowser *b)
{
   TObject *obj;
   TIter next(fBrowsables);

   while ((obj = (TObject *) next()))
      b->Add( obj, (char *) next.GetOption() );
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
TObject *TROOT::FindObject(const char *name, void *&where)
{
//*-*-*-*-*Returns address of a ROOT object if it exists*-*-*-*-*-*-*-*-*-*
//*-*      =============================================
//*-*
//*-*  This function looks in the following order in the ROOT lists:
//*-*     - List of files
//*-*     - List of memory mapped files
//*-*     - List of functions
//*-*     - List of geometries
//*-*     - List of canvases
//*-*     - List of styles
//*-*     - List of specials
//*-*     - List of materials in current geometry
//*-*     - List of shapes in current geometry
//*-*     - List of matrices in current geometry
//*-*     - List of Nodes in current geometry
//*-*     - Current Directory in memory
//*-*     - Current Directory on file
//*-*-
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

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
      temp  = fGeometries->FindObject(name);
      where = fGeometries;
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
   if (!temp && TClassTable::GetDict("TGeometry")) {
      TObjArray *loc = (TObjArray*)ProcessLineFast(Form("TGeometry::Get(\"%s\")",name));
      if (loc) {
         temp  = loc->At(0);
         where = loc->At(1);
      }
   }
   if (!temp && gDirectory) {
      temp  = gDirectory->Get(name);
      where = gDirectory;
   }
   if (!temp && gPad) {
      TVirtualPad *canvas = gPad->GetVirtCanvas();
      if (fCanvases->FindObject(canvas)) {  //this check in case call from TCanvas ctor
         temp  = canvas->GetPrimitive(name);
         where = canvas;
         if (!temp && canvas != gPad) {
            temp  = gPad->GetPrimitive(name);
            where = gPad;
         }
      }
   }
   if (!temp) return 0;
   if (temp->TestBit(kNotDeleted)) return temp;
   return 0;
}

//______________________________________________________________________________
const char *TROOT::FindObjectClassName(const char *name) const
{
//*-*-*-*-*Returns class name of a ROOT object including CINT globals*-*
//*-*      ==========================================================

//*-* Search first in the list of "standard" objects
   TObject *obj = gROOT->FindObject(name);
   if (obj) return obj->ClassName();

//*-* Is it a global variable?
   TGlobal *g = gROOT->GetGlobal(name);
   if (g) return g->GetTypeName();

   return 0;
}

//______________________________________________________________________________
TClass *TROOT::GetClass(const char *name, Bool_t load)
{
//*-*-*-*-*Return pointer to class with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      =================================

   if (!GetListOfClasses()) return 0;

   TClass *cl = (TClass*)GetListOfClasses()->FindObject(name);
   if (cl) return cl;

   TDataType *objType = gROOT->GetType(name,load);
   if (objType) {
     const Char_t *typdfName = objType->GetTypeName();
     if (typdfName && strcmp(typdfName,name)) {
       cl = GetClass(typdfName,load);
       return cl;
     }
   }

   if (!load) return 0;

   VoidFuncPtr_t dict = TClassTable::GetDict(name);
   if (dict) {
      (dict)();
      return GetClass(name);
   }
   return 0;
}

//______________________________________________________________________________
TColor *TROOT::GetColor(Int_t color)
{
//*-*-*-*-*-*-*-*Return address of color with index color*-*-*-*-*-*-*-*-*
//*-*            ========================================
   TColor *col = (TColor*)gROOT->GetListOfColors()->At(color);
   if (col && col->GetNumber() == color) return col;
   TIter   next(gROOT->GetListOfColors());
   while ((col = (TColor *) next()))
      if (col->GetNumber() == color) return col;

   return 0;
}

//______________________________________________________________________________
VoidFuncPtr_t TROOT::GetMakeDefCanvas()
{
//*-*-*-*-*-*-*-*Return default canvas function*-*-*-*-*-*-*-*-*
//*-*            ==============================

   return fgMakeDefCanvas;
}


//______________________________________________________________________________
TDataType *TROOT::GetType(const char *name, Bool_t load)
{
//*-*-*-*-*Return pointer to type with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      =================================

   const char *tname = name + strspn(name," ");
   if (!strncmp(tname,"virtual",7)) {
      tname += 7; tname += strspn(tname," ");
   }
   if (!strncmp(tname,"const",5)) {
      tname += 5; tname += strspn(tname," ");
   }
   size_t nch = strlen(tname);
   while (tname[nch-1] == ' ') nch--;

   TDataType *idcur;
   TIter      next(GetListOfTypes(load));
   while ((idcur = (TDataType *) next())) {
     if (strlen(idcur->GetName()) != nch) continue;
     if (strstr(tname, idcur->GetName()) == tname) {
        return idcur;
     }
   }

   return 0;
}

//______________________________________________________________________________
TFile *TROOT::GetFile(const char *name)
{
//*-*-*-*-*Return pointer to file with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      ================================

   return (TFile*)GetListOfFiles()->FindObject(name);
}

//______________________________________________________________________________
TStyle *TROOT::GetStyle(const char *name)
{
//*-*-*-*-*Return pointer to style with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      =================================

   return (TStyle*)GetListOfStyles()->FindObject(name);
}

//______________________________________________________________________________
TObject *TROOT::GetFunction(const char *name)
{
//*-*-*-*-*Return pointer to function with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      ===================================

   TObject *f1 = fFunctions->FindObject(name);
   if (f1) return f1;

   if (!TClassTable::GetDict("TF1")) return 0;
   ProcessLineFast("TF1::InitStandardFunctions();");

   return fFunctions->FindObject(name);
}

//______________________________________________________________________________
TGlobal *TROOT::GetGlobal(const char *name, Bool_t load)
{
   // Return pointer to global variable by name. If load is true force
   // reading of all currently defined globals from CINT (more expensive).

   return (TGlobal *)GetListOfGlobals(load)->FindObject(name);
}

//______________________________________________________________________________
TGlobal *TROOT::GetGlobal(TObject *addr, Bool_t load)
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
            if (*(long *)g->GetAddress() == (long)addr) return g;
         } else {
            if (g->GetAddress() == addr) return g;
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

      Long_t faddr = (Long_t)fInterpreter->GetInterfaceMethod(0, (char *)function,
                                                              (char *)params);
      if (!faddr) return 0;

      TFunction *f;
      TIter      next(GetListOfGlobalFunctions(load));

      while ((f = (TFunction *) next())) {
         if (faddr == (Long_t) f->InterfaceMethod());
            return f;
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

      Long_t faddr = (Long_t)fInterpreter->GetInterfaceMethodWithPrototype(0,
                                         (char *)function, (char *)proto);
      if (!faddr) return 0;

      TFunction *f;
      TIter      next(GetListOfGlobalFunctions(load));

      while ((f = (TFunction *) next())) {
         if (faddr == (Long_t) f->InterfaceMethod());
            return f;
      }
      return 0;
   }
}

//______________________________________________________________________________
TObject *TROOT::GetGeometry(const char *name)
{
//*-*-*-*-*Return pointer to Geometry with name*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      ===================================

   return GetListOfGeometries()->FindObject(name);
}

//______________________________________________________________________________
TSeqCollection *TROOT::GetListOfGlobals(Bool_t load)
{
   // Return list containing the TGlobals currently defined.
   // Since globals are created and deleted during execution of the
   // program, we need to update the list of globals every time we
   // execute this method. However, when calling this function in
   // a (tight) loop where no interpreter symbols will be created
   // you can set load=kFALSE (default).

   if (!fGlobals) {
      fGlobals = new THashList(this, 100, 3);
      load = kTRUE;
   }

   if (!fInterpreter)
      Fatal("GetListOfGlobals", "fInterpreter not initialized");

   if (load)
      fInterpreter->UpdateListOfGlobals();

   return fGlobals;
}

//______________________________________________________________________________
TSeqCollection *TROOT::GetListOfGlobalFunctions(Bool_t load)
{
   // Return list containing the TFunctions currently defined.
   // Since functions are created and deleted during execution of the
   // program, we need to update the list of functions every time we
   // execute this method. However, when calling this function in
   // a (tight) loop where no interpreter symbols will be created
   // you can set load=kFALSE (default).

   if (!fGlobalFunctions) {
      fGlobalFunctions = new THashList(this, 100, 3);
      load = kTRUE;
   }

   if (!fInterpreter)
      Fatal("GetListOfGlobalFunctions", "fInterpreter not initialized");

   if (load)
      fInterpreter->UpdateListOfGlobalFunctions();

   return fGlobalFunctions;
}

//______________________________________________________________________________
TSeqCollection *TROOT::GetListOfTypes(Bool_t load)
{
   // Return list containing all TDataTypes (typedefs) currently defined.
   // Since types can be added and removed during execution of the
   // program, we need to update the list of types every time we
   // execute this method. However, when calling this function in
   // a (tight) loop where no new types will be created
   // you can set load=kFALSE (default).

   if (!fTypes) {
      fTypes = new THashList(this, 100, 3);
      load = kTRUE;

      // Add also basic types (like a identity typedef "typedef int int"
      fTypes->Add(new TDataType("char"));
      fTypes->Add(new TDataType("unsigned char"));
      fTypes->Add(new TDataType("short"));
      fTypes->Add(new TDataType("unsigned short"));
      fTypes->Add(new TDataType("int"));
      fTypes->Add(new TDataType("unsigned int"));
      fTypes->Add(new TDataType("long"));
      fTypes->Add(new TDataType("unsigned long"));
      fTypes->Add(new TDataType("float"));
      fTypes->Add(new TDataType("double"));
      fTypes->Add(new TDataType("void"));
   }

   if (!fInterpreter)
      Fatal("GetListOfTypes", "fInterpreter not initialized");

   if (load)
      fInterpreter->UpdateListOfTypes();

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
   else {
      if (command && strlen(command))
         fApplication->SetIdleTimer(idleTimeInSec, command);
      else
         Warning("Idle", "must specify non null idle command");
   }
}

//______________________________________________________________________________
Int_t TROOT::IgnoreInclude(const char *fname, const char *expandedfname)
{
  // Return true if the given include file correspond to a class that has
  // been loaded through a compiled dictionnary.

  Int_t result = 0;

  if ( fname == 0 ) return result;

  TString className(fname);

  // Remove extension if any.
  Int_t where = className.Last('.');
  if (where != kNPOS) className.Remove( where );
  className = gSystem->BaseName(className);

  TClass *cla = GetClass(className);
  if ( cla ) {
    result = strcmp( gSystem->BaseName(cla->GetDeclFileName()),fname ) == 0;
  }
  return result;
}

//______________________________________________________________________________
void TROOT::InitSystem()
{
   // Initialize operating system interface.

   if (gSystem == 0) {
#if defined(R__UNIX)
      gSystem = new TUnixSystem;
#elif defined(R__MAC)
      gSystem = new TMacSystem;
#elif defined(WIN32)
      gSystem = new TWinNTSystem;
#elif defined(R__VMS)
      gSystem = new TVmsSystem;
#else
      gSystem = new TSystem;
#endif

      if (gSystem->Init())
         Fatal("InitSystem", "can't init operating system layer");

      // read default files
      gEnv = new TEnv(".rootrc");

      gDebug = gEnv->GetValue("Root.Debug", 0);

      if (gEnv->GetValue("Root.MemStat", 0))
         TStorage::EnableStatistics();
      int msize = gEnv->GetValue("Root.MemStat.size", -1);
      int mcnt  = gEnv->GetValue("Root.MemStat.cnt", -1);
      if (msize != -1 || mcnt != -1)
         TStorage::EnableStatistics(msize, mcnt);

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
Int_t TROOT::LoadClass(const char *classname, const char *libname)
{
   // Check if class "classname" is known to the interpreter. If
   // not it will load library "libname". Returns 0 on successful loading
   // and -1 in case libname does not exist or in case of error.

   if (TClassTable::GetDict(classname)) return 0;

   Int_t err;

   if (classname[0] != 'T')
      err = gSystem->Load(libname, 0, kTRUE);
   else {
      // special case for ROOT classes Txxx
      char *lib, *path;
#ifdef WIN32
      lib = Form("lib%s", libname);       // used to be Root_%s
#else
      lib = Form("lib%s", libname);
#endif
      if ((path = gSystem->DynamicPathName(lib, kTRUE))) {
         err = gSystem->Load(path, 0, kTRUE);
         delete [] path;
      } else
         err = gSystem->Load(libname, 0, kTRUE);
   }

   if (err == 0)
      GetListOfTypes(kTRUE);

   if (err == -1)
      ;  //Error("LoadClass", "library %s could not be loaded", libname);

   if (err == 1) {
      Error("LoadClass", "library %s already loaded, but class %s unknown",
            libname, classname);
      err = -1;
   }

   return err;
}

//______________________________________________________________________________
void TROOT::ls(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*To list all objects of the application*-*-*-*-*-*
//*-*                      ======================================
//*-*  Loop on all objects created in the ROOT linked lists
//*-*  objects may be files and windows or any other object directly
//*-*  attached to the ROOT linked list.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//   TObject::SetDirLevel();
//   GetList()->ForEach(TObject,ls)(option);
   TDirectory::ls(option);
}

//______________________________________________________________________________
void TROOT::LoadMacro(const char *filename)
{
   // Load a macro in the interpreter's memory. Equivalent to the command line
   // command ".L filename".

   if (fInterpreter) {
      char *fn  = Strip(filename);
      char *mac = gSystem->Which(GetMacroPath(), fn, kReadPermission);
      if (!mac)
         Error("LoadMacro", "macro %s not found in path %s", fn, GetMacroPath());
      else
         fInterpreter->LoadMacro(mac);
      delete [] fn;
      delete [] mac;
   }
}

//______________________________________________________________________________
Int_t TROOT::Macro(const char *filename)
{
   // Execute a macro in the interpreter. Equivalent to the command line
   // command ".x filename".

   Int_t result = 0;

   if (fInterpreter) {
      char *fn  = Strip(filename);
      char *mac = gSystem->Which(GetMacroPath(), fn, kReadPermission);
      if (!mac)
         Error("Macro", "macro %s not found in path %s", fn, GetMacroPath());
      else
         result = fInterpreter->ExecuteMacro(mac);
      delete [] fn;
      delete [] mac;

      if (gPad) gPad->Update();
   }

   return result;
}

//______________________________________________________________________________
void  TROOT::Message(Int_t id, TObject *obj)
{
   // Process message id called by obj

   TIter next(fMessageHandlers);
   TMessageHandler *mh;
   while ((mh = (TMessageHandler*)next())) {
      mh->HandleMessage(id,obj);
   }
}

//______________________________________________________________________________
void TROOT::ProcessLine(const char *line)
{
   // Process interpreter command via TApplication::ProcessLine().
   // On Win32 the line will be processed a-synchronously by sending
   // it to the CINT interpreter thread. For explicit synchrounous processing
   // use ProcessLineSync(). On non-Win32 platforms there is not difference
   // between ProcessLine() and ProcessLineSync().

   if (!fApplication) {
      // circular Form() buffer will be re-used in CreateApplication() (too
      // many calls to Form()), so we need to save "line"
      char *sline = StrDup(line);
      TApplication::CreateApplication();
      line = Form("%s", sline);
      delete [] sline;
   }

   fApplication->ProcessLine(line);
}

//______________________________________________________________________________
void TROOT::ProcessLineSync(const char *line)
{
   // Process interpreter command via TApplication::ProcessLine().
   // On Win32 the line will be processed synchronously (i.e. it will
   // only return when the CINT interpreter thread has finished executing
   // the line). On non-Win32 platforms there is not difference between
   // ProcessLine() and ProcessLineSync().

   if (!fApplication)
      TApplication::CreateApplication();

   fApplication->ProcessLine(line, kTRUE);
}

//______________________________________________________________________________
Long_t TROOT::ProcessLineFast(const char *line)
{
   // Process interpreter command directly via CINT interpreter.
   // Only executable statements are allowed (no variable declarations),
   // In all other cases use TROOT::ProcessLine().

   Long_t result = 0;

   if (fInterpreter)
      result = fInterpreter->Calc(line);

   return result;
}

//______________________________________________________________________________
void TROOT::Proof(const char *cluster)
{
   // Start PROOF session on a specific cluster (default is "proof").
   // The cluster configuration is defined either in the file ."cluster".conf,
   // or $HOME/."cluster".conf or /usr/proof/etc/"cluster".conf.
   // The TProof object can be accessed via the gProof global. Creating a
   // new TProof object will delete the current one.

   // make sure libProof is loaded and TProof can be created
   if (gROOT->LoadClass("TProof","Proof")) return;
   if (gROOT->LoadClass("TTreePlayer","TreePlayer")) return;

   ProcessLine(Form("new TProof(\"%s\");", cluster));
}

//______________________________________________________________________________
void TROOT::Reset(Option_t *)
{
   // Delete all global interpreter objects created since the last call to Reset
   //
   // If option="a" is set reset to startup context (i.e. unload also
   // all loaded files, classes, structs, typedefs, etc.).
   //
   // This function is typically used at the beginning (or end) of a macro
   // to clean the environment.

   if (fInterpreter) {
     // if (!strncmp(option, "a", 1)) {
     //    fInterpreter->Reset();
     //    fInterpreter->SaveContext();
     // } else
         fInterpreter->ResetGlobals();

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
      fInterpreter->SaveGlobalsContext();
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
   TClass *cl = gROOT->GetClass(name);
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

   const Int_t kButton    = 101;
   const Int_t kCutG      = 100;
   const Int_t kCurlyLine = 200;
   const Int_t kCurlyArc  = 201;
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
void TROOT::SetMakeDefCanvas(VoidFuncPtr_t makecanvas)
{
   // Static function used to set the address of the default make canvas method.
   // This address is by default initialized to 0.
   // It is set as soon as the library containing the TCanvas class is loaded.

   fgMakeDefCanvas = makecanvas;
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
Bool_t  TROOT::Initialized()
{
   return fgRootInit;
}

//______________________________________________________________________________
const char *TROOT::GetMacroPath()
{
   // Get macro search path. Static utility function.

   static const char *macropath = 0;

   if (macropath == 0) {
      macropath = gEnv->GetValue("Root.MacroPath", (char*)0);
      if (macropath == 0)
#if !defined (__VMS ) && !defined(WIN32)
   #ifdef ROOTMACRODIR
         macropath = ".:" ROOTMACRODIR;
   #else
         macropath = StrDup(Form(".:%s/macros", gRootDir));
   #endif
#elif !defined(__VMS)
   #ifdef ROOTMACRODIR
         macropath = ".;" ROOTMACROPATH;
   #else
         macropath = StrDup(Form(".;%s/macros", gRootDir));
   #endif
#else
/*        if (strrchr(gRootDir,']'))
             *strrchr(gRootDir,']') = '.'; */
         macropath = StrDup(Form("%sTUTORIALS]",gRootDir));
#endif
   }
   return macropath;
}

