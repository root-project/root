// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveUtil.h"
#include "TEveElement.h"

#include "TError.h"
#include "TPad.h"
#include "TGeoManager.h"
#include "TClass.h"
#include "TMath.h"

#include "TStyle.h"
#include "TColor.h"

#include "TROOT.h"
#include "TInterpreter.h"
#include "TSystem.h"

#include "TGClient.h"
#include "TGMimeTypes.h"

#include <list>
#include <algorithm>
#include <iostream>

//______________________________________________________________________________
// TEveUtil
//
// Standard utility functions for Reve.

ClassImp(TEveUtil)

//______________________________________________________________________________
void TEveUtil::SetupEnvironment()
{
   // Setup Include and Macro paths.
   // Since inclusion into ROOT this does nothing but could
   // potentially be reactivated if some common macros are established
   // and shipped with binary ROOT (in macros/eve). For example, these
   // might be used to spawn specific GUI / GL configurations.

   static const TEveException eh("TEveUtil::SetupEnvironment");
   static Bool_t setupDone = kFALSE;

   if (setupDone) {
      Info(eh.Data(), "has already been run.");
      return;
   }

   // Old initialization for ALICE.
   // Left as an example.
   /*
   // Check if REVESYS exists, try fallback to $ALICE_ROOT/EVE.
   if(gSystem->Getenv("REVESYS") == 0) {
      if(gSystem->Getenv("ALICE_ROOT") != 0) {
         Info(eh.Data(), "setting REVESYS from ALICE_ROOT.");
         gSystem->Setenv("REVESYS", Form("%s/EVE", gSystem->Getenv("ALICE_ROOT")));
      } else {
         Error(eh.Data(), "REVESYS not defined, neither is ALICE_ROOT.");
         gSystem->Exit(1);
      }
   }
   if(gSystem->AccessPathName(gSystem->Getenv("REVESYS")) == kTRUE) {
      Error(eh.Data(), "REVESYS '%s' does not exist.", gSystem->Getenv("REVESYS"));
      gSystem->Exit(1);
   }

   TString macPath(gROOT->GetMacroPath());
   macPath += Form(":%s/macros", gSystem->Getenv("REVESYS"));
   gInterpreter->AddIncludePath(gSystem->Getenv("REVESYS"));
   if(gSystem->Getenv("ALICE_ROOT") != 0) {
      macPath += Form(":%s/alice-macros", gSystem->Getenv("REVESYS"));
      gInterpreter->AddIncludePath(Form("%s/include", gSystem->Getenv("ALICE_ROOT")));
      gInterpreter->AddIncludePath(gSystem->Getenv("ALICE_ROOT"));
   }
   gROOT->SetMacroPath(macPath);
   */
}

//______________________________________________________________________________
void TEveUtil::SetupGUI()
{
   // Setup icon pictures and mime-types.

   TString fld( Form("%s/icons/", gSystem->Getenv("ROOTSYS")) );

   TEveElement::fgRnrIcons[0] = gClient->GetPicture(fld + "eve_rnr00_t.xpm");
   TEveElement::fgRnrIcons[1] = gClient->GetPicture(fld + "eve_rnr01_t.xpm");
   TEveElement::fgRnrIcons[2] = gClient->GetPicture(fld + "eve_rnr10_t.xpm");
   TEveElement::fgRnrIcons[3] = gClient->GetPicture(fld + "eve_rnr11_t.xpm");

   TEveElement::fgListTreeIcons[0] = gClient->GetPicture("folder_t.xpm");
   TEveElement::fgListTreeIcons[1] = gClient->GetPicture(fld + "eve_viewer.xpm");
   TEveElement::fgListTreeIcons[2] = gClient->GetPicture(fld + "eve_scene.xpm");
   TEveElement::fgListTreeIcons[3] = gClient->GetPicture(fld + "eve_pointset.xpm");
   TEveElement::fgListTreeIcons[4] = gClient->GetPicture(fld + "eve_track.xpm");
   TEveElement::fgListTreeIcons[5] = gClient->GetPicture(fld + "eve_text.gif");
   TEveElement::fgListTreeIcons[6] = gClient->GetPicture(fld + "eve_axes.xpm");
   TEveElement::fgListTreeIcons[7] = gClient->GetPicture("ofolder_t.xpm");


   gClient->GetMimeTypeList()->AddType("root/tmacro", "TEveMacro",
                                       "tmacro_s.xpm", "tmacro_t.xpm", "");
}

/******************************************************************************/

namespace
{
//______________________________________________________________________________
void ChompTail(TString& s, char c='.')
{
   // Remove last part of string 's', starting from the last
   // occurrence of character 'c'.

   Ssiz_t p = s.Last(c);
   if(p != kNPOS)
      s.Remove(p);
}
}

//______________________________________________________________________________
Bool_t TEveUtil::CheckMacro(const Text_t* mac)
{
   // Checks if macro 'mac' is loaded.

   // Axel's advice; now sth seems slow, using old method below for test.
   // return gROOT->GetInterpreter()->IsLoaded(mac);

   // Previous version expected function with same name and used ROOT's
   // list of global functions.

   TString foo(mac); ChompTail(foo);
   /*
     if(recreate) {
     TCollection* logf = gROOT->GetListOfGlobalFunctions(kFALSE);
     logf->SetOwner();
     logf->Clear();
     }
   */
   if (gROOT->GetGlobalFunction(foo.Data(), 0, kFALSE) != 0)
      return kTRUE;
   else
      return (gROOT->GetGlobalFunction(foo.Data(), 0, kTRUE) != 0);
}

//______________________________________________________________________________
void TEveUtil::AssertMacro(const Text_t* mac)
{
   // Load and execute macro 'mac' if it has not been loaded yet.

   if(CheckMacro(mac) == kFALSE) {
      gROOT->Macro(mac);
   }
}

//______________________________________________________________________________
void TEveUtil::Macro(const Text_t* mac)
{
   // Execute macro 'mac'. Do not reload the macro.

   if(CheckMacro(mac) == kFALSE) {
      gROOT->LoadMacro(mac);
   }
   TString foo(mac); ChompTail(foo); foo += "()";
   gROOT->ProcessLine(foo.Data());
}

//______________________________________________________________________________
void TEveUtil::LoadMacro(const Text_t* mac)
{
   // Makes sure that macro 'mac' is loaded, but do not reload it.

   if(CheckMacro(mac) == kFALSE) {
      gROOT->LoadMacro(mac);
   }
}

/******************************************************************************/
// Color management
/******************************************************************************/

//______________________________________________________________________________
void TEveUtil::ColorFromIdx(Color_t ci, UChar_t col[4], Bool_t alpha)
{
   // Fill col with RGBA values corresponding to index ci. If alpha
   // is true, set alpha component of col to 255.
   // ROOT's indexed color palette does not support transparency.

   if (ci < 0) {
      col[0] = col[1] = col[2] = col[3] = 0;
      return;
   }
   TColor* c = gROOT->GetColor(ci);
   if (c) {
      col[0] = (UChar_t)(255*c->GetRed());
      col[1] = (UChar_t)(255*c->GetGreen());
      col[2] = (UChar_t)(255*c->GetBlue());
      if (alpha) col[3] = 255;
   }
}

//______________________________________________________________________________
void TEveUtil::ColorFromIdx(Float_t f1, Color_t c1, Float_t f2, Color_t c2,
			    UChar_t col[4], Bool_t alpha)
{
   // Fill col with weighted RGBA values corresponding to
   // color-indices c1 and c2. If alpha is true, set alpha component
   // of col to 255.

   TColor* t1 = gROOT->GetColor(c1);
   TColor* t2 = gROOT->GetColor(c2);
   if(t1 && t2) {
      col[0] = (UChar_t)(255*(f1*t1->GetRed()   + f2*t2->GetRed()));
      col[1] = (UChar_t)(255*(f1*t1->GetGreen() + f2*t2->GetGreen()));
      col[2] = (UChar_t)(255*(f1*t1->GetBlue()  + f2*t2->GetBlue()));
      if (alpha) col[3] = 255;
   }
}

//______________________________________________________________________________
Color_t* TEveUtil::FindColorVar(TObject* obj, const Text_t* varname)
{
   // Find address of Color_t data-member with name varname in object
   // obj.
   //
   // This is used to access color information for TGListTreeItem
   // coloration from visualization macros that wrap TObjects into
   // TEveElementObjectPtr instances.

   static const TEveException eh("TEveUtil::FindColorVar");

   Int_t off = obj->IsA()->GetDataMemberOffset(varname);
   if(off == 0)
      throw(eh + "could not find member '" + varname + "' in class " + obj->IsA()->GetName() + ".");
   return (Color_t*) (((char*)obj) + off);
}


/******************************************************************************/
// Text formatting
/******************************************************************************/

//______________________________________________________________________________
const char* TEveUtil::FormAxisValue(Float_t x)
{
   // Returns formatted text suitable for display of value 'x' on an
   // axis tick-mark.

   // There is a problem on windows: for values printed with the %f.0
   // format 8 trailing zeros are displayed.

   using namespace TMath;

   if (Abs(x) > 1000)
      return Form("%d", (Int_t) 10*Nint(x/10.0f));
   if (Abs(x) > 100 || x == Nint(x))
      return Form("%d", (Int_t) Nint(x));
   if (Abs(x) > 10)
      return Form("%.1f", x);
   if (Abs(x) >= 0.01 )
      return Form("%.2f", x);
   return "0";
}


/******************************************************************************/
// TEveException
/******************************************************************************/

//______________________________________________________________________________
//
// Exception class thrown by TEve classes and macros.

ClassImp(TEveException)

//______________________________________________________________________________
bool operator==(const TString& t, const std::string& s)
{ return (s == t.Data()); }

bool operator==(const std::string&  s, const TString& t)
{ return (s == t.Data()); }

// Exc

TEveException::TEveException(const std::string& s) : TString(s.c_str())
{
   // Constructor.
}

// Exc + ops

TEveException operator+(const TEveException &s1, const std::string &s2)
{ TEveException r(s1); r += s2; return r; }

TEveException operator+(const TEveException &s1, const TString &s2)
{ TEveException r(s1); r += s2; return r; }

TEveException operator+(const TEveException &s1,  const char *s2)
{ TEveException r(s1); r += s2; return r; }


/******************************************************************************/
// TEvePadHolder
/******************************************************************************/

//______________________________________________________________________________
//
// Exception safe wrapper for setting gPad.
// Optionally calls gPad->Modified()/Update() in destructor.

ClassImp(TEvePadHolder)

//______________________________________________________________________________
TEvePadHolder::TEvePadHolder(Bool_t modify_update_p, TVirtualPad* new_pad, Int_t subpad) :
   fOldPad        (gPad),
   fModifyUpdateP (modify_update_p)
{
   // Constructor.

   if (new_pad != 0)
      new_pad->cd(subpad);
   else
      gPad = 0;
}

//______________________________________________________________________________
TEvePadHolder::~TEvePadHolder()
{
   // Destructor.

   if(fModifyUpdateP && gPad != 0) {
      gPad->Modified();
      gPad->Update();
   }
   gPad = fOldPad;
}


/******************************************************************************/
// TEveGeoManagerHolder
/******************************************************************************/

//______________________________________________________________________________
//
// Exception safe wrapper for setting gGeoManager.
// Functionality to lock-unlock via setting of a static lock in
// TGeoManager should be added (new feature of TGeoManager).

ClassImp(TEveGeoManagerHolder)

//______________________________________________________________________________
TEveGeoManagerHolder::TEveGeoManagerHolder(TGeoManager* new_gmgr) :
   fManager(gGeoManager)
{
   // Constructor.

   gGeoManager = new_gmgr;
}

//______________________________________________________________________________
TEveGeoManagerHolder::~TEveGeoManagerHolder()
{
   // Destructor.

   gGeoManager = fManager;
}


/******************************************************************************/
// TEveRefCnt
/******************************************************************************/

//______________________________________________________________________________
//
// Base-class for reference-counted objects.
// By default the object is destroyed when zero referece-count is reached.

ClassImp(TEveRefCnt)


/******************************************************************************/
// TEveRefBackPtr
/******************************************************************************/

//______________________________________________________________________________
//
// Base-class for reference-counted objects with reverse references to
// TEveElement objects.

ClassImp(TEveRefBackPtr)

//______________________________________________________________________________
TEveRefBackPtr::TEveRefBackPtr() :
   TEveRefCnt(),
   fBackRefs()
{
   // Default constructor.
}

//______________________________________________________________________________
TEveRefBackPtr::~TEveRefBackPtr()
{
   // Destructor. Noop, should complain if back-ref list is not empty.

   // !!! Complain if list not empty.
}

//______________________________________________________________________________
TEveRefBackPtr::TEveRefBackPtr(const TEveRefBackPtr&) :
   TEveRefCnt(),
   fBackRefs()
{
   // Copy constructor. New copy starts with zero reference count and
   // empty back-reference list.
}

//______________________________________________________________________________
TEveRefBackPtr& TEveRefBackPtr::operator=(const TEveRefBackPtr&)
{
   // Assignment operator. Reference count and back-reference
   // information is not assigned as these object hold pointers to a
   // specific object.

   return *this;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRefBackPtr::IncRefCount(TEveElement* re)
{
   // Increase reference cound and add re to the list of back-references.

   TEveRefCnt::IncRefCount();
   fBackRefs.push_back(re);
}

//______________________________________________________________________________
void TEveRefBackPtr::DecRefCount(TEveElement* re)
{
   // Decrease reference cound and remove re from the list of back-references.

   static const TEveException eh("TEveRefBackPtr::DecRefCount ");

   std::list<TEveElement*>::iterator i =
      std::find(fBackRefs.begin(), fBackRefs.end(), re);
   if (i != fBackRefs.end()) {
      fBackRefs.erase(i);
      TEveRefCnt::DecRefCount();
   } else {
      Warning(eh, Form("render element '%s' not found in back-refs.",
                       re->GetObject(eh)->GetName()));
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRefBackPtr::UpdateBackPtrItems()
{
   // Call UpdateItems() on list of reverse references.

   std::list<TEveElement*>::iterator i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      (*i)->UpdateItems();
      ++i;
   }
}

//______________________________________________________________________________
void TEveRefBackPtr::StampBackPtrElements(UChar_t stamps)
{
   // Add givem stamps to elements in the list of reverse references.

   std::list<TEveElement*>::iterator i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      (*i)->AddStamp(stamps);
      ++i;
   }
}
