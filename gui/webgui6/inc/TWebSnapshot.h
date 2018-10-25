// Author:  Sergey Linev, GSI,  6/04/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWebSnapshot
#define ROOT_TWebSnapshot

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebSnapshot                                                         //
//                                                                      //
// Paint state of object to transfer to JavaScript side                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"

#include <vector>

class TWebSnapshot : public TObject {

protected:

   TString    fObjectID;            ///<   object identifier
   TString    fOption;              ///<   object draw option
   Int_t      fKind{0};             ///<   kind of snapshots
   TObject   *fSnapshot{nullptr};   ///<   snapshot data
   Bool_t     fOwner{kFALSE};       ///<!  if objected owned

   void SetKind(Int_t kind) { fKind = kind; }

public:

   enum {
     kNone = 0,        // dummy
     kObject = 1,      // object itself
     kSVG = 2,         // list of SVG primitives
     kSubPad = 3,      // subpad
     kColors = 4,      // list of ROOT colors
     kPalette = 5      // current color palette
   };

   virtual ~TWebSnapshot();

   void SetObjectIDAsPtr(void* ptr);
   void SetObjectID(const char* id) { fObjectID = id; }
   const char* GetObjectID() const { return fObjectID.Data(); }

   void SetOption(const char* opt) { fOption = opt; }

   void SetSnapshot(Int_t kind, TObject* shot, Bool_t owner = kFALSE);
   Int_t GetKind() const { return fKind; }
   TObject* GetSnapshot() const { return fSnapshot; }

   ClassDef(TWebSnapshot,1)  // Object painting snapshot, used for JSROOT
};

// =================================================================================

class TPadWebSnapshot : public TWebSnapshot {
protected:
   bool fActive{false};                      ///< true when pad is active
   std::vector<TWebSnapshot*> fPrimitives;   ///< list of all primitives, drawn in the pad
public:
   TPadWebSnapshot() { SetKind(kSubPad); }
   virtual ~TPadWebSnapshot();

   void SetActive(bool on = true) { fActive = on; }
   void Add(TWebSnapshot *snap) { fPrimitives.push_back(snap); }

   ClassDef(TPadWebSnapshot,1)  // Pad painting snapshot, used for JSROOT
};
#endif
