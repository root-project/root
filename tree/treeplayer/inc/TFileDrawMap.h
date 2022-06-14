// @(#)root/treeplayer:$Id$
// Author: Rene Brun   15/01/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileDrawMap
#define ROOT_TFileDrawMap


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileDrawMap                                                         //
//                                                                      //
// Draw a 2-d map of the objects in a file                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

class TH1;
class TFile;
class TDirectory;
class TBox;
class TBranch;

class TFileDrawMap : public TNamed {

protected:
   TFile         *fFile;           ///< Pointer to the file
   TH1           *fFrame;          ///< Histogram used to draw the map frame
   TString        fKeys;           ///< List of keys
   TString        fOption;         ///< Drawing options
   Int_t          fXsize;          ///< Size in bytes of X axis
   Int_t          fYsize;          ///< Size in K/Mbytes of Y axis

   virtual void     DrawMarker(Int_t marker, Long64_t eseek);
   virtual Bool_t   GetObjectInfoDir(TDirectory *dir, Int_t px, Int_t py, TString &info) const;
   virtual void     PaintBox(TBox &box, Long64_t bseek, Int_t nbytes);
   virtual void     PaintDir(TDirectory *dir, const char *keys);
   virtual TObject *GetObject();

public:
   TFileDrawMap();
   TFileDrawMap(const TFile *file, const char *keys, Option_t *option);
   virtual ~TFileDrawMap();

   virtual void  AnimateTree(const char *branches=""); // *MENU*
   virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
   virtual void  DrawObject(); // *MENU*
   virtual void  DumpObject(); // *MENU*
   virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual char *GetObjectInfo(Int_t px, Int_t py) const;
   virtual void  InspectObject(); // *MENU*
   virtual void  Paint(Option_t *option);

   ClassDef(TFileDrawMap,1);  //Draw a 2-d map of the objects in a file
};

#endif
