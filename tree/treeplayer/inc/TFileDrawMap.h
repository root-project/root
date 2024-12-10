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

#include <map>

class TH2;
class TFile;
class TDirectory;
class TBox;
class TBranch;

class TFileDrawMap : public TNamed {

protected:
   TFile         *fFile = nullptr;      ///<! Pointer to the file, cannot be persistent
   std::map<TBranch*, Int_t> fBranchColors; ///<! map of generated colors for the branches
   TH2           *fFrame = nullptr;     ///< Histogram used to draw the map frame
   TString        fKeys;                ///< List of keys
   Int_t          fXsize = 0;           ///< Size in bytes of X axis
   Int_t          fYsize = 0;           ///< Size in K/Mbytes of Y axis

   virtual void     DrawMarker(Int_t marker, Long64_t eseek);
   virtual bool     GetObjectInfoDir(TDirectory *dir, Int_t px, Int_t py, TString &info) const;
   virtual void     PaintBox(TBox &box, Long64_t bseek, Int_t nbytes);
   virtual void     PaintDir(TDirectory *dir, const char *keys);
   virtual TObject *GetObject();

   TString GetRecentInfo();

public:
   TFileDrawMap();
   TFileDrawMap(const TFile *file, const char *keys, Option_t *option = "");
   ~TFileDrawMap() override;

   virtual void  AnimateTree(const char *branches=""); // *MENU*
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   virtual void  DrawObject(); // *MENU*
   virtual void  DumpObject(); // *MENU*
   char *GetObjectInfo(Int_t px, Int_t py) const override;
   virtual void  InspectObject(); // *MENU*
   void  Paint(Option_t *option) override;

   ClassDefOverride(TFileDrawMap,2);  //Draw a 2-d map of the objects in a file
};

#endif
