// @(#)root/treeplayer:$Name:  $:$Id: TFileMap.h,v 1.1 2003/01/15 18:48:16 brun Exp $
// Author: Rene Brun   15/01/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileMap
#define ROOT_TFileMap


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileMap                                                             //
//                                                                      //
// Draw a 2-d map of the objects in a file                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TH1;
class TFile;
class TDirectory;

class TFileMap : public TNamed {

protected:
    TFile         *fFile;           //pointer to the file
    TH1           *fFrame;          //histogram used to draw the map frame
    TString        fKeys;           //list of keys
    TString        fOption;         //drawing options
    Int_t          fXsize;          //size in bytes of X axis
    Int_t          fYsize;          //size in K/Mbytes of Y axis
           
    virtual Bool_t   GetObjectInfoDir(TDirectory *dir, Int_t px, Int_t py, char *info) const;
    virtual void     PaintDir(TDirectory *dir, const char *keys);
    virtual TObject *GetObject();    
public:
    TFileMap();
    TFileMap(const TFile *file, const char *keys, Option_t *option);
    virtual ~TFileMap();

    virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
    virtual void  DrawObject(); // *MENU*
    virtual void  DumpObject(); // *MENU*
    virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual char *GetObjectInfo(Int_t px, Int_t py) const;
    virtual void  InspectObject(); // *MENU*
    virtual void  Paint(Option_t *option);
    
    ClassDef(TFileMap,1)  //Draw a 2-d map of the objects in a file
};

#endif
