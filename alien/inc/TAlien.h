// @(#)root/alien:$Name:  $:$Id: TAlien.h,v 1.1 2002/05/13 10:38:10 rdm Exp $
// Author: Fons Rademakers   13/5/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlien
#define ROOT_TAlien


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlien                                                               //
//                                                                      //
// Class defining interface to AliEn GRID services.                     //
//                                                                      //
// To open a connection to a AliEn GRID use the static method           //
// TGrid::Connect("alien://<host>", ..., ...).                          //
//                                                                      //
// Related classes are TAlienResult and TAlienAttrResult.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

#if !defined(__CINT__)
#include <AliEn.h>
#else
typedef unsigned long Alien_t;
#endif


class TAlien : public TGrid {

private:
   Alien_t    fAlien;    // connection to AliEn server

public:
   TAlien(const char *grid, const char *uid = 0, const char *pw = 0);
   ~TAlien();
   void         Close(Option_t *option="");

   //--- file catalog query
   TGridResult *Query(const char *wildcard);

   //--- file catalog management
   Int_t        AddFile(const char *lfn, const char *pfn, Int_t size);
   Int_t        DeleteFile(const char *lfn);
   Int_t        Mkdir(const char *dir, const char *options = 0) = 0;
   Int_t        Rmdir(const char *dir, const char *options = 0) = 0;
   char        *GetPhysicalFileName(const char *lfn);
   TGridResult *GetPhysicalFileNames(const char *lfn);

   //--- file attribute management
   Int_t        AddAttribute(const char *lfn, const char *attrname,
                             const char *attrval);
   Int_t        DeleteAttribute(const char *lfn, const char *attrname);
   TGridResult *GetAttributes(const char *lfn);

   //--- catalog navigation & browsing
   const char  *Pwd();
   Int_t        Cd(const char *dir);
   TGridResult *Ls(const char *dir, const char *options = 0);
   void         Browse(TBrowser *b);

   //--- status and info
   const char  *GetInfo();

   ClassDef(TAlien,0)  // Interface to AliEn GRID services
};

#endif
