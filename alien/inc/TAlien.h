// @(#)root/alien:$Name:  $:$Id: TAlien.h,v 1.8 2003/11/13 17:01:15 rdm Exp $
// Author: Andreas Peters   5/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
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
// Class defining interface to TAlien GRID services.                    //
//                                                                      //
// To start a local API Grid service, use                               //
//   - TGrid::Connect("alien://localhost");                             //
//   - TGrid::Connect("alien://");                                      //
//                                                                      //
// To force to connect to a running API Service, use                    //
//   - TGrid::Connect("alien://<apihosturl>/?direct");                  //
//                                                                      //
// To get a remote API Service from the API factory service, use        //
//   - TGrid::Connect("alien://<apifactoryurl>");                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

#ifndef ROOT_TGridResult
#include "TGridResult.h"
#endif

#ifndef ROOT_TDSet
#include "TDSet.h"
#endif

class GliteUI;
class TGridJob;


class TAlien : public TGrid {

private:
   // Stream content types.
   // The streams are originally defined in the CODEC.h of
   // the external gliteUI library.
   enum {kSTDOUT = 0, kSTDERR = 1 , kOUTPUT = 2, kENVIR = 3};

   GliteUI   *fGc;    // the GliteUI object implementing the communication layer

   TGridResult         *Command(const char *command, bool interactive = kFALSE);
   virtual TGridResult *Query(const char *path, const char *pattern,
                              const char *conditions, const char *options);
   virtual TGridResult *LocateSites();
   virtual TGridResult *OpenDataset(const char *lfn, const char *options = "");

public:
   TAlien(const char *gridurl, const char *uid=0, const char *passwd=0,
	  const char *options=0);
   virtual ~TAlien();

   void Shell();           // start an interactive ALIEN shell

   void Stdout();          // print the stdout of the last executed command
   void Stderr();          // print the stderr of the last executed command

   TString Escape(const char *input);
   virtual TGridJob *Submit(const char *jdl); // submit a grid job
   virtual TGridJDL *GetJDLGenerator();       // get a AliEn grid JDL object

   ClassDef(TAlien,0)  // Interface to Alien GRID services
};

#endif
