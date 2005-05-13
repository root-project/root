// @(#)root/net:$Name:  $:$Id: TGrid.h,v 1.10 2005/05/12 13:19:39 rdm Exp $
// Author: Fons Rademakers   3/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGrid
#define ROOT_TGrid

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGrid                                                                //
//                                                                      //
// Abstract base class defining interface to common GRID services.      //
//                                                                      //
// To open a connection to a GRID use the static method Connect().      //
// The argument of Connect() is of the form:                            //
//    <grid>://<host>[:<port>], e.g.                                    //
// alien://alice.cern.ch, globus://glsvr1.cern.ch, ...                  //
// Depending on the <grid> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
//                                                                      //
// Related classes are TGridResult.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TStopwatch
#include "TStopwatch.h"
#endif

class TGridResult;
class TGridJDL;
class TGridJob;
class TDSet;


class TGrid : public TObject {

protected:
   TString        fGrid;   // type of GRID (AliEn, Globus, ...)
   TString        fHost;   // GRID portal to which we are connected
   TString        fUser;   // user name
   Int_t          fPort;   // port to which we are connected
   TStopwatch     fWatch;  // stop watch to measure file copy speed

   void           PrintProgress(Long_t bytesread, Long_t size);

public:
   TGrid() : fPort(-1) { }

   virtual ~TGrid();

   const char    *GetGrid() const { return fGrid; }
   const char    *GetHost() const { return fHost; }
   const char    *GetUser() const { return fUser; }
   Int_t          GetPort() const { return fPort; }
   virtual Bool_t IsConnected() const { return fPort == -1 ? kFALSE : kTRUE; }

   virtual void Shell() = 0;
   virtual void Stdout() = 0;
   virtual void Stderr() = 0;

   virtual TGridResult *Command(const char *command, Bool_t interactive = kFALSE) = 0;

   virtual TGridResult *Query(const char *path, const char *pattern,
                              const char *conditions, const char *options) = 0;

   virtual TGridResult *LocateSites() { return 0; }

   virtual Bool_t Query2Dataset(TDSet * /*dset*/, const char * /*path*/,
                                const char * /*pattern*/,
                                const char * /*conditions*/,
                                const char * /*options*/) { return kFALSE; }

   //--- file management interface
   virtual Bool_t Cp(const char *src, const char *dst, Bool_t progressbar = kTRUE,
                     UInt_t buffersize = 1000000);
   virtual Bool_t SetCWD(const char * /*path*/) { return kFALSE; }
   virtual const char *GetCWD() { return 0; }

   //--- job Submission Interface
   virtual TGridJob *Submit(const char * /*jdl*/) { return 0; }
   virtual TGridJDL *GetJDLGenerator() { return 0; }

   //--- load desired plugin and setup conection to GRID
   static TGrid *Connect(const char *grid, const char *uid = 0,
                         const char *pw = 0, const char *options = 0);

   ClassDef(TGrid,0)  // ABC defining interface to GRID services
};

R__EXTERN TGrid *gGrid;

#endif
