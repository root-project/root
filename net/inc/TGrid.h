// @(#)root/net:$Name:  $:$Id: TGrid.h,v 1.1 2002/05/13 10:35:19 rdm Exp $
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

class TGridResult;
class TBrowser;


class TGrid : public TObject {

protected:
   TString    fGrid;     // type of GRID (AliEn, Globus, ...)
   TString    fHost;     // GRID portal to which we are connected
   Int_t      fPort;     // port to which we are connected

   TGrid() { fPort = -1; }

public:
   virtual ~TGrid() { }
   virtual void         Close(Option_t *option="") = 0;

   //--- file catalog query
   virtual TGridResult *Query(const char *wildcard) = 0;

   //--- file catalog management
   virtual Int_t        AddFile(const char *lfn, const char *pfn, Int_t size) = 0;
   virtual Int_t        DeleteFile(const char *lfn) = 0;
   virtual Int_t        Mkdir(const char *dir, const char *options = 0) = 0;
   virtual Int_t        Rmdir(const char *dir, const char *options = 0) = 0;
   virtual char        *GetPhysicalFileName(const char *lfn) = 0;
   virtual TGridResult *GetPhysicalFileNames(const char *lfn) = 0;

   //--- file attribute management
   virtual Int_t        AddAttribute(const char *lfn, const char *attrname,
                                     const char *attrval) = 0;
   virtual Int_t        DeleteAttribute(const char *lfn, const char *attrname) = 0;
   virtual TGridResult *GetAttributes(const char *lfn) = 0;

   //--- catalog navigation & browsing
   virtual const char  *Pwd() = 0;
   virtual Int_t        Cd(const char *dir) = 0;
   virtual TGridResult *Ls(const char *dir, const char *options = 0) = 0;
   virtual void         Browse(TBrowser *b) = 0;

   //--- status and info
   virtual const char  *GetInfo() = 0;
   const char          *GetGrid() const { return fGrid; }
   const char          *GetHost() const { return fHost; }
   Int_t                GetPort() const { return fPort; }
   Bool_t               IsConnected() const { return fPort == -1 ? kFALSE : kTRUE; }

   //--- load desired plugin and setup conection to GRID
   static TGrid *Connect(const char *grid, const char *uid = 0,
                         const char *pw = 0);

   ClassDef(TGrid,0)  // ABC defining interface to GRID services
};

#endif
