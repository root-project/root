// @(#) root/glite:$Id$
// Author: Anar Manafov <A.Manafov@gsi.de> 2006-03-20

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/************************************************************************/
/*! \file TGLite.h
Interface of the class which
defines interface to gLite GRID services. *//*

         version number:    $LastChangedRevision: 1678 $
         created by:        Anar Manafov
                            2006-03-20
         last changed by:   $LastChangedBy: manafov $ $LastChangedDate: 2008-01-21 18:22:14 +0100 (Mon, 21 Jan 2008) $

         Copyright (c) 2006-2008 GSI GridTeam. All rights reserved.
*************************************************************************/

#ifndef ROOT_TGLite
#define ROOT_TGLite

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLite                                                               //
//                                                                      //
// Class defining interface to gLite GRID services.                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

class TGLite: public TGrid
{
public:
   TGLite(const char *_gridurl, const char* /*_uid*/ = NULL, const char* /*_passwd*/ = NULL, const char* /*_options*/ = NULL);
   virtual ~TGLite();

public:
   virtual Bool_t IsConnected() const;

   virtual void Shell();
   virtual void Stdout();
   virtual void Stderr();

   virtual TGridResult* Command(const char* /*command*/, Bool_t /*interactive*/ = kFALSE, UInt_t /*stream*/ = 2);
   virtual TGridResult* Query(const char *_path, const char *_pattern = NULL, const char* /*conditions*/ = "", const char* /*options*/ = "");
   virtual TGridResult* LocateSites();

   //--- Catalog Interface
   virtual TGridResult* Ls(const char *_ldn = "", Option_t* /*options*/ = "", Bool_t /*verbose*/ = kFALSE);
   virtual const char* Pwd(Bool_t /*verbose*/ = kFALSE);
   virtual Bool_t Cd(const char *_ldn = "", Bool_t /*verbose*/ = kFALSE);
   virtual Int_t  Mkdir(const char *_ldn = "", Option_t* /*options*/ = "", Bool_t /*verbose*/ = kFALSE);
   virtual Bool_t Rmdir(const char *_ldn = "", Option_t* /*options*/ = "", Bool_t /*verbose*/ = kFALSE);
   virtual Bool_t Register(const char *_lfn, const char *_turl , Long_t /*size*/ = -1, const char *_se = 0, const char *_guid = 0, Bool_t /*verbose*/ = kFALSE);
   virtual Bool_t Rm(const char *_lfn, Option_t* /*option*/ = "", Bool_t /*verbose*/ = kFALSE);

   //--- Job Submission Interface
   virtual TGridJob* Submit(const char *_jdl);
   virtual TGridJDL* GetJDLGenerator();
   virtual Bool_t Kill(TGridJob *_gridjob);
   virtual Bool_t KillById(TString _id);

private:
   std::string fFileCatalog_WrkDir;

   ClassDef(TGLite, 1) // Interface to gLite Grid services
};

#endif
