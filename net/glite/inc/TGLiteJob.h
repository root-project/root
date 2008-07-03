// @(#) root/glite:$Id$
// Author: Anar Manafov <A.Manafov@gsi.de> 2006-04-10

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/************************************************************************/
/*! \file TGLiteJob.h
gLite implementation of TGridJob*//*

         version number:    $LastChangedRevision: 1678 $
         created by:        Anar Manafov
                            2006-04-10
         last changed by:   $LastChangedBy: manafov $ $LastChangedDate: 2008-01-21 18:22:14 +0100 (Mon, 21 Jan 2008) $

         Copyright (c) 2006-2008 GSI GridTeam. All rights reserved.
*************************************************************************/

#ifndef ROOT_TGLiteJob
#define ROOT_TGLiteJob

#ifndef ROOT_TGridJob
#include "TGridJob.h"
#endif

class TGLiteJob : public TGridJob
{

public:
   TGLiteJob(TString jobID) : TGridJob(jobID) {}
   virtual ~TGLiteJob() {}

   virtual TGridJobStatus* GetJobStatus() const;
   Int_t GetOutputSandbox(const char *_localpath, Option_t* /*opt*/ = 0);
   virtual Bool_t Resubmit();
   virtual Bool_t Cancel();

   ClassDef(TGLiteJob, 1) // gLite implementation of TGridJob
};

#endif
