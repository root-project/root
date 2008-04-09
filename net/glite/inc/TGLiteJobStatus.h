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
/*! \file TGLiteJobStatus.h
gLite implementation of TGridJobStatus*//*

         version number:    $LastChangedRevision: 1678 $
         created by:        Anar Manafov
                            2006-04-10
         last changed by:   $LastChangedBy: manafov $ $LastChangedDate: 2008-01-21 18:22:14 +0100 (Mon, 21 Jan 2008) $

         Copyright (c) 2006-2008 GSI GridTeam. All rights reserved.
*************************************************************************/

#ifndef ROOT_TGLiteJobStatus
#define ROOT_TGLiteJobStatus

#ifndef ROOT_TGridJobStatus
#include "TGridJobStatus.h"
#endif

#ifndef ROOT_TGridJob
#include "TGridJob.h"
#endif

class TGLiteJobStatus : public TGridJobStatus
{
public:
   TGLiteJobStatus() {}
   TGLiteJobStatus(TString jobID);
   virtual ~TGLiteJobStatus() {}

public:
   virtual EGridJobStatus GetStatus() const;

private:
   std::string m_sJobID;

   ClassDef(TGLiteJobStatus, 1) // gLite implementation of TGridJobStatus
};

#endif
