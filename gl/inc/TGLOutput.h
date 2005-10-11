// @(#)root/gl:$Name:  $:$Id: TGLOutput.h,v 1.3 2005/10/03 15:19:35 brun Exp $
// Author:  Richard Maunder, Olivier Couet  02/07/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLOutput
#define ROOT_TGLOutput

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TGLViewer;

class TGLOutput {
public:
   enum EFormat { kEPS_SIMPLE, kEPS_BSP, kPDF_SIMPLE, kPDF_BSP };
   static Bool_t Capture(TGLViewer & viewer, EFormat format,
                         const char * filePath = 0);
   virtual ~TGLOutput() { }

private:
   static Bool_t CapturePostscript(TGLViewer & viewer, EFormat format,
                                   const char * filePath);

   ClassDef(TGLOutput,0) // Collection of GL capture & output routines
};

#endif
