// @(#)root/gl:$Id$
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOutput                                                            //
//                                                                      //
// Wrapper class for GL capture & output routines                       //
//////////////////////////////////////////////////////////////////////////

class TGLOutput {
public:
   enum EFormat { kEPS_SIMPLE, kEPS_BSP, kPDF_SIMPLE, kPDF_BSP };
   static Bool_t Capture(TGLViewer & viewer, EFormat format,
                         const char * filePath = 0);

   static void   StartEmbeddedPS();//gl to ps in pad, "embedded" does not mean eps format here
   static void   CloseEmbeddedPS();
   static void   Capture(TGLViewer & viewer);

   virtual ~TGLOutput() { }

private:
   static Bool_t CapturePostscript(TGLViewer & viewer, EFormat format,
                                   const char * filePath);

   ClassDef(TGLOutput,0) // Wrapper class for GL capture & output routines
};

#endif
