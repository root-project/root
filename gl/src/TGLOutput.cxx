// @(#)root/gl:$Name:  $:$Id: TGLOutput.cxx,v 1.4 2005/10/20 08:22:32 couet Exp $
// Author:  Richard Maunder, Olivier Couet  02/07/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLOutput.h"
#include "TGLViewer.h"
#include "TSystem.h" // For gSystem
#include "gl2ps.h"
#include "TError.h"
#include "Riostream.h"
#include <assert.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOutput                                                            //
//                                                                      //
// Wrapper class for GL capture & output routines                       //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLOutput)

//______________________________________________________________________________
Bool_t TGLOutput::Capture(TGLViewer & viewer, EFormat format, const char * filePath)
{
   // Capture viewer to file. Arguments are:
   // 'viewer' - viewer object to capture from
   // 'format' - output format - only postscript types presently.
   //            One of kEPS_SIMPLE, kEPS_BSP, kPDF_SIMPLE or kPDF_BSP
   //             See TGLOutput::CapturePostscript() for meanings
   // 'filePath' - file output name. If null defaults to './viewer.eps' or './viewer.pdf'
   // depending on format requested
   // 
   // Note : Output files can be large and take considerable time (up to mins)
   // to generate
   switch(format) {
      case(kEPS_SIMPLE):
      case(kEPS_BSP):
      case(kPDF_SIMPLE):
      case(kPDF_BSP): {
         return CapturePostscript(viewer, format, filePath);
      }
   }

   assert(kFALSE);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLOutput::CapturePostscript(TGLViewer & viewer, EFormat format, const char * filePath)
{
   // Capture viewer to postscript file. Arguments are:
   // 'viewer' - viewer object to capture from
   // 'format' - output format
   //                kEPS_SIMPLE - lower quality EPS
   //                kEPS_BSP    - higher quality EPS
   //                kPDF_SIMPLE - lower quality PDF
   //                kPDF_BSP    - higher quality PDF
   // 'filePath' - file output name. If null defaults to './viewer.eps' or './viewer.pdf'
   // depending on format requested
   if (!filePath || strlen(filePath) == 0) {
      if (format == kEPS_SIMPLE || format == kEPS_BSP) {
         filePath = "viewer.eps";
      } else if (format == kPDF_SIMPLE || format == kPDF_BSP) {
         filePath = "viewer.pdf";
      }
   }
   Info("TGLOutput::Postscript()", "Start creating %s.", filePath);
   std::cout << "Please wait.";

   if (FILE *output = fopen (filePath, "w+b"))
   {
      Int_t gl2psFormat;
      Int_t gl2psSort;

      switch(format) {
         case kEPS_SIMPLE:
            gl2psFormat = GL2PS_EPS;
            gl2psSort = GL2PS_SIMPLE_SORT;
            break;
         case kEPS_BSP:
            gl2psFormat = GL2PS_EPS;
            gl2psSort = GL2PS_BSP_SORT;
            break;
         case kPDF_SIMPLE:
            gl2psFormat = GL2PS_PDF;
            gl2psSort = GL2PS_SIMPLE_SORT;
            break;
         case kPDF_BSP:
            gl2psFormat = GL2PS_PDF;
            gl2psSort = GL2PS_BSP_SORT;
            break;
         default:
            assert(kFALSE);
            return kFALSE;
      }
      Int_t buffsize = 0, state = GL2PS_OVERFLOW;
      viewer.DoDraw();
      viewer.fIsPrinting = kTRUE;
      while (state == GL2PS_OVERFLOW) {
         buffsize += 1024*1024;
         gl2psBeginPage ("ROOT Scene Graph", "ROOT", NULL,
         gl2psFormat, gl2psSort, GL2PS_USE_CURRENT_VIEWPORT
         | GL2PS_POLYGON_OFFSET_FILL | GL2PS_SILENT
         | GL2PS_BEST_ROOT | GL2PS_OCCLUSION_CULL
         | 0,
         GL_RGBA, 0, NULL,0, 0, 0,
         buffsize, output, NULL);
         viewer.DoDraw();
         state = gl2psEndPage();
         std::cout << ".";
      }
      std::cout << std::endl;
      fclose (output);
      viewer.fIsPrinting = kFALSE;
      if (!gSystem->AccessPathName(filePath)) {
         Info("TGLOutput::Postscript", "Finished creating %s.", filePath);
         return kTRUE;
      }
   } else {
      Error("TGLOutput::Postscript", "Failed to create %s. ", filePath);
   }

   return kFALSE;
}
