// @(#)root/graf:$Id$
//  Author: Valeriy Onuchin   23/06/05

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TASPluginGS - allows to read PS/EPS/PDF files via GhostScript        //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TASPluginGS.h"
#include "TSystem.h"

#ifndef WIN32
#   include <X11/Xlib.h>
#   define popen_flags "r"
#else
#   include "Windows4root.h"
#   define popen_flags "rb"
#endif

extern "C" {
#ifndef WIN32
#   include <afterbase.h>
#else
#   include <win32/config.h>
#   include <win32/afterbase.h>
#   define X_DISPLAY_MISSING 1
#endif
#   include <import.h>
}


ClassImp(TASPluginGS)


//______________________________________________________________________________
TASPluginGS::TASPluginGS(const char *ext) : TASImagePlugin(ext)
{
   // ctor

#ifndef WIN32
   fInterpreter = gSystem->Which(gSystem->Getenv("PATH"), "gs", kExecutePermission);
#else
   fInterpreter = gSystem->Which(gSystem->Getenv("PATH"), "gswin32c.exe", kExecutePermission);
   if (fInterpreter) {
      // which returned path may include blanks, like "Program Files" which popen does not like
      delete [] fInterpreter;
      fInterpreter = StrDup("gswin32c.exe");
   }
#endif
}

//______________________________________________________________________________
TASPluginGS::~TASPluginGS()
{
   // dtor

   delete [] fInterpreter;
   fInterpreter = 0;
}

//______________________________________________________________________________
ASImage *TASPluginGS::File2ASImage(const char *filename)
{
   // read PS/EPS/PDF file and convert it to ASImage

   if (!fInterpreter) {
      Warning("File2ASImage", "GhostScript is not available");
      return 0;
   }

   if (gSystem->AccessPathName(filename)) {
      Warning("File2ASImage", "input file %s is not accessible", filename);
      return 0;
   }

   TString ext = (strrchr(filename, '.') + 1);
   ext.Strip();
   ext.ToLower();

   UInt_t width = 0;
   UInt_t height = 0;
   Bool_t eps = kFALSE;

   if (ext == "eps") {
      eps = kTRUE;
      FILE *fd = fopen(filename, "r");
      if (!fd) {
         Warning("File2ASImage", "input file %s is not readable", filename);
         return 0;
      }

      do {
         char buf[128];
         TString line = fgets(buf, 128, fd);
         if (line.IsNull() || !line.BeginsWith("%")) break;

         if (line.BeginsWith("%%BoundingBox:")) {
            int lx, ly, ux, uy;
            line = line(14, line.Length());
            sscanf(line.Data(), "%d %d %d %d", &lx, &ly, &ux, &uy);
            width = TMath::Abs(ux - lx);
            height = TMath::Abs(uy - ly);
            break;
         }
      } while (!feof(fd));

      fclose(fd);
   }

   // command line to execute
   TString cmd = fInterpreter;
   if (eps) {
      cmd += Form(" -g%dx%d", width, height);
   }
   cmd += " -dSAFER -dBATCH -dNOPAUSE -dQUIET -sDEVICE=png16m -dGraphicsAlphaBits=4 -sOutputFile=- ";
   cmd += filename;
   FILE *in = gSystem->OpenPipe(cmd.Data(), popen_flags);

   if (!in) {
      return 0;
   }

   const UInt_t kBuffLength = 32768;
   static char buf[kBuffLength];
   TString raw;

   do {
      Long_t r = fread(&buf, 1, kBuffLength, in);
      raw.Append((const char*)&buf, r);
   } while (!feof(in));

   gSystem->ClosePipe(in);

   ASImageImportParams params;
   params.flags =  0;
   params.width = width;
   params.height = height;
   params.filter = SCL_DO_ALL;
   params.gamma = 0;
   params.gamma_table = 0;
   params.compression = 0;
   params.format = ASA_ASImage;
   params.search_path = 0;
   params.subimage = 0;

   ASImage *ret = PNGBuff2ASimage((CARD8 *)raw.Data(), &params);
   return ret;
}
