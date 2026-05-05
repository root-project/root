// @(#)root/graf:$Id$
//  Author: Valeriy Onuchin   23/06/05

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TASPluginGS
\ingroup asimage

Allows to read PS/EPS/PDF files via GhostScript
*/

#include "TASPluginGS.h"
#include "TSystem.h"
#include "RConfigure.h"

#ifdef R__HAS_COCOA
#   define X_DISPLAY_MISSING 1
#   define popen_flags "r"
#elif defined (WIN32)
#   include "Windows4root.h"
#   define popen_flags "rb"
#else
#   include <X11/Xlib.h>
#   define popen_flags "r"
#endif

#ifndef WIN32
#   include <afterbase.h>
#else
#   define X_DISPLAY_MISSING 1
#   include <afterbase.h>
#endif
#   include <import.h>



////////////////////////////////////////////////////////////////////////////////
/// ctor

TASPluginGS::TASPluginGS(const char *ext) : TASImagePlugin(ext)
{
#ifndef WIN32
   fGsExe = "gs";
   gSystem->FindFile(gSystem->Getenv("PATH"), fGsExe, kExecutePermission);
#else
   fGsExe = "gswin32c.exe";
   // FindFile returned path may include blanks, like "Program Files" which popen does not like
   // Therefore if executable found in defined paths, just use it name as is
   if (gSystem->FindFile(gSystem->Getenv("PATH"), fGsExe, kExecutePermission))
      fGsExe = "gswin32c.exe";
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TASPluginGS::~TASPluginGS()
{
   ROOT::CallRecursiveRemoveIfNeeded(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// read PS/EPS/PDF file and convert it to ASImage

ASImage *TASPluginGS::File2ASImage(const char *filename)
{
   if (fGsExe.IsNull()) {
      Warning("File2ASImage", "GhostScript is not available");
      return nullptr;
   }

   if (gSystem->AccessPathName(filename)) {
      Warning("File2ASImage", "input file %s is not accessible", filename);
      return nullptr;
   }

   const char *ppos = strrchr(filename, '.');

   TString ext;
   if (ppos) {
      ext = ppos + 1;
      ext.Strip();
      ext.ToLower();
   }

   UInt_t width = 0, height = 0;
   Bool_t eps = ext == "eps";

   if (eps) {
      FILE *fd = fopen(filename, "r");
      if (!fd) {
         Warning("File2ASImage", "input file %s is not readable", filename);
         return nullptr;
      }

      do {
         char buf[128];
         TString line = fgets(buf, 128, fd);
         if (line.IsNull() || !line.BeginsWith("%"))
            break;

         if (line.BeginsWith("%%BoundingBox:")) {
            int lx, ly, ux, uy;
            line = line(14, line.Length());
            sscanf(line.Data(), "%d %d %d %d", &lx, &ly, &ux, &uy);
            width = std::abs(ux - lx);
            height = std::abs(uy - ly);
            break;
         }
      } while (!feof(fd));

      fclose(fd);
   }

   // command line to execute
   TString cmd = fGsExe;
   if (eps)
      cmd += TString::Format(" -g%dx%d", width, height);
   cmd += " -dSAFER -dBATCH -dNOPAUSE -dQUIET -sDEVICE=png16m -dGraphicsAlphaBits=4 -sOutputFile=- ";
   cmd += filename;
   FILE *in = gSystem->OpenPipe(cmd.Data(), popen_flags);

   if (!in)
      return nullptr;

   const UInt_t kBuffLength = 32768;
   char buf[kBuffLength];
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
   params.gamma_table = nullptr;
   params.compression = 0;
   params.format = ASA_ASImage;
   params.search_path = nullptr;
   params.subimage = 0;

   ASImage *ret = PNGBuff2ASimage((CARD8 *)raw.Data(), &params);
   return ret;
}
