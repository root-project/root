// @(#)root/graf:$Name:  $:$Id: TASPluginGS.cxx,v 1.6 2005/06/14 15:29:06 rdm Exp $
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
#include "afterimage.h"


ClassImp(TASPluginGS)


//______________________________________________________________________________
TASPluginGS::TASPluginGS(const char *ext) : TASImagePlugin(ext)
{
   // ctor

   fInterpreter = gSystem->Which(gSystem->Getenv("PATH"), "gs", kExecutePermission);   
}

//______________________________________________________________________________
TASPluginGS::~TASPluginGS()
{
   // dtor

   delete fInterpreter;
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

   // command line to execute
   TString cmd = fInterpreter;
   cmd += " -dSAFER -dBATCH -dNOPAUSE -dQUIET -sDEVICE=png16m -dGraphicsAlphaBits=4 -sOutputFile=- ";
   cmd += filename;
   FILE *in = gSystem->OpenPipe(cmd.Data(), "r");

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
   params.flags = 0;
   params.width = 0;
   params.height = 0 ;
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


