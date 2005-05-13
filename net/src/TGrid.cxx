// @(#)root/net:$Name:  $:$Id: TGrid.cxx,v 1.7 2005/05/12 13:19:39 rdm Exp $
// Author: Fons Rademakers   3/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGrid                                                                //
//                                                                      //
// Abstract base class defining interface to common GRID services.      //
//                                                                      //
// To open a connection to a GRID use the static method Connect().      //
// The argument of Connect() is of the form:                            //
//    <grid>[://<host>][:<port>], e.g.                                  //
// alien, alien://alice.cern.ch, globus://glsvr1.cern.ch, ...           //
// Depending on the <grid> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
//                                                                      //
// Related classes are TGridResult.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGrid.h"
#include "TUrl.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TFile.h"
#include "TError.h"

TGrid *gGrid = 0;


ClassImp(TGrid)

//______________________________________________________________________________
TGrid *TGrid::Connect(const char *grid, const char *uid, const char *pw,
                      const char *options)
{
   // The grid should be of the form:  <grid>://<host>[:<port>],
   // e.g.:  alien://alice.cern.ch, globus://glsrv1.cern.ch, ...
   // The uid is the username and pw the password that should be used for
   // the connection. Depending on the <grid> the shared library (plugin)
   // for the selected system will be loaded. When the connection could not
   // be opened 0 is returned. For AliEn the supported options are:
   // -domain=<domain name>
   // -debug=<debug level from 1 to 10>
   // Example: "-domain=cern.ch -debug=5"

   TPluginHandler *h;
   TGrid *g = 0;

   if (!grid) {
      ::Error("TGrid::Connect", "no grid specified");
      return 0;
   }
   if (!uid)
      uid = "";
   if (!pw)
      pw = "";
   if (!options)
      options = "";

   if ((h = gROOT->GetPluginManager()->FindHandler("TGrid", grid))) {
      if (h->LoadPlugin() == -1)
         return 0;
      g = (TGrid *) h->ExecPlugin(4, grid, uid, pw, options);
   }

   return g;
}

//______________________________________________________________________________
TGrid::~TGrid()
{
   // cleanup
}

//______________________________________________________________________________
void TGrid::PrintProgress(Long_t bytesread, Long_t size)
{
   // Print file copy progress.

   fprintf(stderr, "[xrootd] Total %.02f MB\t|", (float)size/1024/1024);
   for (int l = 0; l < 20; l++) {
      if (l < ((int)(20.0*bytesread/size)))
         fprintf(stderr, "=");
      if (l == ((int)(20.0*bytesread/size)))
         fprintf(stderr, ">");
      if (l > ((int)(20.0*bytesread/size)))
         fprintf(stderr, ".");
   }

   fWatch.Stop();
   float lCopy_time = fWatch.RealTime();
   fprintf(stderr, "| %.02f %% [%.01f Mb/s]\r", 100.0*bytesread/size,bytesread/lCopy_time/1000.0/1000.0);
   fWatch.Continue();
}

//______________________________________________________________________________
Bool_t TGrid::Cp(const char *src, const char *dst, Bool_t progressbar,
                 UInt_t buffersize)
{
   // Allows to copy file from src to dst URL.

   Bool_t success= kFALSE;
   TUrl sURL(src);
   TUrl dURL(dst);
   char* copybuffer=0;

   TFile* sfile=0;
   TFile* dfile=0;

   sfile = TFile::Open(src,"-READ");

   if (!sfile) {
      Error("Cp", "cannot open source file %s",src);
      goto Copyout;
   }

   dfile = TFile::Open(dst,"-RECREATE");

   if (!dfile) {
      Error("Cp", "cannot open destination file %s",dst);
      goto Copyout;
   }

   sfile->Seek(0);
   dfile->Seek(0);

   copybuffer = new char[buffersize];
   if (!copybuffer) {
      Error("Cp", "cannot allocate the copy buffer");
      goto Copyout;
   }

   Int_t read;
   Int_t written;
   Bool_t readop;
   Bool_t writeop;
   Long_t totalread;
   Long_t filesize;
   filesize = sfile->GetSize();
   totalread=0;
   fWatch.Start();

   do {
      if (progressbar) PrintProgress(totalread,filesize);

      Double_t b0=sfile->GetBytesRead();
      Int_t readsize;
      if ((filesize-(Int_t)b0) > (Int_t)buffersize) {
         readsize = buffersize;
      } else {
         readsize = (filesize-(Int_t)b0);
      }

      readop = sfile->ReadBuffer(copybuffer, readsize);
      read = Int_t(sfile->GetBytesRead()-b0);
      if (read < 0) {
         Error("Cp", "cannot read from source file %s",src);
         goto Copyout;
      }

      Double_t w0=dfile->GetBytesWritten();
      writeop= dfile->WriteBuffer(copybuffer, read);
      written = Int_t(dfile->GetBytesWritten()-w0);
      if (written != read) {
         Error("Cp", "cannot write %d bytes to destination file %s",read,dst);
         goto Copyout;
      }
      totalread+=read;
   } while (read == (Int_t)buffersize);

   if (progressbar) {
      PrintProgress(totalread,filesize);
      fprintf(stderr,"\n");
   }

   success = kTRUE;

Copyout:
   if (sfile) sfile->Close("NOROOT");
   if (dfile) dfile->Close("NOROOT");

   if (sfile) delete sfile;
   if (dfile) delete dfile;
   if (copybuffer) delete copybuffer;
   fWatch.Stop();
   fWatch.Reset();

   return success;
}
