// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// ************************************************************************* //
// *                                                                       * //
// *                        p q 2 a c t i o n s                            * //
// *                                                                       * //
// * This file implements the action functions used in PQ2                 * //
// *                                                                       * //
// ************************************************************************* //

#include <errno.h>

#include "pq2actions.h"
#include "pq2wrappers.h"
#include "redirguard.h"

#include "TFile.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "THashList.h"
#include "TH1D.h"
#include "TParameter.h"
#include "TRegexp.h"
#include "TString.h"
#include "TSystem.h"

// Global variables defined by other PQ2 components
extern TString flog;
extern TString ferr;
extern TString fres;
extern Int_t gverbose;

// Local globals
static const char *glabMet[] = { "#files", "size" };
THashList gProtoPortMap;
// Local functions
int do_anadist_ds(TFileCollection *fc, const char *newsrvs = 0, const char *ignsrvs = 0,
                  const char *excsrvs = 0, int met = 0, const char *fnout = 0,
                  TList *distinfo = 0, const char *outfile = 0, const char *infile = 0);
void do_anadist_getkey(const char *p, TString &key);
void do_anadist_getkey(TUrl *u, TString &key);

// Usefule macros
#define SDELTWO(x,y)  { SafeDelete(x); SafeDelete(y); }
#define SDELTRE(x,y,z)  { SafeDelete(x); SafeDelete(y); SafeDelete(z); }
#define SDELETE(x,y,z,w,t)  { SafeDelete(x); SafeDelete(y); SafeDelete(z); SafeDelete(w); SafeDelete(t); }

//_______________________________________________________________________________________
void do_cache(bool clear, const char *ds)
{
   // Execute 'cache'

   // Show / Clear the cache
   DataSetCache(clear, ds);
   // Remove the error file
   gSystem->Unlink(ferr.Data());
}

//_______________________________________________________________________________________
void do_ls(const char *ds, const char *opt)
{
   // Execute 'ls'

   // List the data sets
   ShowDataSets(ds, opt);
   // Remove the error file
   gSystem->Unlink(ferr.Data());
}

//_______________________________________________________________________________________
void do_ls_files_server(const char *ds, const char *server)
{
   // Execute 'ls-files'

   const char *action = (server && strlen(server) > 0) ? "pq2-ls-files-server" : "pq2-ls-files";

   // We need to scan all the datasets to find the matching ones ...
   TFileCollection *fc = 0;
   { redirguard rog(flog.Data(), "a", gverbose);
      if (server && strlen(server) > 0) {
         fc = GetDataSet(ds, server);
      } else {
         fc = GetDataSet(ds);
      }
   }
   if (!fc) {
      // Notify
      gSystem->Rename(flog.Data(), ferr.Data());
      Printf("%s: ERROR: problems retrieving info about dataset '%s'", action, ds);
      return;
   }

   // Overall info
   if (server && strlen(server) > 0) {
      Printf("%s: dataset '%s' has %d files on server: %s",
                                              action, ds, fc->GetList()->GetSize(), server);
   } else {
      Printf("%s: dataset '%s' has %d files", action, ds, fc->GetList()->GetSize());
   }

   // Header
   TString num("      #   ");
   TString nam("File"); nam.Resize(80);
   TString siz("        Size");
   TString met("#Objs Obj|Type|Entries, ...");
   TString header;
   header.Form("%s%s%s   %s", num.Data(), nam.Data(), siz.Data(), met.Data());

   // Iterate
   const char *unit[4] = {"kB", "MB", "GB", "TB"};
   TString uu, meta, name;
   TIter nxf(fc->GetList());
   TFileInfo *fi = 0;
   Int_t nf = 0;
   while ((fi = (TFileInfo *) nxf())) {
      nf++;
      if (nf == 1)
         Printf("%s:%s", action, header.Data());
      // URL
      uu = fi->GetCurrentUrl()->GetUrl();
      if (uu.Length() < 80) uu.Resize(80);
      // Size renormalize to kB, MB or GB
      Int_t k = 0;
      Long64_t refsz = 1024;
      Long64_t xsz = (Long64_t) (fi->GetSize() / refsz);
      while (xsz > 1024 && k < 3) {
         k++;
         refsz *= 1024;
         xsz = (Long64_t) (fi->GetSize() / refsz);
      }
      // Meta info
      meta = "";
      if (fi->GetMetaDataList()) {
         meta.Form("  %d  ", fi->GetMetaDataList()->GetSize());
         Bool_t firstObj = kTRUE;
         TIter nxm(fi->GetMetaDataList());
         TFileInfoMeta *fim = 0;
         while ((fim = (TFileInfoMeta *) nxm())) {
            if (!firstObj) meta += ",";
            name = fim->GetObject();
            if (strcmp(fim->GetDirectory(),"/")) name = fim->GetName();
            meta += Form("%s|%s|%lld", name.Data(), fim->GetClass(), fim->GetEntries());
            firstObj = kFALSE;
         }
      }
      // Printout
      if (xsz > 0) {
         Printf("%s:  %5d   %s %8lld %s    %s", action, nf, uu.Data(), xsz, unit[k], meta.Data());
      } else {
         Printf("%s:  %5d   %s         N/A    N/A", action, nf, uu.Data());
      }
   }
}

//______________________________________________________________________________
void printDataSet(TFileCollection *fc, Int_t popt)
{
   // Formatted printout of the content of TFileCollection 'fc'.
   // Options in the form
   //           popt = u * 10 + f
   //     f    0 => header only, 1 => header + files
   //   when printing files
   //     u    0 => print file name only, 1 => print full URL

   if (!fc) return;

   Int_t f = popt%10;
   Int_t u = popt - 10 * f;

   Printf("+++");
   if (fc->GetTitle() && (strlen(fc->GetTitle()) > 0)) {
      Printf("+++ Dumping: %s: ", fc->GetTitle());
   } else {
      Printf("+++ Dumping: %s: ", fc->GetName());
   }
   Printf("%s", fc->ExportInfo("+++ Summary:", 1)->GetName());
   if (f == 1) {
      Printf("+++ Files:");
      Int_t nf = 0;
      TIter nxfi(fc->GetList());
      TFileInfo *fi = 0;
      while ((fi = (TFileInfo *)nxfi())) {
         if (u == 1)
            Printf("+++ %5d. %s", ++nf, fi->GetCurrentUrl()->GetUrl());
         else
            Printf("+++ %5d. %s", ++nf, fi->GetCurrentUrl()->GetFile());
      }
   }
   Printf("+++");
}

//_______________________________________________________________________________________
void do_info_server(const char *server)
{
   // Execute 'info-server'

   const char *action = "pq2-info-server";

   // We need to scan all the datasets to find the matching ones ...
   TMap *dsmap = 0;
   {  redirguard rog(flog.Data(), "a", gverbose);
      dsmap = GetDataSets("/*/*", server);
   }
   if (!dsmap) {
      // Notify
      gSystem->Rename(flog.Data(), ferr.Data());
      Printf("%s: ERROR: problems retrieving info about datasets for server '%s'", action, server);
      return;
   }

   redirguard rog(fres.Data(), "w", gverbose);
   Int_t popt = 0;
   TIter nxk(dsmap);
   TObject *k = 0;
   TFileCollection *fc = 0;
   while ((k = nxk()) && (fc = (TFileCollection *) dsmap->GetValue(k))) {
      printDataSet(fc, popt);
   }
   delete dsmap;

   // Done
   return;
}

//_______________________________________________________________________________________
Int_t pq2register(const char *dsname, const char *files, const char *opt) {

   // If the dataset exists already do not continue
   TString oo(opt);
   oo.ToUpper();
   redirguard rog(flog.Data(), "a", gverbose);
   if (ExistsDataSet(dsname) &&
       !oo.Contains("O") && !oo.Contains("U")) {
      return 2;
   }
   // Create the file collection
   TFileCollection *fc = new TFileCollection("dum", "dum", files);

   // The option may contain the default tree name and/or the staged status
   Int_t itb = kNPOS, ite = kNPOS;
   TString o(opt), deftree;
   if ((itb = o.Index("tree:")) != kNPOS) {
      deftree = o(itb + 5, o.Length());
      if ((ite = deftree.Index('|')) != kNPOS) deftree.Remove(ite);
      o.ReplaceAll(TString::Format("tree:%s|", deftree.Data()), "");
      if (!deftree.BeginsWith("/")) deftree.Insert(0, "/");
      if (!deftree.IsNull()) fc->SetDefaultTreeName(deftree);
   }
   if (o.Contains("staged|")) {
      fc->SetBitAll(TFileInfo::kStaged);
      o.ReplaceAll("staged|", "");
   }
   // Update the collection
   fc->Update();

   // Register the file collection
   Int_t rc =0;
   if (RegisterDataSet(dsname, fc, o) == 0) rc = 1;
   // Cleanup
   delete fc;

   // Done
   return rc;
}

//_______________________________________________________________________________________
void do_put(const char *files, const char *opt)
{
   // Execute 'put'

   const char *action = "pq2-put";

   // Check the file path makes sense
   if (!files || strlen(files) <= 0) {
      // Notify
      gSystem->Rename(flog.Data(), ferr.Data());
      Printf("%s: ERROR: path files not defined!", action);
      return ;
   }

   // Find out if it is a single file or a directory or contains a wildcard
   Bool_t isDir = kFALSE;
   Bool_t isWild = kFALSE;
   TString dir, base;
   FileStat_t st;
   if (gSystem->GetPathInfo(files, st) != 0) {
      // Path does not exists: check the basename for wild cards; in such a case
      // we have to do a selective search in the directory
      base = gSystem->BaseName(files);
      if (base.Contains("*")) {
         isWild = kTRUE;
         base.ReplaceAll("*", ".*");
         isDir = kTRUE;
         dir = gSystem->DirName(files);
      }

   } else {
      // Path exists: is it a dir or a file ?
      if (R_ISDIR(st.fMode)) {
         isDir = kTRUE;
         dir = files;
      }
   }

   Int_t ndp = 0, nd = 0;
   Int_t printerr = 1;
   Int_t rc = 0;
   // If simple file ...
   if (!isDir) {
      nd++;
      // ... just register and exit
      TString dsname = gSystem->BaseName(files);
      if ((rc = pq2register(dsname.Data(), files, opt)) != 0) {
         // Notify
         gSystem->Rename(flog.Data(), ferr.Data());
         if (rc == 2) {
            Printf("%s: WARNING: dataset '%s' already exists - ignoring request", action, dsname.Data());
            Printf("%s:          (use '-o O' to overwrite,  '-o U' to update)", action);
         } else {
            Printf("%s: ERROR: problems registering '%s' from '%s'", action, dsname.Data(), files);
         }
         return;
      }
      printerr = 0;
   } else {
      // ... else, scan the directory
      void *dirp = gSystem->OpenDirectory(dir.Data());
      if (!dirp) {
         // Notify
         gSystem->Rename(flog.Data(), ferr.Data());
         Printf("%s: ERROR: could not open directory '%s'", action, dir.Data());
         return;
      }
      printerr = 0;
      // Loop over the entries
      TString file;
      TRegexp reg(base);
      const char *ent = 0;
      while ((ent = gSystem->GetDirEntry(dirp))) {
         // Skip default entries
         if (!strcmp(ent, ".") || !strcmp(ent, "..")) continue;
         if (isWild) {
            file = ent;
            if (file.Index(reg) == kNPOS) continue;
         }
         nd++;
         file.Form("%s/%s", dir.Data(), ent);
         if ((rc = pq2register(ent, file.Data(), opt)) != 0) {
            nd--;
            ndp++;
            printerr = 1;
            // Notify
            if (rc == 1) {
               Printf("%s: ERROR: problems registering '%s' from '%s'", action, ent, file.Data());
            } else {
               Printf("%s: WARNING: dataset '%s' already exists - ignoring request", action, ent);
               Printf("%s:          (use '-o O' to overwrite,  '-o U' to update)", action);
            }
            continue;
         }
      }
      gSystem->FreeDirectory(dirp);
   }

   // If no match, notify
   if (printerr == 1) {
      if (ndp > 0)
         Printf("%s: WARNING: problems with %d dataset(s)", action, ndp);
      else
         Printf("%s: WARNING: some problems occured", action);
      gSystem->Rename(flog.Data(), ferr.Data());
   }
   Printf("%s: %d dataset(s) registered", action, nd);

   // Done
   return;
}

//_______________________________________________________________________________________
void do_rm(const char *dsname)
{
   // Execute 'rm'

   const char *action = "pq2-ds";

   Int_t nd = 0;
   Int_t printerr = 1;
   TString ds(dsname);
   if (!ds.Contains("*")) {
      nd++;
      // Remove the dataset
      redirguard rog(flog.Data(), "a", gverbose);
      if (RemoveDataSet(dsname) != 0) {
         // Notify
         gSystem->Rename(flog.Data(), ferr.Data());
         Printf("%s: ERROR: problems removing dataset '%s'", action, dsname);
         return;
      }
      printerr = 0;
   } else {
      // We need to scan all the datasets to find the matching ones ...
      TMap *dss = 0;
      {  redirguard rog(flog.Data(), "a", gverbose);
         dss = GetDataSets();
      }
      if (!dss) {
         // Notify
         gSystem->Rename(flog.Data(), ferr.Data());
         Printf("%s: ERROR: problems retrieving info about datasets", action);
         return;
      }
      printerr = 0;
      // Iterate
      TRegexp reg(dsname, kTRUE);
      TIter nxd(dss);
      TObjString *os = 0;
      while ((os = dynamic_cast<TObjString*>(nxd()))) {
         ds = os->GetName();
         if (ds.Index(reg) != kNPOS) {
            nd++;
            // Remove the dataset
            redirguard rog(flog.Data(), "a", gverbose);
            if (RemoveDataSet(ds.Data()) != 0) {
               printerr = 1;
               // Notify
               Printf("%s: ERROR: problems removing dataset '%s'", action, ds.Data());
               continue;
            }
         }
      }

   }

   // If no match, notify
   if (nd == 0) {
      Printf("%s: WARNING: no matching dataset found!", action);
   } else {
      Printf("%s: %d dataset(s) removed", action, nd);
   }
   if (printerr)
      gSystem->Rename(flog.Data(), ferr.Data());

   // Done
   return;
}

//_______________________________________________________________________________________
int do_verify(const char *dsname, const char *opt, const char *redir)
{
   // Execute 'verify'

   const char *action = "pq2-verify";

   Int_t nd = 0, rc = -1;
   Int_t printerr = 1;
   TString ds(dsname);
   if (!ds.Contains("*")) {
      nd++;
      // Verify the dataset
      if ((rc = VerifyDataSet(dsname, opt, redir)) < 0) {
         // Notify
         Printf("%s: ERROR: problems verifing dataset '%s'", action, dsname);
         return rc;
      } else if (rc > 0) {
         // Notify
         Printf("%s: WARNING: %s: some files not yet online (staged)", action, dsname);
      }
      printerr = 0;
   } else {
      // We need to scan all the datasets to find the matching ones ...
      TMap *dss = GetDataSets(dsname, "", "list");
      if (!dss) {
         // Notify
         Printf("%s: ERROR: problems retrieving info about datasets", action);
         return rc;
      }
      printerr = 0;
      // Iterate
      Int_t xrc = -1;
      TIter nxd(dss);
      TObjString *os = 0;
      while ((os = dynamic_cast<TObjString*>(nxd()))) {
         nd++;
         // Verify the dataset
         Printf("%s: start verification of dataset '%s' ...", action, os->GetName());
         if ((xrc = VerifyDataSet(os->GetName(), opt, redir)) < 0) {
            printerr = 1;
            // Notify
            Printf("%s: ERROR: problems verifying dataset '%s'", action, os->GetName());
            continue;
         } else if (xrc > 0) {
            // At least one is not fully available
            rc = 1;
            // Notify
            Printf("%s: WARNING: %s: some files not yet online (staged)", action, os->GetName());
         } else if (rc < 0) {
            // At least one is good
            rc = 0;
         }
      }
   }

   // If no match, notify
   if (nd == 0) {
      Printf("%s: WARNING: no matching dataset found!", action);
   } else {
      Printf("%s: %d dataset(s) verified", action, nd);
   }
   if (printerr)
      gSystem->Rename(flog.Data(), ferr.Data());

   // Done
   return rc;
}

//_______________________________________________________________________________________
void do_anadist(const char *ds, const char *servers, const char *ignsrvs,
                const char *excsrvs, const char *metrics, const char *fnout,
                const char *plot, const char *outfile, const char *infile)
{
   // Execute 'analyze-distribution' for the dataset(s) described by 'ds'.
   // The result is output to the screan and the details about file movement to file
   // 'fnout', if defined.

   const char *action = "pq2-ana-dist";

   // Running mode
   Bool_t plot_m = (plot && strlen(plot)) ? kTRUE : kFALSE;
   Bool_t plotonly_m = (plot_m && infile && strlen(infile)) ? kTRUE : kFALSE;

   // We need to scan all the datasets to find the matching ones ...
   TMap *fcmap = 0;
   TObject *k = 0;
   if (!plotonly_m) {
      redirguard rog(flog.Data(), "a", gverbose);
      TString dss(ds), d;
      Int_t from = 0;
      while (dss.Tokenize(d, from, ",")) {
         TMap *xm = GetDataSets(d);
         if (xm) {
            if (!fcmap) {
               fcmap = xm;
            } else {
               TIter nxds(xm);
               while ((k = nxds())) {
                  fcmap->Add(k, xm->GetValue(k));
                  xm->Remove(k);
               }
            }
            if (xm != fcmap) {
               xm->SetOwner(kFALSE);
               SafeDelete(xm);
            }
         }
      }
      if (!fcmap || fcmap->GetSize() <= 0) {
         SafeDelete(fcmap);
         // Notify
         gSystem->Rename(flog.Data(), ferr.Data());
         Printf("%s: ERROR: problems retrieving info about dataset '%s' (or empty dataset)", action, ds);
         return;
      }
      if (gverbose > 0) fcmap->Print();
   }

   // Which metrics
   Int_t optMet = 0;  // # of files
   if (metrics && !strcmp(metrics, "S")) optMet = 1;  // Size in bytes
   if (gverbose > 0)
      Printf("%s: using metrics: '%s'", action, glabMet[optMet]);

   TList distinfo;
   if (plotonly_m) {
      // Get the dist info
      if (do_anadist_ds(0, 0, 0, 0, optMet, 0, &distinfo, 0, infile) != 0) {
         Printf("%s: problems getting dist info from '%s'", action, infile);
      }
   } else {
      // Name
      TString cname(ds);
      if (cname.BeginsWith("/")) cname.Remove(0,1);
      Ssiz_t ilst = kNPOS;
      if (cname.EndsWith("/") && (ilst = cname.Last('/')) != kNPOS) cname.Remove(ilst);
      cname.ReplaceAll("/", "-");
      cname.ReplaceAll("*", "-star-");
      distinfo.SetName(cname);
      TFileCollection *fc = 0;
      TIter nxd(fcmap);
      TFileCollection *fctot = 0;
      while ((k = nxd()) && (fc = (TFileCollection *) fcmap->GetValue(k))) {
         if (!fctot) {
            // The first one
            fctot = fc;
            fcmap->Remove(k);
         } else {
            // Add
            fctot->Add(fc);
         }
      }
      // Analyse the global dataset
      if (do_anadist_ds(fctot, servers, ignsrvs, excsrvs,
                        optMet, fnout, &distinfo, outfile, infile) != 0) {
         Printf("%s: problems analysing dataset '%s'", action, fc ? fc->GetName() : "<undef>");
      }
      // Cleanup
      SafeDelete(fcmap);
      SafeDelete(fctot);
   }

   // Save histo, if any
   TString fileplot(plot), gext;
   if (!(fileplot.IsNull())) {
      if (fileplot.Contains(".")) {
         gext = fileplot(fileplot.Last('.') + 1, fileplot.Length());
      } else {
         gext = "png";
      }
      const char *fmts[9] = {"png", "eps", "ps", "pdf", "svg", "gif", "xpm", "jpg", "tiff" };
      Int_t iplot = 0;
      while (iplot < 9 && gext != fmts[iplot]) { iplot++; }
      if (iplot == 9) {
         Printf("%s: graphics format '%s' not supported: switching to 'png'", action, gext.Data());
         gext = "png";
      }
      if (!(fileplot.EndsWith(gext))) {
         if (!(fileplot.EndsWith("."))) fileplot += ".";
         fileplot += gext;
      }
      // Create the histogram
      TH1D *h1d = 0;
      if (distinfo.GetSize() > 0) {
         h1d = new TH1D("DistInfoHist", distinfo.GetName(), distinfo.GetSize(), 0.5, distinfo.GetSize() + .5);
         TIter nxs(&distinfo);
         TParameter<Double_t> *ent = 0;
         Double_t x = 0;
         while ((ent = (TParameter<Double_t> *) nxs())) {
            x += 1.;
            h1d->Fill(x, ent->GetVal());
            TString nn(TUrl(ent->GetName()).GetHost()), nnn(nn);
            nnn.ReplaceAll(".", "");
            if (!nnn.IsDigit() && nn.Contains(".")) nn.Remove(nn.First('.'));
            Int_t i = h1d->FindBin(x);
            h1d->GetXaxis()->SetBinLabel(i, nn.Data());
         }
         h1d->GetXaxis()->SetLabelSize(0.03);
      } else {
         Printf("%s: plot requested but no server found (info list is empty)!", action);
      }
      if (h1d) {
         TString filehist(fileplot);
         filehist.Remove(filehist.Last('.')+1);
         filehist += "root";
         TFile *f = TFile::Open(filehist, "RECREATE");
         if (f) {
            f->cd();
            h1d->Write(0,TObject::kOverwrite);
            SafeDelete(f);
            // Write the instruction for the plotting macro
            TString filetmp = TString::Format("%s/%s.tmp", gSystem->TempDirectory(), action);
            FILE *ftmp = fopen(filetmp.Data(), "w");
            if (ftmp) {
               fprintf(ftmp, "%s %s %s", filehist.Data(), fileplot.Data(), glabMet[optMet]);
               fclose(ftmp);
            } else {
               Printf("%s: problems opening temp file '%s' (errno: %d)", action, filetmp.Data(), errno);
               Printf("%s: relevant info: %s %s %s (input to pq2PlotDist.C)",
                                             action, filehist.Data(), fileplot.Data(), glabMet[optMet]);
            }
         } else {
            Printf("%s: problems opening file '%s'", action, filehist.Data());
         }
      } else {
         Printf("%s: histogram requested but not found", action);
      }
   }

   // Done
   return;
}

//_______________________________________________________________________________________
int do_anadist_ds(TFileCollection *fc, const char *servers, const char *ignsrvs,
                  const char *excsrvs, int met, const char *fnout,
                  TList *distinfo, const char *outfile, const char *infile)
{
   // Do analysis of dataset 'fc'

   const char *action = "pq2-ana-dist-ds";

   // Check the inputs
   Bool_t distonly_m = (!fc && distinfo && infile && strlen(infile) > 0) ? kTRUE : kFALSE;
   const char *dsname = 0;
   if (!distonly_m) {
      if (!fc) {
         Printf("%s: dataset undefined!", action);
         return -1;
      }
      dsname = fc->GetName();
      if (fc->GetList()->GetSize() <= 0) {
         Printf("%s: dataset '%s' is empty", action, dsname);
         return -1;
      }
   } else {
      dsname = distinfo->GetName();
   }

   THashList *ignore = 0, *targets = 0, *exclude = 0;
   Bool_t addmode = kFALSE;
   if (!distonly_m) {
      TString ss, k, key;
      Int_t from = 0;
      // List of servers to be ignored
      if (ignsrvs && strlen(ignsrvs)) {
         ss = ignsrvs;
         from = 0;
         while (ss.Tokenize(k, from, ",")) {
            do_anadist_getkey(k.Data(), key);
            if (!(key.IsNull())) {
               if (!ignore) ignore = new THashList();
               ignore->Add(new TObjString(key));
            }
         }
      }
      // List of servers to be excluded
      if (excsrvs && strlen(excsrvs)) {
         ss = excsrvs;
         from = 0;
         while (ss.Tokenize(k, from, ",")) {
            do_anadist_getkey(k.Data(), key);
            if (!(key.IsNull())) {
               if (!exclude) exclude = new THashList();
               exclude->Add(new TObjString(key));
            }
         }
      }
      // List of sub-TFileCollection for target servers: in add mode we complete it during
      // the first scan
      targets = new THashList();
      if (servers && strlen(servers)) {
         ss = servers;
         if (ss.BeginsWith("+")) {
            addmode = kTRUE;
            ss.Remove(0,1);
         }
         from = 0;
         while (ss.Tokenize(k, from, ",")) {
            do_anadist_getkey(k.Data(), key);
            if (!(key.IsNull())) targets->Add(new TFileCollection(key));
         }
      } else {
         addmode = kTRUE;
      }
   }
   // List of sub-TFileCollection for each server, so that we automatically count
   // the sizes and totals
   TNamed *fcsls_title = 0;
   THashList *fcsls = 0;
   Int_t targets_size = -1;
   if (infile && strlen(infile)) {
      TFile *flist = TFile::Open(infile);
      if (flist) {
         Printf("%s: reading info from file '%s' ", action, infile);
         if (!(fcsls = (THashList *) flist->Get("FileDistList"))) {
            Printf("%s: could not find starting file distribution 'FileDistList' in input file '%s' ",
                        action, infile);
            flist->Close();
            SafeDelete(flist);
            SDELTRE(ignore, targets, exclude);
            return -1;
         }
         // Get the title
         fcsls_title = (TNamed *) flist->Get("FileDistList_Title");
         // Get the targets size
         TParameter<Int_t> *psz = (TParameter<Int_t> *) flist->Get("Targets_Size");
         if (psz) targets_size = psz->GetVal();
         // Close
         flist->Close();
         SafeDelete(flist);
         // Add info about the current run in the title
         if (distinfo && fcsls_title && strlen(fcsls_title->GetTitle()) > 0) {
            TString runt(fcsls_title->GetTitle());
            if (strcmp(distinfo->GetName(), "TList")) {
               if (!(runt.IsNull())) runt += ",";
               runt += distinfo->GetName();
               fcsls_title->SetTitle(runt);
            }
            distinfo->SetName(fcsls_title->GetTitle());
         }
      } else {
         Printf("%s: problems opening input file '%s' ", action, infile);
         SDELTRE(ignore, targets, exclude);
         return -1;
      }
   }
   if (!fcsls) {
      fcsls = new THashList();
      fcsls->SetName("FileDistList");
      fcsls_title = new TNamed("FileDistList_Title", distinfo ? distinfo->GetName() : "");
   }

   // Set initial values for the counters, if needed
   Long64_t totsz = 0, totfiles = 0;
   TFileCollection *fcs = 0;
   TIter nxfc(fcsls);
   while ((fcs = (TFileCollection *) nxfc())) {
      fcs->Update();
      totfiles += fcs->GetNFiles();
      totsz += fcs->GetTotalSize();
   }

   // Analyze the file collection content now
   TFileInfo *fi = 0;
   if (fc) {
      TIter nxfi(fc->GetList());
      while ((fi = (TFileInfo *) nxfi())) {
         // Get the key
         TString key;
         do_anadist_getkey(fi->GetCurrentUrl(), key);
         // Ignore if requested
         if (ignore && ignore->FindObject(key)) continue;
         // Get the TFileCollection for this server
         if (!(fcs = (TFileCollection *) fcsls->FindObject(key))) {
            if (gverbose > 0)
               Printf("%s:%s: found server '%s' ... ", action, dsname, key.Data());
            fcs = new TFileCollection(key);
            fcsls->Add(fcs);
         }
         fcs->Add(fi);
         // In add mode, add  as target, if needed
         Bool_t excluded = (exclude && exclude->FindObject(key)) ? kTRUE : kFALSE;
         if (!excluded) {
            if (targets && !(fcs = (TFileCollection *) targets->FindObject(key))) {
               if (addmode) {
                  if (gverbose > 0)
                     Printf("%s:%s: add new target server '%s' ...", action, dsname, key.Data());
                  fcs = new TFileCollection(key);
                  targets->Add(fcs);
               }
            }
            if (fcs) fcs->Add(fi);
         }
         // Count
         totsz += fi->GetSize();
         totfiles++;
      }

      // Nothing to do if no targets
      if (targets->GetSize() <= 0) {
         Printf("%s:%s: target servers list is empty!", action, dsname);
         SDELETE(ignore, targets, exclude, fcsls, fcsls_title);
         return -1;
      } else {
         Printf("%s:%s: %d target servers found", action, dsname, targets->GetSize());
         if (gverbose > 0) targets->Print();
      }
   }
   SDELTWO(ignore, exclude);

   // Separate into 'excess' and 'defect' lists
   TList *excls = new TList;
   TList *defls = new TList;
   targets_size = (targets) ? targets->GetSize() : targets_size;
   Double_t avgfiles = 0, avgsize = 0;
   if (targets_size > 0) {
      avgfiles = (Double_t)totfiles / targets_size;
      avgsize = (Double_t)totsz / targets_size / 1024. / 1024. / 1024.;
      Printf("%s:%s: %d servers found, %lld files; in average: %.3f files / %.3f GBs per server",
                  action, dsname, fcsls->GetSize(), totfiles, avgfiles, avgsize);
   } else {
      // Cannot continue;
      Printf("%s:%s: target size is null or negative", action, dsname);
      SDELETE(ignore, targets, exclude, fcsls, fcsls_title);
      return -1;
   }
   // Before redistribution
   if (gverbose > 0) Printf("\n%s:%s: Before redistribution:", action, dsname);
   nxfc.Reset();
   while ((fcs = (TFileCollection *) nxfc())) {
      fcs->Update();
      Long64_t nfexcess = fcs->GetNFiles() - (Long64_t) avgfiles;
      Double_t xdf = nfexcess / avgfiles;
      Double_t fcsz = fcs->GetTotalSize() / 1024. / 1024. / 1024.;
      Double_t szexcess = fcsz - avgsize;
      Double_t xdsz = szexcess / avgsize;
      // Fill the output histogram, if needed
      if (distinfo) {
         TParameter<Double_t> *ent = (TParameter<Double_t> *) distinfo->FindObject(fcs->GetName());
         if (!ent) {
            ent = new TParameter<Double_t>(fcs->GetName(), 0.);
            distinfo->Add(ent);
         }
         if (met == 0) {
            ent->SetVal(ent->GetVal() + (Double_t) fcs->GetNFiles());
         } else if (met == 1) {
            ent->SetVal(ent->GetVal() + fcsz);
         }
      }
      if (gverbose > 0)
         Printf("%s:%s:  server %s:  %lld files (diff: %lld, %.3f) - %.3f GBs (diff: %.3f, %.3f)",
                action, dsname, fcs->GetName(), fcs->GetNFiles(), nfexcess, xdf, fcsz, szexcess, xdsz);
      if (fc) {
         // Move to the appropriate list
         Bool_t isExcess = kFALSE;
         if (targets && targets->FindObject(fcs->GetName())) {
            if (met == 0) {
               if (nfexcess > 0.) isExcess = kTRUE;
            } else if (met == 1) {
               if (szexcess > 0.) isExcess = kTRUE;
            }
         } else {
            // This server needs to be freed
            isExcess = kTRUE;
         }
         if (isExcess) {
            excls->Add(fcs);
         } else {
            defls->Add(fcs);
         }
      }
   }
   if (outfile && strlen(outfile)) {
      TFile *flist = TFile::Open(outfile, "RECREATE");
      if (flist) {
         flist->cd();
         Printf("%s: saving info to file '%s' ", action, outfile);
         fcsls->Write("FileDistList", TObject::kOverwrite | TObject::kSingleKey);
         if (fcsls_title) fcsls_title->Write("FileDistList_Title", TObject::kOverwrite);
         if (targets) {
            TParameter<Int_t> *psz = new TParameter<Int_t>("Targets_Size", targets->GetSize());
            psz->Write("Targets_Size", TObject::kOverwrite);
         }
         flist->Close();
         SafeDelete(flist);
      } else {
         Printf("%s: problems opening output file '%s' ", action, outfile);
         return -1;
      }
   }
   // Cleanup
   fcsls->SetOwner(0);
   SDELTWO(fcsls, fcsls_title);

   // If we just need the dist info we are done
   if (!fc) {
      SDELETE(targets, fcsls, fcsls_title, excls, defls);
      return 0;
   }

   // Notify
   if (gverbose > 0) {
      Printf("%s:%s: %d servers found in excess", action, dsname, excls->GetSize());
      excls->Print();
      Printf("%s:%s: %d servers found in defect", action, dsname, defls->GetSize());
      defls->Print();
   } else {
      Printf("%s:%s: %d servers found in excess, %d in defect", action, dsname, excls->GetSize(), defls->GetSize());
   }

   // Open output file, if requested
   FILE *fout = 0;
   if (fnout && strlen(fnout) > 0) {
      if (!(fout = fopen(fnout, "a"))) {
         Printf("%s: problems opening output file '%s' (errno: %d)", action, fnout, errno);
         SDELETE(targets, fcsls, fcsls_title, excls, defls);
         return -1;
      }
   }

   // Get the list of files to be moved
   THashList szls;
   TIter nxefc(excls);
   TIter nxdfc(defls);
   Int_t mvfiles = 0;
   Bool_t printheader = kTRUE;
   TFileCollection *fcd = (TFileCollection *) nxdfc();
   while ((fcs = (TFileCollection *) nxefc())) {
      Bool_t isTarget = (targets->FindObject(fcs->GetName())) ? kTRUE : kFALSE;
      Long64_t fcfiles = 0;
      Double_t fcsz = 0.;
      TIter nxefi(fcs->GetList());
      while ((fi = (TFileInfo *) nxefi())) {
         if (!fcd) {
            Printf("%s:%s: WARNING: processing list in excess '%s': no more lists in deficit!",
                   action, dsname, fcs->GetName());
            break;
         }
         // Count
         fcfiles++;
         fcsz += (fi->GetSize() / 1024. / 1024. / 1024.) ;
         if (!isTarget ||
            (((met == 0) && (fcfiles > avgfiles)) || ((met == 1) && (fcsz > avgsize)))) {
            // Write record in output file, if requested
            TUrl u(fi->GetCurrentUrl()->GetUrl());
            u.SetAnchor("");
            u.SetOptions("");
            TString php(u.GetUrl());
            php.Remove(php.Index(u.GetFile()));
            Ssiz_t ilst = kNPOS;
            if (php.EndsWith("/") && ((ilst = php.Last('/')) != kNPOS)) php.Remove(ilst);
            if (fout) {
               fprintf(fout,"%s %s %s\n", u.GetFile(), php.Data(), fcd->GetName());
            } else {
               if (printheader) Printf(" File  Source_Server  Destination_Server");
               Printf("%s %s %s", u.GetFile(), php.Data(), fcd->GetName());
               printheader = kFALSE;
            }
            fcs->GetList()->Remove(fi);
            fcd->Add(fi);
            Bool_t getnext = kFALSE;
            if (met == 0 && fcd->GetList()->GetSize() > avgfiles) getnext = kTRUE;
            if (met == 1) {
               Long64_t xfcsz = 0;
               TParameter<Long64_t> *ptot = (TParameter<Long64_t> *) szls.FindObject(fcd);
               if (!ptot) {
                  fcd->Update();
                  ptot = new TParameter<Long64_t>(fcd->GetName(), fcd->GetTotalSize());
                  xfcsz = ptot->GetVal();
                  szls.Add(ptot);
               } else {
                  xfcsz = ptot->GetVal();
                  xfcsz += fi->GetSize();
               }
               if ((xfcsz / 1024. / 1024. / 1024.) > avgsize) getnext = kTRUE;
            }
            if (getnext) fcd = (TFileCollection *) nxdfc();
            // Count files to be moved
            mvfiles++;
         }
      }
   }
   // Close the file
   if (fout) {
      if ((fclose(fout)) != 0)
         Printf("%s: problems closing output file '%s' (errno: %d)", action, fnout, errno);
   }
   Printf("%s:%s: %d files should be moved to make the distribution even (metrics: %s)",
          action, dsname, mvfiles, glabMet[met]);

   // After redistribution
   if (gverbose > 0) {
      Printf("\n%s:%s: After redistribution:", action, dsname);
      nxefc.Reset();
      while ((fcs = (TFileCollection *) nxefc())) {
         fcs->Update();
         Long64_t nfexcess = fcs->GetNFiles() - (Long64_t) avgfiles;
         Double_t xdf = nfexcess / avgfiles;
         Double_t fcsz = fcs->GetTotalSize() / 1024. / 1024. / 1024.;
         Double_t szexcess = fcsz - avgsize;
         Double_t xdsz = szexcess / avgsize;
         Printf("%s:%s:  Server %s:  %lld files (diff: %lld, %.3f) - %.3f GBs (diff: %.3f, %.3f)",
                action, dsname, fcs->GetName(), fcs->GetNFiles(), nfexcess, xdf, fcsz, szexcess, xdsz);
      }
      nxdfc.Reset();
      while ((fcs = (TFileCollection *) nxdfc())) {
         fcs->Update();
         Long64_t nfexcess = fcs->GetNFiles() - (Long64_t) avgfiles;
         Double_t xdf = nfexcess / avgfiles;
         Double_t fcsz = fcs->GetTotalSize() / 1024. / 1024. / 1024.;
         Double_t szexcess = fcsz - avgsize;
         Double_t xdsz = szexcess / avgsize;
         Printf("%s:%s:  server %s:  %lld files (diff: %lld, %.3f) - %.3f GBs (diff: %.3f, %.3f)",
                action, dsname, fcs->GetName(), fcs->GetNFiles(), nfexcess, xdf, fcsz, szexcess, xdsz);
      }
   }
   // Cleanup
   SDELETE(targets, fcsls, fcsls_title, excls, defls);
   // Done
   return 0;
}

//_______________________________________________________________________________________
void do_anadist_getkey(const char *p, TString &key)
{
   // Get the key corresponding to path 'p'.

   TUrl u(p);
   if (strncmp(p, u.GetProtocol(), strlen(u.GetProtocol()))) {
      u.SetProtocol("root");
      TString sport = TString::Format(":%d", u.GetPort());
      if (!strstr(p, sport.Data())) u.SetPort(1094);
   }
   if (gverbose > 0) Printf("do_anadist_getkey: url: %s", u.GetUrl());
   return do_anadist_getkey(&u, key);
}

//_______________________________________________________________________________________
void do_anadist_getkey(TUrl *u, TString &key)
{
   // Get the key corresponding to url 'u'.

   key = "";
   if (u) {
      TParameter<Int_t> *php = (TParameter<Int_t> *)gProtoPortMap.FindObject(u->GetProtocol());
      if (!php) {
         TUrl xu(TString::Format("%s://host//file", u->GetProtocol()));
         php = new TParameter<Int_t>(u->GetProtocol(), xu.GetPort());
         gProtoPortMap.Add(php);
      }
      if (u->GetPort() != php->GetVal()) {
         key.Form("%s://%s:%d", u->GetProtocol(), u->GetHost(), u->GetPort());
      } else {
         key.Form("%s://%s", u->GetProtocol(), u->GetHost());
      }
   }
   // Done
   return;
}
//_______________________________________________________________________________________
int do_anadist_plot(TH1D *h1d, const char */*fnout*/)
{
   // Create the plot for the histogram, and save to 'fnout'.
   // Format determined by th extension of fnout.

   Printf("do_anadist_plot: will be doing a plot here ... ");

   if (h1d) {
      Printf("do_anadist_plot: we save the histo for now (to testhist.root)");
      TFile *f = TFile::Open("testhist.root", "RECREATE");
      if (f) {
         f->cd();
         h1d->Write(0,TObject::kOverwrite);
         SafeDelete(f);
         return 0;
      }
   }
   return -1;
}
