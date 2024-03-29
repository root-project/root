// @(#)root/test:$Id$
// Author: David Smith   20/10/14

/////////////////////////////////////////////////////////////////
//
//___A test for I/O plugins by reading files___
//
//   The files used in this test have been generated by
//   stress.cxx and preplaced on some data servers.
//   stressIOPlugins reads the remote files via various data
//   access protocols to test ROOT IO plugins. The data read are
//   tested via tests based on some of stress.cxx tests.
//
//   Can be run as:
//     stressIOPlugins [name]
//
//   The name parameter is a protocol name, as expected
//   in a url. The supported names are: xroot, root, http, https.
//   If the name is omitted a selection of schemes are tested
//   based on feature availability:
//
//           feature          protocol    multithreaded test available
//
//            xrootd           root                no
//            davix            http                no
//
// An example of output of a non multithreaded test, when all the tests
// run OK is shown below:
//
// ****************************************************************************
// *  Starting stressIOPlugins test for protocol http
// *  Test files will be read from:
// *  http://root.cern/files/StressIOPluginsTestFiles/
// ****************************************************************************
// Test  1 : Check size & compression factor of a Root file........ using stress_2.root
//         : opened file with plugin class......................... TDavixFile
//         : Check size & compression factor of a Root file........ OK
// Test  2 : Test graphics & Postscript............................ using stress_5.root
//         : opened file with plugin class......................... TDavixFile
//         : Test graphics & Postscript............................ OK
// Test  3 : Trees split and compression modes..................... using Event_8a.root
//         : opened file with plugin class......................... TDavixFile
//         : Trees split and compression modes..................... using Event_8b.root
//         : opened file with plugin class......................... TDavixFile
//         : Trees split and compression modes..................... OK
// Test  4 : Filename formats when adding files to TChain.......... using Event_8a.root and Event_8b.root
//         : treename in chain..................................... OK
//         : treename to AddFile................................... OK
//         : treename in filenames, slash-suffix style............. OK
//         : bad treename to AddFile, good in filename............. OK
//         : treename and url query in filename.................... OK
//         : treename given in url frag in filename................ OK
//         : filename with a url query in Add...................... OK
// ****************************************************************************
//_____________________________batch only_____________________
#ifndef __CINT__

#include <cstdlib>
#include <snprintf.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TMath.h>
#include <TF1.h>
#include <TF2.h>
#include <TCanvas.h>
#include <TPostScript.h>
#include <TTree.h>
#include <TChain.h>
#include <TTreeCache.h>
#include <TSystem.h>
#include <TApplication.h>
#include <TClassTable.h>
#include <Compression.h>
#include "Event.h"

R__LOAD_LIBRARY( libEvent )

void stressIOPlugins();
void stressIOPluginsForProto(const char *protoName = 0, int multithread = 0);
void stressIOPlugins1();
void stressIOPlugins2();
void stressIOPlugins3();
void stressIOPlugins4();
void stressIOPlugins5();
void cleanup();

int main(int argc, char **argv)
{
   std::string inclRootSys = ("-I" + TROOT::GetRootSys() + "/test").Data();
   TROOT::AddExtraInterpreterArgs({inclRootSys});

   gROOT->SetBatch();
   TApplication theApp("App", &argc, argv);
   const char *proto = 0;
   if (argc > 1)  proto = argv[1];
   stressIOPluginsForProto(proto);
   return 0;
}

#endif

class TH1;
class TTree;

//_______________________ common part_________________________

Double_t ntotin=0, ntotout=0;
TString gPfx,gCurProtoName;

void Bprint(Int_t id, const char *title)
{
  // Print test program number and its title
   const Int_t kMAX = 65;
   char header[80];
   if (id > 0) {
      snprintf(header,80,"Test %2d : %s",id,title);
   } else {
      snprintf(header,80,"        : %s",title);
   }
   Int_t nch = strlen(header);
   for (Int_t i=nch;i<kMAX;i++) header[i] = '.';
   header[kMAX] = 0;
   header[kMAX-1] = ' ';
   printf("%s",header);
}

TFile *openTestFile(const char *fn, const char *title) {

   printf("using %s\n", fn);

   TFile *f = TFile::Open(gPfx + fn);
   Bprint(0,"opened file with plugin class");

   if (!f) {
      printf("FAILED\n");
      Bprint(0, title);
      return 0;
   }

   printf("%s\n", f->ClassName());

   Bprint(0, title);
   return f;
}   

Bool_t isFeatureAvailable(const char *name) {
   TString configfeatures = gROOT->GetConfigFeatures();
   return configfeatures.Contains(name);
}

int setPath(const char *proto)
{
   if (!proto) return -1;
   TString p(proto);
   gCurProtoName = p;
   if (p == "root" || p == "xroot") {
      gPfx = p + "://eospublic.cern.ch//eos/root-eos/StressIOPluginsTestFiles/";
      return 0;
   }
   if (p == "http" || p == "https") {
      gPfx = p + "://root.cern/files/StressIOPluginsTestFiles/";
      return 0;
   }
   return -1;
}

void stressIOPluginsForProto(const char *protoName /*=0*/, int multithread /*=0*/)
{
   //Main control function invoking all test programs
   if (!protoName) {
     if (isFeatureAvailable("xrootd")) {
        stressIOPluginsForProto("root");
     } else {
        printf("* Skipping root protocol test because 'xrootd' feature not available\n");
     }
     if (isFeatureAvailable("davix")) {
        stressIOPluginsForProto("http");
     } else {
        printf("* Skipping http protocol test because 'davix' feature not available\n");
     }
     return;
   }

   if (setPath(protoName)) {
     printf("No server and path available to test protocol %s\n", protoName);
     return;
   }

   if (multithread) {
     printf("No multithreaded tests are available\n");
     return;
   }

   printf("****************************************************************************\n");
   printf("*  Starting stressIOPlugins test for protocol %s\n", protoName);
   printf("*  Test files will be read from:\n");
   printf("*  %s\n", gPfx.Data());
   printf("****************************************************************************\n");

   stressIOPlugins1();
   stressIOPlugins2();
   stressIOPlugins3();
   stressIOPlugins4();
   stressIOPlugins5();

   cleanup();

   printf("****************************************************************************\n");
}

void stressIOPlugins()
{
   stressIOPluginsForProto((const char*)0,0);
}

////////////////////////////////////////////////////////////////////////////////
///based on stress2()
///check length and compression factor in stress.root

void stressIOPlugins1()
{
   const char *title = "Check size & compression factor of a Root file";
   Bprint(1, title);

   TFile *f = openTestFile("stress_2.root", title);
   if (!f) {
      printf("FAILED\n");
     return;
   }

   Long64_t last = f->GetEND();
   Float_t comp = f->GetCompressionFactor();

   Bool_t OK = kTRUE;
   Long64_t lastgood = 9428;
   if (last <lastgood-200 || last > lastgood+200 || comp <2.0 || comp > 2.4) OK = kFALSE;
   if (OK) printf("OK\n");
   else    {
      printf("FAILED\n");
      printf("%-8s last =%lld, comp=%f\n"," ",last,comp);
   }
   delete f;
}

////////////////////////////////////////////////////////////////////////////////
/// based on stress5()
/// Test of Postscript.
/// Make a complex picture. Verify number of lines on ps file
/// Testing automatically the graphics package is a complex problem.
/// The best way we have found is to generate a Postscript image
/// of a complex canvas containing many objects.
/// The number of lines in the ps file is compared with a reference run.
/// A few lines (up to 2 or 3) of difference may be expected because
/// Postscript works with floats. The date and time of the run are also
/// different.
/// You can also inspect visually the ps file with a ps viewer.

void stressIOPlugins2()
{
   const char *title = "Test graphics & Postscript";
   Bprint(2,title);

   TCanvas *c1 = new TCanvas("c1","stress canvas",800,600);
   gROOT->LoadClass("TPostScript","Postscript");
   TString psfname = TString::Format("stressIOPlugins-%d.ps", gSystem->GetPid());
   TPostScript ps(psfname,112);

   //Get objects generated in previous test
   TFile *f = openTestFile("stress_5.root",title);
   if (!f) {
     printf("FAILED\n");
     return;
   }

   TF1  *f1form = (TF1*)f->Get("f1form");
   TF2  *f2form = (TF2*)f->Get("f2form");
   TH1F *h1form = (TH1F*)f->Get("h1form");
   TH2F *h2form = (TH2F*)f->Get("h2form");

   //Divide the canvas in subpads. Plot with different options
   c1->Divide(2,2);
   c1->cd(1);
   f1form->Draw();
   c1->cd(2);
   h1form->Draw();
   c1->cd(3);
   h2form->Draw("box");
   f2form->Draw("cont1same");
   c1->cd(4);
   f2form->Draw("surf");

   ps.Close();

   //count number of lines in ps file
   FILE *fp = fopen(psfname,"r");
   if (!fp) {
      printf("FAILED\n");
      printf("%-8s could not open %s\n"," ",psfname.Data());
      delete c1;
      delete f;
      return;
   }
   char line[260];
   Int_t nlines = 0;
   Int_t nlinesGood = 632;
   while (fgets(line,255,fp)) {
      nlines++;
   }
   fclose(fp);
   ntotin  += f->GetBytesRead();
   ntotout += f->GetBytesWritten();
   Bool_t OK = kTRUE;
   if (nlines < nlinesGood-110 || nlines > nlinesGood+110) OK = kFALSE;
   if (OK) printf("OK\n");
   else    {
      printf("FAILED\n");
      printf("%-8s nlines in %s file = %d\n"," ",psfname.Data(),nlines);
   }
   delete c1;
   delete f;
}

////////////////////////////////////////////////////////////////////////////////

Int_t test3read(const TString &fn, const char *title)
{
//  Read the event file
//  Loop on all events in the file (reading everything).
//  Count number of bytes read

   Int_t nevent = 0;

   TFile *hfile = openTestFile(fn,title);
   if (!hfile) {
      return 0;
   }
   TTree *tree; hfile->GetObject("T",tree);
   Event *event = 0;
   tree->SetBranchAddress("event",&event);
   Int_t nentries = (Int_t)tree->GetEntries();
   Int_t nev = TMath::Max(nevent,nentries);
   //activate the treeCache
   Int_t cachesize = 10000000; //this is the default value: 10 MBytes
   tree->SetCacheSize(cachesize);
   TTreeCache::SetLearnEntries(1); //one entry is sufficient to learn
   TTreeCache *tc = (TTreeCache*)hfile->GetCacheRead();
   tc->SetEntryRange(0,nevent);
   Int_t nb = 0;
   for (Int_t ev = 0; ev < nev; ev++) {
      nb += tree->GetEntry(ev);        //read complete event in memory
   }
   ntotin  += hfile->GetBytesRead();

   delete event;
   delete hfile;
   return nb;
}


////////////////////////////////////////////////////////////////////////////////
/// based on stress8()

void stressIOPlugins3()
{
   const char *title = "Trees split and compression modes";
   Bprint(3,title);

   Int_t nbr0 = test3read("Event_8a.root",title);
   Event::Reset();

   Int_t nbr1 = test3read("Event_8b.root",title);
   Event::Reset();

   Bool_t OK = kTRUE;
   if (nbr0 == 0 || nbr0 != nbr1) OK = kFALSE;
   if (OK) printf("OK\n");
   else    {
      printf("FAILED\n");
      printf("%-8s nbr0=%d, nbr1=%d\n"," ",nbr0,nbr1);
   }
}

////////////////////////////////////////////////////////////////////////////////

void stressIOPlugins4()
{
   Long64_t nent;
   Bool_t tryquery = kTRUE;
   Bool_t trywildcard = kFALSE;
   Bool_t tryqueryInAdd = kFALSE;
   Bool_t tryziparchive = kFALSE;

   const char *title = "Filename formats when adding files to TChain";
   Bprint(4,title);
   printf("using Event_8a.root and Event_8b.root\n");

   if (gCurProtoName == "xroot" || gCurProtoName == "root") {
      trywildcard = kTRUE;
      tryqueryInAdd = kTRUE;
      tryziparchive = kTRUE;
   }

   if (gCurProtoName == "http" || gCurProtoName == "https") {
      tryqueryInAdd = kTRUE;
   }

   {
      Bprint(0,"treename in chain");
      TChain  mychain("T");
      mychain.AddFile(gPfx + "Event_8a.root");
      mychain.AddFile(gPfx + "Event_8b.root");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   {
      Bprint(0,"treename to AddFile");
      TChain  mychain("nosuchtree");
      mychain.AddFile(gPfx + "Event_8a.root", TTree::kMaxEntries, "T");
      mychain.AddFile(gPfx + "Event_8b.root", TTree::kMaxEntries, "T");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   {
      Bprint(0,"treename in filenames, slash-suffix style");
      TChain  mychain("nosuchtree");
      mychain.AddFile(gPfx + "Event_8a.root/T");
      mychain.AddFile(gPfx + "Event_8b.root/T");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   {
      Bprint(0,"bad treename to AddFile, good in filename");
      TChain  mychain("nosuchtree");
      mychain.AddFile(gPfx + "Event_8a.root/T", TTree::kMaxEntries, "nosuchtree2");
      mychain.AddFile(gPfx + "Event_8b.root/T", TTree::kMaxEntries, "nosuchtree2");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   if (tryquery) {
      Bprint(0,"treename and url query in filename");
      TChain  mychain("nosuchtree");
      mychain.AddFile(gPfx + "Event_8a.root/T?myq=xyz");
      mychain.AddFile(gPfx + "Event_8b.root/T?myq=xyz");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   if (tryquery) {
      Bprint(0,"treename given in url frag in filename");
      TChain  mychain("nosuchtree");
      mychain.AddFile(gPfx + "Event_8a.root?myq=xyz#T");
      mychain.AddFile(gPfx + "Event_8b.root?myq=xyz#T");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   if (tryqueryInAdd) {
      Bprint(0,"filename with a url query in Add");
      TChain  mychain("T");
      mychain.Add(gPfx + "Event_8a.root?myq=xyz");
      mychain.Add(gPfx + "Event_8b.root?myq=xyz");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   if (trywildcard) {
      Bprint(0,"wildcarded filename");
      TChain  mychain("T");
      mychain.Add(gPfx + "Event_8*ot");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }

   if (trywildcard) {
      Bprint(0,"wildcarded filename with treename");
      TChain  mychain("nosuchtree");
      mychain.Add(gPfx + "Event_8*.root/T");
      nent = mychain.GetEntries();
      if (nent != 200) {
         printf("FAILED\n");
      } else {
         printf("OK\n");
      }
   }
   if (tryziparchive) {
      Bprint(0,"zip archive");
      printf("using multi_8.zip\n");

      Bprint(0,"sub-file name in fragment");
      {
         TChain  mychain("T");
         mychain.Add(gPfx + "multi_8.zip#Event_8a.root");
         nent = mychain.GetEntries();
         if (nent != 100) {
            printf("FAILED\n");
         } else {
            printf("OK\n");
         }
      }
      Bprint(0,"sub-file index in query");
      {
         TChain  mychain("T");
         mychain.AddFile(gPfx + "multi_8.zip?myq=xyz&zip=0");
         nent = mychain.GetEntries();
         if (nent != 100) {
            printf("FAILED\n");
         } else {
            printf("OK\n");
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////

void stressIOPlugins5()
{
   Bool_t tryziparchive = kFALSE;
   const char *title = "Read content from a zip archive";
   Bprint(5,title);

   if (gCurProtoName == "xroot" || gCurProtoName == "root") {
      tryziparchive = kTRUE;
   }

   if (!tryziparchive) {
      printf("skipping\n");
      return;
   }

   TFile *hfile = openTestFile("multi_8.zip?&zip=1","find tree");
   if (!hfile) {
      printf("FAILED\n");
      return;
   }
   TTree *tree = 0;
   hfile->GetObject("T",tree);
   if (!tree) {
      printf("FAILED\n");
      delete hfile;
      return;
   }
   tree->SetCacheSize(0);
   Int_t nentries = (Int_t)tree->GetEntries();
   if (nentries != 100) {
      printf("FAILED\n");
      delete hfile;
      return;
   } else {
      printf("OK\n");
   }
   Event *event = 0;
   tree->SetBranchAddress("event",&event);
   tree->GetEntry(0);
   Bprint(0,"read event (no cache)");
   if (!event || event->GetNtrack() != 603) {
      printf("FAILED\n");
   } else {
      printf("OK\n");
   }
   delete event;
   delete hfile;
}

void cleanup()
{
   TString psfname = TString::Format("stressIOPlugins-%d.ps", gSystem->GetPid());
   gSystem->Unlink(psfname);
}
