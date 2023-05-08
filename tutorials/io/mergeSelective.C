/// \file
/// \ingroup tutorial_io
/// \notebook -nodraw
/// Merge only part of the content of a set of files.
/// This macro demonstrates how to merge only a part of the content of a set
/// of input files, specified via the interface.
/// ~~~{.cpp}
///     TFileMerger::AddObjectNames(const char *names)
/// ~~~
/// The method can be called several times to add object names, or using a single
/// string with names separated by a blank. Directory names contained in the files
/// to be merged are accepted.
///
/// Two modes are supported:
/// 1. kOnlyListed: via <tt>TFileMerger::PartialMerge(kOnlyListed)</tt>
///    This will merge only the objects in the files having the names in the
///    specified list. If a folder is specified, its whole content will be merged
///
/// 2. kSkipListed: via <tt>TFileMerger::PartialMerge(kSkipListed)</tt>
///    This will skip merging of specified objects. If a folder is specified, its
///    whole content will be skipped. 
/// 
/// Important note: the kOnlyListed and kSkipListed flags have to be bitwise OR-ed
/// on top of the merging defaults: kAll | kIncremental (as in the example)
///
/// The files to be merged have the following structure:
/// - hpx          (TH1F)
/// - hpxpy        (TH2F)
/// - hprof        (TProfile)
/// - ntuple       (TNtuple)
/// - folder       (TDirectory)
///      - hpx1    (TH1F)
///
/// The example first merges exclusively hprof and the content of "folder",
/// producing the file exclusive.root, then merges all content but skipping
/// hprof and the content of "folder". The result can be inspected in the
/// browser.
///
/// \macro_code
///
/// \author The Root Team


void CreateFile(const char *);

void mergeSelective(Int_t nfiles=5)
{

   // Create the files to be merged
   TStopwatch timer;
   timer.Start();
   TString tutdir = gROOT->GetTutorialDir();
   if (gROOT->LoadMacro(tutdir + "/hsimple.C")) return;
   Int_t i;
   for (i=0; i<nfiles; i++) CreateFile(Form("tomerge%03d.root",i));

   //------------------------------------
   // Merge only the listed objects
   //------------------------------------
   TFileMerger *fm;
   fm = new TFileMerger(kFALSE);
   fm->OutputFile("exclusive.root");
   fm->AddObjectNames("hprof folder");
   for (i=0; i<nfiles; i++) fm->AddFile(Form("tomerge%03d.root",i));
   // Must add new merging flag on top of the default ones
   Int_t default_mode = TFileMerger::kAll | TFileMerger::kIncremental;
   Int_t mode = default_mode | TFileMerger::kOnlyListed;
   fm->PartialMerge(mode);
   fm->Reset();

   //------------------------------------
   // Skip merging of the listed objects
   //------------------------------------
   fm->OutputFile("skipped.root");
   fm->AddObjectNames("hprof folder");
   for (i=0; i<nfiles; i++) fm->AddFile(Form("tomerge%03d.root",i));
   // Must add new merging flag on top of the default ones
   mode = default_mode | TFileMerger::kSkipListed;
   fm->PartialMerge(mode);
   delete fm;


   // Cleanup initial files
   for (i=0; i<nfiles; i++) gSystem->Unlink(Form("tomerge%03d.root",i));
   // Open files to inspect in the browser
   TFile::Open("exclusive.root");
   TFile::Open("skipped.root");
   new TBrowser();
   timer.Stop();
   timer.Print();
}

void CreateFile(const char *fname)
{
   TFile *example = (TFile*)gROOT->ProcessLineFast("hsimple(1)");
   if (!example) return;
   TH1F *hpx = (TH1F*)example->Get("hpx");
   hpx->SetName("hpx1");
   TFile::Cp(example->GetName(), fname);
   TFile *file = TFile::Open(fname, "UPDATE");
   file->mkdir("folder")->cd();
   hpx->Write();
   file->Close();
   example->Close();
   TString sname(fname);
   if (sname.Contains("000")) {
      TFile::Cp(fname, "original.root");
      TFile::Open("original.root");
   }
}
