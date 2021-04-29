/// \file
/// \ingroup tutorial_eve
/// Complex example showing ALICE ESD visualization in several views.
///   alice_esd_split.C - a simple event-display for ALICE ESD tracks and clusters
///                       version with several windows in the same workspace
///
///
///   Only standard ROOT is used to process the ALICE ESD files.
///
///   No ALICE code is needed, only four simple coordinate-transformation
///   functions declared in this macro.
///
///   A simple geometry of 10KB, extracted from the full TGeo-geometry, is
///   used to outline the central detectors of ALICE.
///
///   All files are access from the web by using the "CACHEREAD" option.
///
///
///   ### Automatic building of ALICE ESD class declarations and dictionaries.
///
///   ALICE ESD is a TTree containing tracks and other event-related
///   information with one entry per event. All these classes are part of
///   the AliROOT offline framework and are not available to standard
///   ROOT.
///
///   To be able to access the event data in a natural way, by using
///   data-members of classes and object containers, the header files and
///   class dictionaries are automatically generated from the
///   TStreamerInfo classes stored in the ESD file by using the
///   TFile::MakeProject() function. The header files and a shared library
///   is created in the aliesd/ directory and can be loaded dynamically
///   into the ROOT session.
///
///   See the run_alice_esd.C macro.
///
///
///   ### Creation of simple GUI for event navigation.
///
///   Most common use of the event-display is to browse through a
///   collection of events. Thus a simple GUI allowing this is created in
///   the function make_gui().
///
///   Eve uses the configurable ROOT-browser as its main window and so we
///   create an extra tab in the left working area of the browser and
///   provide backward/forward buttons.
///
///
///   ### Event-navigation functions.
///
///   As this is a simple macro, we store the information about the
///   current event in the global variable 'Int_t esd_event_id'. The
///   functions for event-navigation simply modify this variable and call
///   the load_event() function which does the following:
///   1. drop the old visualization objects;
///   2. retrieve given event from the ESD tree;
///   3. call alice_esd_read() function to create visualization objects
///      for the new event.
///
///
///   ### Reading of ALICE data and creation of visualization objects.
///
///   This is performed in alice_esd_read() function, with the following
///   steps:
///   1. create the track container object - TEveTrackList;
///   2. iterate over the ESD tracks, create TEveTrack objects and append
///      them to the container;
///   3. instruct the container to extrapolate the tracks and set their
///      visual attributes.
///
/// \image html eve_alice_esd_split.png
/// \macro_code
///
/// \author Bertrand Bellenot

void alice_esd_split()
{
   TString dir = gSystem->UnixPathName(__FILE__);
   dir.ReplaceAll("alice_esd_split.C","");
   dir.ReplaceAll("/./","/");
   gROOT->LoadMacro(dir +"SplitGLView.C+");
   const char* esd_file_name = "http://root.cern.ch/files/alice_ESDs.root";
   TFile::SetCacheFileDir(".");
   TString lib(Form("aliesd/aliesd.%s", gSystem->GetSoExt()));

   if (gSystem->AccessPathName(lib, kReadPermission)) {
      TFile* f = TFile::Open(esd_file_name, "CACHEREAD");
      if (!f) return;
      TTree *tree = (TTree*) f->Get("esdTree");
      tree->SetBranchStatus ("ESDfriend*", 1);
      f->MakeProject("aliesd", "*", "++");
      f->Close();
      delete f;
   }
   gSystem->Load(lib.Data());
   gROOT->ProcessLine(".x run_alice_esd_split.C");
}
