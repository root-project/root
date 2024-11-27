/// \file
/// \ingroup tutorial_tree
///
/// Example to write & read a Tree built with a complex class inheritance tree.
/// It demonstrates usage of inheritance and TClonesArrays
/// This is simplified / stripped extract of an event structure which was used
/// within the Marabou project.
///
/// To run this example, do:
/// ~~~
///  root > .x tree131_clones_event.C
/// ~~~
/// \macro_code
///
/// \author The ROOT Team

#ifndef CLONESA_EVENT_SECOND_RUN

void tree131_clones_event()
{
   std::string s1(__FILE__);
   TString dir = gSystem->UnixPathName(s1.substr(0, s1.find_last_of("\\/")).c_str());
   gROOT->ProcessLine(TString(".L ") + dir + "/clones_event.cxx+");
#define CLONESA_EVENT_SECOND_RUN yes
   gROOT->ProcessLine("#include \"" __FILE__ "\"");
   gROOT->ProcessLine("tree131_clones_event(true)");
}

#else

void write_clones_event()
{
   // protect against old ROOT versions
   if ( gROOT->GetVersionInt() < 30503 ) {
      std::cout << "Works only with ROOT version >= 3.05/03" << std::endl;
      return;
   }
   if ( gROOT->GetVersionDate() < 20030406 ) {
      std::cout << "Works only with ROOT CVS version after 5. 4. 2003" << std::endl;
      return;
   }

   // write a Tree
   auto hfile = TFile::Open("clones_event.root", "RECREATE", "Test TClonesArray");
   auto tree  = new TTree("clones_event", "An example of a ROOT tree");
   auto event1 = new TUsrSevtData1();
   auto event2 = new TUsrSevtData2();
   tree->Branch("top1", "TUsrSevtData1", &event1, 8000, 99);
   tree->Branch("top2", "TUsrSevtData2", &event2, 8000, 99);
   for (Int_t ev = 0; ev < 10; ev++) {
      std::cout << "event " << ev << std::endl;
      event1->SetEvent(ev);
      event2->SetEvent(ev);
      tree->Fill();
      if (ev <3)
         tree->Show(ev);
   }
   tree->Write();
   tree->Print();
   delete hfile;
}

void read_clones_event()
{
   //read the Tree
   auto hfile = TFile::Open("clones_event.root");
   auto tree = hfile->Get<TTree>("clones_event");

   TUsrSevtData1 * event1 = 0;
   TUsrSevtData2 * event2 = 0;
   tree->SetBranchAddress("top1", &event1);
   tree->SetBranchAddress("top2", &event2);
   for (Int_t ev = 0; ev < 8; ev++) {
      tree->Show(ev);
      std::cout << "Pileup event1: " <<  event1->GetPileup() << std::endl;
      std::cout << "Pileup event2: " <<  event2->GetPileup() << std::endl;
      event1->Clear();
      event2->Clear();
      // gObjectTable->Print();          // detect possible memory leaks
   }
   delete hfile;
}

void tree131_clones_event(bool /*secondrun*/)
{
   // Embedding this load inside the first run of the script is not yet
   // supported in v6
   // gROOT->ProcessLine(".L clones_event.cxx+");  // compile shared lib
   write_clones_event();                            // write the tree
   read_clones_event();                            // read back the tree
}

#endif
