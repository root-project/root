// $Id: make_event_trees.C,v 1.11 2007/07/31 16:20:23 rdm Exp $
//
//

#include "Riostream.h"
#include "TProof.h"
#include "TString.h"


Bool_t make_event_trees(const char *basedir, Int_t events_per_file,
                        Int_t files_per_node)
{
   // This script generates files on PROOF nodes with trees containing
   // Event objects.
   // The created files can be used for benchmarking and demonstrations.

   if (!gProof) {
      cout << "Must Start PROOF before using make_event_trees.C" << endl;
      return kFALSE;
   }

   if (!basedir) {
      cout << "'basedir' must not be empty" << endl;
      return kFALSE;
   }

   if (events_per_file <= 0) {
      cout << "events_per_file must be > 0" << endl;
      return kFALSE;
   }

   if (files_per_node <= 0) {
      cout << "files_per_node must be > 0" << endl;
      return kFALSE;
   }

   if (gProof->UploadPackage("event.par")) return kFALSE;
   if (gProof->EnablePackage("event")) return kFALSE;

   ofstream slavemacro("build_trees.C");

   slavemacro << "#include \"TSystem.h\""                                                 << endl;
   slavemacro << "#include \"TProof.h\""                                                  << endl;
   slavemacro << "#include \"TProofServ.h\""                                              << endl;
   slavemacro << "#include \"TRandom.h\""                                                 << endl;
   slavemacro << "#include \"TFile.h\""                                                   << endl;
   slavemacro << "#include \"TTree.h\""                                                   << endl;
   slavemacro << "#include \"Riostream.h\""                                               << endl;
   slavemacro << "#include \"event/Event.h\""                                                   << endl;
   slavemacro << "void build_trees(const char *basedir, Int_t nevents, Int_t nfiles) {" << endl;
   slavemacro << "   Int_t slave_number = -1;"                                            << endl;
   slavemacro << "   Int_t nslaves = 0;"                                                  << endl;
   if (!strncmp(gProof->GetMaster(), "localhost", 9))
      slavemacro << "   TString hn = \"localhost\";"                                      << endl;
   else
      slavemacro << "   TString hn = gSystem->HostName();"                                << endl;
   slavemacro << "   TString ord = gProofServ->GetOrdinal();"                             << endl;
   slavemacro <<                                                                             endl;
   TList* l = gProof->GetListOfSlaveInfos();
   for (Int_t i=0; i<l->GetSize(); i++) {
      TSlaveInfo* si = dynamic_cast<TSlaveInfo*>(l->At(i));
      if (si->fStatus != TSlaveInfo::kActive) continue;
      slavemacro << "   if (hn == \"";
      slavemacro << si->fHostName;
      slavemacro << "\") { nslaves++; if (ord == \"";
      slavemacro << si->fOrdinal;
      slavemacro << "\") slave_number = nslaves; }" << endl;
   }

   slavemacro << "  if (gSystem->AccessPathName(basedir)) {"                              << endl;
   slavemacro << "     Printf(\"No such file or directory: %s\", basedir);"               << endl;
   slavemacro << "     return;"                                                           << endl;
   slavemacro << "  }"                                                                    << endl;
   slavemacro <<                                                                             endl;
   slavemacro << "   if (slave_number >= 0) {"                                            << endl;
   slavemacro << "      for(Int_t i=slave_number; i<=nfiles; i+=nslaves) {"               << endl;
   slavemacro <<                                                                             endl;
   slavemacro << "         TString seed = hn;"                                            << endl;
   slavemacro << "         seed += \"_\";"                                                << endl;
   slavemacro << "         seed += i;"                                                    << endl;
   slavemacro << "         gRandom->SetSeed(static_cast<UInt_t>(TMath::Hash(seed)));"     << endl;
   slavemacro <<                                                                             endl;
   slavemacro << "         TString filename = basedir;"                                   << endl;
   slavemacro << "         filename += \"/event_tree_\";"                                 << endl;
   slavemacro << "         filename += seed;"                                             << endl;
   slavemacro << "         filename += \".root\";"                                        << endl;
   slavemacro << "         TDirectory* savedir = gDirectory;"                             << endl;
   slavemacro << "         TFile *f = TFile::Open(filename, \"RECREATE\");"               << endl;
   slavemacro << "         savedir->cd();"                                                << endl;
   slavemacro <<                                                                             endl;
   slavemacro << "         if (!f || f->IsZombie()) break;"                               << endl;
   slavemacro << "         Event event;"                                                  << endl;
   slavemacro << "         Event *ep = &event;"                                           << endl;
   slavemacro << "         TTree eventtree(\"EventTree\", \"Event Tree\");"               << endl;
   slavemacro << "         eventtree.SetDirectory(f);"                                    << endl;
   slavemacro << "         eventtree.Branch(\"event\", \"Event\", &ep, 32000, 1);"        << endl;
   slavemacro << "         eventtree.AutoSave();"                                         << endl;
   slavemacro <<                                                                             endl;
   slavemacro << "         for(Int_t j=0; j<nevents; j++) {"                              << endl;
   slavemacro << "            event.Build(j,3,0);"                                        << endl;
   slavemacro << "            eventtree.Fill();"                                          << endl;
   slavemacro << "         }"                                                             << endl;
   slavemacro <<                                                                             endl;
   slavemacro << "         savedir = gDirectory;"                                         << endl;
   slavemacro << "         f->cd();"                                                      << endl;
   slavemacro << "         eventtree.Write();"                                            << endl;
   slavemacro << "         eventtree.SetDirectory(0);"                                    << endl;
   slavemacro << "         f->Close();"                                                   << endl;
   slavemacro << "         delete f;"                                                     << endl;
   slavemacro << "         f = 0;"                                                        << endl;
   slavemacro << "         savedir->cd();"                                                << endl;
   slavemacro <<                                                                             endl;
   slavemacro << "      }"                                                                << endl;
   slavemacro << "   } else {"                                                            << endl;
   slavemacro << "      cout << \"Could not find slave hostname=\";"                      << endl;
   slavemacro << "      cout << hn << \", ordinal=\" << ord;"                             << endl;
   slavemacro << "      cout << \" in file production list.\" << endl;"                   << endl;
   slavemacro << "      cout << \"Make sure the proof.conf contains the \";"              << endl;
   slavemacro << "      cout << \"correct slave hostnames.\" << endl;"                    << endl;
   slavemacro << "   }"                                                                   << endl;
   slavemacro << "}"                                                                      << endl;

   slavemacro.close();

   gProof->Load("build_trees.C+");

   TString cmd = "build_trees(\"";
   cmd += basedir;
   cmd += "\",";
   cmd += events_per_file;
   cmd += ",";
   cmd += files_per_node;
   cmd += ")";

   if (gProof->Exec(cmd)<0) return kFALSE;

   return kTRUE;
}
