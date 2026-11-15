#include "header.h"

#include "TClass.h"
#include "THashTable.h"
#include "TSystem.h"

#include <iostream>
#include <initializer_list>
#include <set>

namespace {
int checkClass(const char *rootName, std::initializer_list<const char *> classNames)
{
   int errors = 0;
   auto rootcl = TClass::GetClass(rootName);
   if (!rootcl) {
      std::cerr << "Could not find TClass for " << rootName << '\n';
      return 1;
   }

   THashTable missing;
   rootcl->GetMissingDictionaries(missing, true);

   std::set<TClass *> expected;
   for (auto className : classNames) {
      auto cl = TClass::GetClass(className);
      if (!cl) {
         std::cerr << "Could not find TClass for " << className << '\n';
         ++errors;
      } else if (!cl->HasDictionary()) {
         expected.insert(cl);
      }
   }

   for (auto cl : expected) {
      if (cl == nullptr || !missing.FindObject(cl)) {
         std::cerr << rootName << ": missing dictionary for " << (cl ? cl->GetName() : "<unknown class>")
                   << " was not reported\n";
         ++errors;
      }
   }
   for (auto cl : TRangeDynCast<TClass>(missing)) {
      if (!cl) {
         std::cerr << rootName << ": unexpected non-TClass missing dictionary report\n";
         ++errors;
      } else if (!expected.count(cl)) {
         std::cerr << rootName << ": unexpected missing dictionary report for " << cl->GetName() << '\n';
         ++errors;
      }
   }
   return errors;
}

int checkSelection(int mask)
{
   const char *names[] = {"pair<edm::Ptr<l1t::io_v1::L1Candidate>,float>",
                          "edm::Ptr<l1t::io_v1::PFCandidate>",
                          "edm::Ref<vector<l1t::io_v1::SAMuon>,2>",
                          "edm::Ptr<reco::io_v1::GsfTrack>",
                          "edm::Ptr<ticl::io_v1::Trackster>",
                          "vector<EcalEBTriggerPrimitiveSample>",
                          "vector<pair<edm::Ptr<l1t::io_v1::L1Candidate>,float> >",
                          "vector<edm::Ptr<l1t::io_v1::PFCandidate> >",
                          "vector<l1ct::io_v1::JetTagClass>",
                          "vector<edm::Ref<vector<l1t::io_v1::SAMuon>,2> >",
                          "vector<l1t::io_v1::SAMuon>",
                          "vector<edm::Ptr<reco::io_v1::GsfTrack> >",
                          "vector<edm::Ptr<ticl::io_v1::Trackster> >",
                          "vector<array<unsigned int,2> >"};
   int errors = 0;
   for (unsigned int i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
      const bool expected = (mask & (1 << i)) != 0;
      auto cl = TClass::GetClass(names[i]);
      if (!cl || cl->HasDictionary() != expected) {
         std::cerr << "Selection mask " << mask << ": unexpected dictionary state for "
                   << (cl ? cl->GetName() : names[i]) << '\n';
         ++errors;
      }
   }
   const char *rootNames[] = {"TriggerPrimitive", "Decoder", "Puppi", "Muons", "Candidate"};
   for (auto name : rootNames) {
      auto cl = TClass::GetClass(name);
      if (!cl || !cl->HasDictionary()) {
         std::cerr << "Selection did not generate a dictionary for " << name << '\n';
         ++errors;
      }
   }
   return errors;
}
} // namespace

int checkMissingDictionaries(int mask, const char *dictionary)
{
   if (gSystem->Load(dictionary) < 0) {
      std::cerr << "Could not load dictionary library " << dictionary << '\n';
      return 1;
   }

   int errors = 0;
   errors += checkClass("TriggerPrimitive",
                        {"TriggerPrimitive", "vector<EcalEBTriggerPrimitiveSample>", "EcalEBTriggerPrimitiveSample"});
   errors += checkClass("Decoder", {"Decoder", "vector<pair<edm::Ptr<l1t::io_v1::L1Candidate>,float> >",
                                    "edm::Ptr<l1t::io_v1::L1Candidate>", "l1t::io_v1::L1Candidate"});
   errors +=
      checkClass("Puppi", {"Puppi", "vector<edm::Ptr<l1t::io_v1::PFCandidate> >", "edm::Ptr<l1t::io_v1::PFCandidate>",
                           "l1t::io_v1::PFCandidate", "vector<l1ct::io_v1::JetTagClass>", "l1ct::io_v1::JetTagClass"});
   errors += checkClass("Muons",
                        {"Muons", "vector<edm::Ref<vector<l1t::io_v1::SAMuon>,2> >",
                         "edm::Ref<vector<l1t::io_v1::SAMuon>,2>", "vector<l1t::io_v1::SAMuon>", "l1t::io_v1::SAMuon"});
   errors += checkClass("Candidate", {"Candidate", "vector<edm::Ptr<reco::io_v1::GsfTrack> >",
                                      "edm::Ptr<reco::io_v1::GsfTrack>", "reco::io_v1::GsfTrack",
                                      "vector<edm::Ptr<ticl::io_v1::Trackster> >", "edm::Ptr<ticl::io_v1::Trackster>",
                                      "ticl::io_v1::Trackster", "vector<array<unsigned int,2> >"});
   errors += checkSelection(mask);
   return errors;
}
