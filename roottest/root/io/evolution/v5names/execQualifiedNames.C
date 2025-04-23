#include <Rtypes.h>
#include <vector>

namespace evt {

   namespace rec {

      class Track {
      public:
         Track(const int v = 0) : fValue(v) {}
         int GetVal() const { return fValue; }

      private:
         Int_t fValue;

         ClassDefNV(Track, 1)
      };

   }

   class Event {
   public:
      void MakeTrack(const int v) { fTracks.push_back(rec::Track(v)); }

      const rec::Track& GetTrack(const int i) const { return fTracks.at(i); }

   private:
      std::vector<rec::Track> fTracks;

      ClassDefNV(Event, 2);
   };
   
   
}


#include <TTree.h>
#include <TFile.h>
#include <TError.h>


void write(const char *filename)
{
   evt::Event e;
   evt::Event* ePtr = &e;

   e.MakeTrack(1);
   e.MakeTrack(2);

   TTree t("t", "");
   t.Branch("fEvent", "evt::Event", &ePtr);
   t.Fill();
   TFile f(filename, "RECREATE");
   t.Write();
}

int read(const char *filename)
{
   evt::Event e;
   evt::Event* ePtr = &e;

   TFile f(filename);
   if (f.IsZombie()) {
      Error("read","Could not open file: %s",filename);
      return 1;
   }

   TTree* t = nullptr;
   f.GetObject("t",t);
   if (!t) {
      Error("read","Could not find TTree (t) in file: %s",filename);
      return 2;
   }
   t->GetBranch("fEvent")->SetAddress(&ePtr);
   t->GetEntry(0);
   const bool isExpected = e.GetTrack(0).GetVal() == 1;

   return !isExpected;
}

int execQualifiedNames() {

   return read("qualifiedName_v5.root") + read("qualifiedName_v6.root");
}
