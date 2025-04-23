class Event {
public:
  int fValue;
};

#include "TFile.h"

void writeFile() {
   Event e;
   TFile *f = new TFile("tc_state.root","RECREATE");
   f->WriteObject(&e,"event");
   f->Close();
}

