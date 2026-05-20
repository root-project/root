#include "TFile.h"

class base {
   ClassDef(base,1);
};
class event : public TObject, public base {
   ClassDef(event,1);
};

void t03() {
   TFile *file = new TFile("testbase.root","RECREATE");
   event *e = new event;
   e->Write("emptyevent");
   file->Write();
}
