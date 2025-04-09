#include "TFile.h"

class empty {};
class event : public TObject, public empty {
   ClassDef(event,1);
};

void t02() {
   TFile *file = new TFile("testempty.root","RECREATE");
   event *e = new event;
   e->Write("emptyevent");
   file->Write();
}
