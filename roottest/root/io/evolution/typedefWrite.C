#include <TObject.h>
#include <list>

namespace std {} using namespace std;

class UHTDecay {};
typedef list<UHTDecay> DecayListType;

class UHTTimeFitter : public TObject{
 public:

#if defined(__CINT__) && 0
  list<UHTDecay> decayList;
#else
  DecayListType decayList;
#endif

  ClassDef(UHTTimeFitter,1);
};

#include <TFile.h>
void typedefWrite(const char *filename = "typedef.root") 
{
   TFile *f = new TFile(filename,"RECREATE");
   UHTTimeFitter *u = new UHTTimeFitter;
   u->Write("myobject");
   f->Write();
   delete f;
}
