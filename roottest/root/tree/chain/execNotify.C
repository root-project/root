#include "TObject.h"
#include <stdio.h>

class Notified : public TObject {
  unsigned int fValue = 0;

public:
  Notified(unsigned int value) : fValue (value) {}
  Bool_t Notify() override { printf("Notifying #%d\n", fValue); return true; }
};

#include "TList.h"
#include "TObjArray.h"
#include "TTree.h"



template <typename Collection>
void execNotifyImpl() {
  Collection l;
  TTree t("t","t");
  l.Add(new Notified(1));
  t.SetNotify(&l);

  printf("First notify\n");
  t.LoadTree(0);

  l.Add(nullptr);
  l.Add(new Notified(2));
  printf("Second notify\n");
  t.LoadTree(0);
}

void execNotify() {
  execNotifyImpl<TList>();
  execNotifyImpl<TObjArray>();
 
}
