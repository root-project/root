#include <vector>
#include "TTree.h"

class MyHit : public TObject {
private:
  /* only basic types */
   float fdata;
   ClassDefOverride(MyHit,1);
};

class MySubEvent : public TObject {
public:
private:
   std::vector<MyHit> hits;
   ClassDefOverride(MySubEvent,1);
};

class MyEvent : public TObject {
public:
  MyEvent() {}; // def ctor, never called
  // MyEvent(non_root_class* c) : se(c->SubClass()) {...}; // working (copy) ctor
private:
   std::vector<MyHit> hitsintop;
   MySubEvent se;
   ClassDefOverride(MyEvent,1);
};


TTree* vectorInSubObj() {
   MyEvent *ev_ptr = 0;
   TTree* mytree = new TTree("mytree","mytitle");
   mytree->Branch("mybranch", "MyEvent", &ev_ptr);
   return mytree;
} 
