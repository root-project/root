#ifndef TESTREFOBJ_H
#define TESTREFOBJ_H

#include "TRef.h"
#include "TRefArray.h"

// https://its.cern.ch/jira/browse/ROOT-7249
class TestRefObj : public TObject {
protected:
   TRefArray lChildren;

public:
   TestRefObj() : lChildren() {}
   virtual ~TestRefObj() {}
   virtual void Clear(Option_t *opt = "C") { lChildren.Clear(opt); }
   virtual void SetChild(TestRefObj *aChild) { lChildren.Add(aChild); }
   virtual const TestRefObj *GetChild(Int_t idx = 0) const { return static_cast<TestRefObj *>(lChildren.At(idx)); }
   virtual Bool_t HasChild() const { return (lChildren.GetEntriesFast() > 0); }
   ClassDef(TestRefObj, 1);
};

#endif
