#include "TTree.h"

class TPhysObj : public TObject {
public:
   ClassDefOverride(TPhysObj,1)
};

class TEmcl : public TPhysObj {
public:
   int e;
   ClassDefOverride(TEmcl,1)
};

class TNonPhysObj : public TObject {
public:
   ClassDefOverride(TNonPhysObj,0)
};

class TNonEmcl : public TNonPhysObj {
public:
   int e;
   ClassDefOverride(TNonEmcl,1)
};


