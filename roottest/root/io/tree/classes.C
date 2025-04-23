#include "TTree.h"

class TPhysObj : public TObject {
public:
   ClassDef(TPhysObj,1)
};

class TEmcl : public TPhysObj {
public:
   int e;
   ClassDef(TEmcl,1)
};

class TNonPhysObj : public TObject {
public:
   ClassDef(TNonPhysObj,0)
};

class TNonEmcl : public TNonPhysObj {
public:
   int e;
   ClassDef(TNonEmcl,1)
};


