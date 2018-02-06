#include "TWebSnapshot.h"

#include "TString.h"

TWebSnapshot::TWebSnapshot() :
   TObject(),
   fObjectID(),
   fOption(),
   fKind(0),
   fSnapshot(0),
   fOwner(kFALSE)
{
}

TWebSnapshot::~TWebSnapshot()
{
   SetSnapshot(0,0);
}


void TWebSnapshot::SetSnapshot(Int_t kind, TObject *shot, Bool_t owner)
{
   if (fSnapshot && fOwner) delete fSnapshot;
   fKind = kind;
   fSnapshot = shot;
   fOwner = owner;
}

void TWebSnapshot::SetObjectIDAsPtr(void *ptr)
{
   UInt_t hash = TString::Hash(&ptr, sizeof(ptr));
   SetObjectID(TString::UItoa(hash,10));
}

// ========================================

TPadWebSnapshot::~TPadWebSnapshot()
{
   for (unsigned n=0; n<fPrimitives.size(); ++n)
      delete fPrimitives[n];
   fPrimitives.clear();
}

