#include "TObject.h"
#include "TFile.h"
#include "TString.h"
#include "TVirtualStreamerInfo.h"

#include "MemberWise.h"

void writeMemberWise() {
   TClass::GetClass("WithExplicitCannotSplit")->SetCanSplit(0);

   Holder holder;
   holder.Init();
   
   TFile *out = TFile::Open("memberwise.root","RECREATE");
   TVirtualStreamerInfo::SetStreamMemberWise(kFALSE);
   cout << "Writing in object-wise mode\n";
   out->WriteObject(&holder,"objwise");
   TVirtualStreamerInfo::SetStreamMemberWise(kTRUE);
   cout << "Writing in member-wise mode\n";
   out->WriteObject(&holder,"memwise");
   delete out;
   
   TFile *f = TFile::Open("memberwise.root","READ");
   Holder *pointer;
   cout << "Reading in object-wise mode\n";
   f->GetObject("objwise",pointer);
   cout << "Reading in member-wise mode\n";
   f->GetObject("memwise",pointer);
}
