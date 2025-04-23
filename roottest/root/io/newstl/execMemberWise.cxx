#include "TObject.h"
#include "TFile.h"
#include "TString.h"
#include "TVirtualStreamerInfo.h"

#define NewMember
#include "MemberWise.h"

void execMemberWise() {
 
   TFile *f = TFile::Open("memberwise.root","READ");
   Holder *pointer;
   cout << "Reading in object-wise mode\n";
   f->GetObject("objwise",pointer);
   cout << "Reading in member-wise mode\n";
   f->GetObject("memwise",pointer);
}
