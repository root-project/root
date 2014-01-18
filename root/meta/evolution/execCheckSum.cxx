#include <vector>
#include "TObject.h"
#include "TClass.h"

typedef long MyLong_t;

class HasTypeDef
{
public:
   HasTypeDef() : fInt1(1),fInt2(2) {}
   virtual ~HasTypeDef() {}

   std::vector<Int_t>    fVec1;
   std::vector<MyLong_t> fVec2;
   MyLong_t fInt1;
   Int_t    fInt2;
   void Print() {
      fprintf(stderr,"The %s object has:\n",Class()->GetName());
      fprintf(stderr,"  fVec1.size() : %ld\n", fVec1.size());
      fprintf(stderr,"  fVec2.size() : %ld\n", fVec2.size());
      fprintf(stderr,"  fInt1        : %ld\n", fInt1);
      fprintf(stderr,"  fInt2        : %d\n", fInt2);
   }

   ClassDef(HasTypeDef,3);
};

#include "TFile.h"
#include "TError.h"
#include "TVirtualStreamerInfo.h"

void writeFile(const char *filename = "checksum.root")
{
   TFile *f = TFile::Open(filename,"RECREATE");
   HasTypeDef obj;
   f->WriteObject(&obj,"obj");
   f->Write();
   delete f;
}

void readFile(const char *filename = "checksum.root")
{
   TFile *f = TFile::Open(filename,"READ");

   TClass *c = TClass::GetClass("HasTypeDef");
   fprintf(stderr,"In memory the checksum is: 0x%x\n",c->GetCheckSum());
   TVirtualStreamerInfo *info =  (TVirtualStreamerInfo*)f->GetStreamerInfoList()->FindObject("HasTypeDef");
   fprintf(stderr,"On file   the checksum is: 0x%x\n",info->GetCheckSum());

   HasTypeDef *obj;
   f->GetObject("obj",obj);
   if (!obj) {
      Error("readFile","Could not read the object");
   } else {
      obj->Print();
   }
   delete f;
}

