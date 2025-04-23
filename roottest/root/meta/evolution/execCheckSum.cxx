#include <vector>
#include "TObject.h"
#include "TClass.h"

typedef long MyLong_t;

template <typename T> class UserTmplt
{
public:
   T fValue;
};

class HasTypeDef
{
public:
   HasTypeDef() : fInt1(1),fInt2(2) {}
   virtual ~HasTypeDef() {}

   typedef TNamed Named_t;
   std::vector<Int_t>    fVec1;
   std::vector<MyLong_t> fVec2;
   std::vector<Named_t>  fVec3;
   UserTmplt<Int_t>      fUser1;
   UserTmplt<Named_t>    fUser2;
   MyLong_t fInt1;
   Int_t    fInt2;
   void Print() {
      fprintf(stdout,"The %s object has:\n",Class()->GetName());
      fprintf(stdout,"  fVec1.size() : %ld\n", (long)fVec1.size());
      fprintf(stdout,"  fVec2.size() : %ld\n", (long)fVec2.size());
      fprintf(stdout,"  fVec3.size() : %ld\n", (long)fVec3.size());
      fprintf(stdout,"  fInt1        : %ld\n", fInt1);
      fprintf(stdout,"  fInt2        : %d\n", fInt2);
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
   if (!f) {
      fprintf(stderr,"Could not open %s\n",filename);
      return;
   }
   TClass *c = TClass::GetClass("HasTypeDef");
   fprintf(stdout,"In memory the checksum is: 0x%x\n",c->GetCheckSum());
   TVirtualStreamerInfo *info =  (TVirtualStreamerInfo*)f->GetStreamerInfoList()->FindObject("HasTypeDef");
   fprintf(stdout,"On file   the checksum is: 0x%x\n",info->GetCheckSum());

   HasTypeDef *obj;
   f->GetObject("obj",obj);
   if (!obj) {
      Error("readFile","Could not read the object");
   } else {
      obj->Print();
   }
   delete f;
}

int execCheckSum()
{
   readFile("checksum_v53418.root");
   readFile("checksum_v5.root");
   readFile("checksum_v6.root");
   return 0;
}
