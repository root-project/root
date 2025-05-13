#include "TFile.h"
#include "TNamed.h"
#include <cstdio>


void runXmlDir()
{
   TFile *f = TFile::Open("filesubdirs.xml", "recreate");
   if (!f) {
      printf("Cannot create filesubdirs.xml\n");
      return;
   }

   TDirectory *dir1 = f->mkdir("dir1");
   TNamed *n1 = new TNamed("name1","title1");
   dir1->WriteObject(n1, "obj1");

   TDirectory *dir2 = dir1->mkdir("dir2");
   TNamed *n2 = new TNamed("name2","title2");
   dir2->WriteObject(n2, "obj2");


   TDirectory *dir3 = dir2->mkdir("dir3");
   TNamed *n3 = new TNamed("name3","title3");
   dir3->WriteObject(n3, "obj3");

   TNamed *n4 = new TNamed("name4","title4");
   dir2->WriteObject(n4, "obj4");

   delete f;

   delete n1; n1 = nullptr;
   delete n2; n2 = nullptr;
   delete n3; n3 = nullptr;
   delete n4; n4 = nullptr;

   f = TFile::Open("filesubdirs.xml");
   if (!f) {
      printf("Cannot open filesubdirs.xml for reading\n");
      return;
   }

   f->GetObject("dir1/dir2/dir3/obj3", n3);
   f->GetObject("dir1/dir2/obj2", n2);
   f->GetObject("dir1/obj1", n1);
   f->GetObject("dir1/dir2/obj4", n4);

   if (n1) printf("dir1/obj1: %s %s\n", n1->GetName(), n1->GetTitle());
      else printf("Fail to read dir1/obj1\n");
   if (n2) printf("dir1/dir2/obj2: %s %s\n", n2->GetName(), n2->GetTitle());
      else printf("Fail to read dir1/dir2/obj2\n");
   if (n3) printf("dir1/dir2/dir3/obj3: %s %s\n", n3->GetName(), n3->GetTitle());
      else printf("Fail to read dir1/dir2/dir3/obj3\n");
   if (n4) printf("dir1/dir2/obj4: %s %s\n", n4->GetName(), n4->GetTitle());
      else printf("Fail to read dir1/dir2/obj4\n");

   delete f;
}
