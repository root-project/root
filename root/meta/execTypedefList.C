#include "TROOT.h"
#include "TDataType.h"

namespace  myNamespace { struct MyClass {}; }

int check(const char *name, const char *target)
{
   TObject *dobj;
   TCollection *l = gROOT->GetListOfTypes();

   dobj = l->FindObject(name);
   if (!dobj) {
      fprintf(stderr,"Couldn't find the TDataType for %s\n",name);
      return 1;
   }
   if (strcmp(dobj->GetName(),name) != 0) {
      fprintf(stderr,"Found the wrong TDataType for: %s when searching for %s\n",dobj->GetName(),name);
      return 2;
   }
   if (strcmp(((TDataType*)dobj)->GetTypeName(),target) != 0) {
      fprintf(stderr,"Found the wrong TDataType for %s target is %s rather than %s\n",
              name,((TDataType*)dobj)->GetTypeName().Data(),target);
      return 3;
   }
   return 0;
}

int check_target(const char *name, const char *target)
{
   TObject *dobj;
   TCollection *l = gROOT->GetListOfTypes();

   dobj = l->FindObject(name);
   if (!dobj) {
      fprintf(stderr,"Couldn't find the TDataType for %s\n",name);
      return 1;
   }
   //if (strcmp(dobj->GetName(),name) != 0) {
   //   fprintf(stderr,"Found the wrong TDataType for: %s when searching for %s\n",dobj->GetName(),name);
   //   return 2;
   //}
   if (strcmp(((TDataType*)dobj)->GetTypeName(),target) != 0) {
      fprintf(stderr,"Found the wrong TDataType for %s target is %s rather than %s\n",
              name,((TDataType*)dobj)->GetTypeName().Data(),target);
      return 3;
   }
   return 0;
}

int check_missing(const char *name)
{
   TObject *dobj;
   TCollection *l = gROOT->GetListOfTypes();

   dobj = l->FindObject(name);
   if (dobj) {
      fprintf(stderr,"Surpringly found the TDataType for %s typedef to %s\n",
              name,((TDataType*)dobj)->GetTypeName().Data());
      return 1;
   }
   return 0;
}

int check_exist(const char *name)
{
   TObject *dobj;
   TCollection *l = gROOT->GetListOfTypes();

   dobj = l->FindObject(name);
   if (!dobj) {
      fprintf(stderr,"Couldn't find the TDataType for %s\n",name);
      return 1;
   }
   return 0;
}

#include <iostream>
#include <fstream>

int check_file(const char *filename, int expected_count)
{
   std::ifstream f(filename);
   int res = 0;
   int count = 0;
   int found = 0;
   char what[1000];
   while( f.getline(what,1000) ) {
      ++count;
      if (what[0]=='#') continue;
      int lres = check_exist(what);
      if (lres) {
         fprintf(stderr,"Failed on count == %d in %s\n",count,filename);
         res = lres;
      }
      ++found;
   }
   if (found != expected_count) {
      fprintf(stderr,"Found only %d typedefs (expected %d)\n",found, expected_count);
      if (!res)
         res = 4;
   }
   f.close();
   return res;
}



int execTypedefList() {
   int res;

   // Just in case we have a small pch.
   const char *whatToLoad [] = { "TPainter3dAlgorithms", "TLego", "TAuthenticate", "TProofDraw", "TChainIndex", "TF1", "TGeoBoolNode", "TShape", "TXMLEngine" };
   for(unsigned int i = 0 ; i < sizeof(whatToLoad) / sizeof(const char*); ++i) {
      gInterpreter->AutoLoad(whatToLoad[i]);
      gInterpreter->AutoParse(whatToLoad[i]);
   }

   res = check("int","int"); if (res) return res;
   res = check("Int_t","int"); if (res) return res;
   res = check("UInt_t","unsigned int"); if (res) return res;
   res = check("vector<int>::value_type","int"); if (res) return res;
   res = check("vector<int>::reference","int"); if (res) return res;
   res = check("Option_t","char"); if (res) return res;
   res = check("KeySym_t","unsigned long"); if (res) return res;
   res = check("TBuffer::CacheList_t","vector<TVirtualArray*>"); if (res) return res;
   res = check_missing("TBuffer::CacheList_notAtype"); if (res) return res;

   // The iterator typedef is now desugared.
   // res = check("vector<myNamespace::MyClass*>::const_iterator","vector<myNamespace::MyClass*>::const_iterator"); if (res) return res;

   res = check_target("std::map<std::string, int>::key_type","string"); if (res) return res;
   res = check_target("std::map<std::string, int>::value_type","pair<const string,int>"); if (res) return res;
   // The iterator typedef is now desugared.
   // res = check_target("std::list<std::string>::const_iterator","list<string>::const_iterator"); if (res) return res;

#ifdef _MSC_VER
   res = check_file("typelist_win32.v5.txt",348); if (res) return res;
   #if __cplusplus > 201402L
      res = check_file("typelist_win32.v6.cxx17.txt",1408); if (res) return res;
   #else
      res = check_file("typelist_win32.v6.txt",1420); if (res) return res;
   #endif
#else
   res = check_file("typelist.v5.txt",349); if (res) return res;
   res = check_file("typelist.v6.txt",1465); if (res) return res;
#endif

   return 0;
}
