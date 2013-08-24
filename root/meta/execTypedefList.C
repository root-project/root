#include "TROOT.h"
#include "TDataType.h"


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
      fprintf(stderr,"Found the wrong TDataType for int: %s\n",dobj->GetName());
      return 2;
   }
   if (strcmp(((TDataType*)dobj)->GetTypeName(),target) != 0) {
      fprintf(stderr,"Found the wrong TDataType for %s target is %s rather than %s\n",
              name,((TDataType*)dobj)->GetTypeName(),target);
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
      fprintf(stderr,"Surpringly found the TDataType for %s typedef to %s\n",name,((TDataType*)dobj)->GetTypeName());
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

   res = check("int","int"); if (res) return res;
   res = check("Int_t","int"); if (res) return res;
   res = check("UInt_t","unsigned int"); if (res) return res;
   res = check("vector<int>::value_type","int"); if (res) return res;
   res = check("vector<int>::reference","int"); if (res) return res;
   res = check("Option_t","char"); if (res) return res;
   res = check("KeySym_t","unsigned long"); if (res) return res;
   res = check("TBuffer::CacheList_t","vector<TVirtualArray*>"); if (res) return res;
   res = check_missing("TBuffer::CacheList_notAtype"); if (res) return res;
   
   res = check_file("typelist.v5.txt",360); if (res) return res;
   res = check_file("typelist.v6.txt",1700); if (res) return res;

   return 0;
}
