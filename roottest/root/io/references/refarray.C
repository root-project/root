#include "TRef.h"

#include "TRefArray.h"
#include "TFile.h"

#include <iostream>
using namespace std;

const char *fname = "refarray.root";


void write(const char *filename = fname)
{
   TFile *f = TFile::Open(filename,"RECREATE");
   TObject *obj = new TObject();
   TRef ref = obj;
   f->WriteObject(obj,"therefobj");
   f->WriteObject(&ref,"theref");
   delete obj;
   obj = new TObject();
   f->WriteObject(obj,"theobj");
   delete obj;
   delete f;
}

bool read(const char *filename = fname)
{
   TFile *f = TFile::Open(filename,"READ");
   TObject *obj;
   TRef *ref;
   
   f->GetObject("therefobj",obj);
   f->GetObject("theref",ref);
   
   if (obj==0 || ref==0) {
      cout << "Could read one of the objects (obj=" << (void*)obj;
      cout << " ref=" << (void*)ref << ")\n";
      return false;
   }

   if (ref->GetObject() != obj) {
      cout << "The reference is not pointing to the object: ";
      cout << "obj=" << (void*)obj;
      cout << " ref= " << (void*)ref->GetObject() << endl;
      return false;
   }
   

   TRefArray wrong;
   wrong.Add(new TObject());
   TRefArray good( TProcessID::GetProcessWithUID(obj) );
   
   wrong.Add(obj);
   good.Add(obj);

   TObject *obj2;
   f->GetObject("theobj",obj2);

   wrong.Add(obj2);
   good.Add(obj2);

   if ( good.GetEntries()!=1 ) {
      cout << "Old file TRefArray does not have the right number of entries. It is " << good.GetEntries() <<
         " but should be 1\n";
   } else if ( good.At(0)!=obj) {
      cout << "Old file TRefArray does not point to the correct object\n";
   }
   if ( wrong.GetEntries()!=2 ) {
      cout << "New process TRefArray does not have the right number of entries. It is " << wrong.GetEntries() <<
         " but should be 2\n";
   } else if ( wrong.At(1)!=obj2) {
      cout << "New process TRefArray does not point to the correct object\n";
   }

   return (wrong.GetEntries()==2 && good.GetEntries()==1 && good.At(0)==obj && wrong.At(1)==obj2);
}

int refarray(int mode = 2, const char *filename = fname)
{
   if (mode &1) write(filename);
   if (mode &2) return !read(filename);
   return 0;
}

   
