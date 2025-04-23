#ifndef MAPTOVECTOR_H
#define MAPTOVECTOR_H

#include "TFile.h"
#include "Riostream.h"
#include "TString.h"

#include <map>
#include <vector>
#include <list>

using namespace std;

class myclass {
public:
#if defined(CASE_map)
   map<int,double> fCont;
   void insert( pair<int,double> val ) {
      fCont.insert(val);
   }
#elif defined(CASE_multimap)
   multimap<int,double> fCont;
   void insert( pair<int,double> val ) {
      fCont.insert(val);
   }
#elif defined(CASE_vector)
   vector<pair<int,double> > fCont;
   void insert( pair<int,double> val ) {
      fCont.push_back(val);
   }
#elif defined(CASE_list)
   list<pair<int,double> > fCont;
   void insert( pair<int,double> val ) {
      fCont.push_back(val);
   }
#endif

   void Seed() {
      insert(make_pair<int,double>(2,4.0));
      insert(make_pair<int,double>(7,14.0));
   }
};

template <typename T>
T diff(const T& left, const T& right)
{
   if (left > right) {
      return left - right;
   } else {
      return right - left;
   }
}

template <typename T> 
bool Compare(const T& ref, const T&obj, const char *name) {
   bool result = diff(ref,obj) <= 1;
   if (!result) {
      cout << "Value of " << name << " is " << obj << " instead of " << ref << endl;
   }
   return result;
}

template <typename T> 
bool ComparePair(const T& ref, const T&obj) {
   bool result = true;
   result = Compare(ref.first,obj.first,"first");
   result = Compare(ref.second,obj.second,"second") && result;
   return result;
}

template <typename T> 
bool CompareCont(const T& ref, const T&obj) {
   if ( ref.size() != obj.size() ) {
      cout << "The size are different (" << ref.size() << " vs " << obj.size() << ")\n";
      return false;
   }
   typename T::const_iterator refiter = ref.begin();
   typename T::const_iterator objiter = obj.begin();
   bool result = true;
   for( ; refiter != ref.end() && objiter != obj.end(); ++refiter, ++objiter ) {
      result = ComparePair( *refiter, *objiter) && result;
   }
   if (!result) {
      cout << "The content are different\n";
   }
   return result;
}



bool Compare(myclass *ref, myclass *obj) {
   bool result = true;
   result = CompareCont( ref->fCont, obj->fCont );
   return result;
};

void write(const char *filename)
{
   TFile *f = TFile::Open(filename,"RECREATE");
   myclass *c = new myclass;
   c->Seed();
   f->WriteObject(c,"myobj");
   f->Write();
}

bool readfile(const char *filename, Bool_t checkValue = kTRUE)
{
   TFile *f = TFile::Open(filename,"READ");
   if (f==0) return false;

   cout << "Reading " << filename << endl;
   myclass *c; f->GetObject("myobj",c);
   myclass *ref = new myclass;
   ref->Seed();

   if (checkValue) {
      return Compare(ref,c);
   } else {
      return c != 0;
   }
}

#ifdef __MAKECINT__
#pragma link C++ class myclass+;
// #pragma link C++ class pair<int,double>+;
#pragma link C++ function readfile;
#pragma link C++ function write;
#endif

#endif
