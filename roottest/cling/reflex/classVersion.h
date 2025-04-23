#ifndef CLASSVERSION_H
#define CLASSVERSION_H

#include "TObject.h"

class WithClassVersion {
 public:
   int hasAClassVersion;
};

class WithClassDef {
   ClassDef(WithClassDef, 12);
};

class NoIO {
   ClassDef(NoIO, 0);
};

class ReallyNoDict {

};

class FromTObject: public TObject {
   ReallyNoDict fTransientByComment; //! this is intended to be transient
   ReallyNoDict fTransientBySelection; // this is intended to be transient, marked by selection.xml

   ClassDef(FromTObject, 13);
};

class NoDictionary {
   ClassDef(NoDictionary, 14);
};
TClass* NoDictionary::Class() { return nullptr; }
const char* NoDictionary::Class_Name() { return nullptr; }
TClass* NoDictionary::Dictionary() { return nullptr; }
void NoDictionary::Streamer(TBuffer&) {}
int NoDictionary::ImplFileLine() { return -1; }
const char* NoDictionary::ImplFileName() { return nullptr; }

class NoDictionaryTObj: public TObject {
};

#include <vector>
#include <string>

template <class T> class MyTemp {
   T value;
};

template class std::vector<std::string>;
template class std::vector<WithClassVersion>;
template class MyTemp<std::vector<std::string> >;

template <class T> class TemplateWithVersion {
   T value;
   public: static Int_t Class_Version() { return 10; }
};

template class TemplateWithVersion<int>;
template class TemplateWithVersion<double>;

#endif


