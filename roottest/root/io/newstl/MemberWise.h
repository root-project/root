#include "Riostream.h"
#include "TString.h"
#include "TBuffer.h"
#include <vector>
#include "TClass.h"

class Member {
public:
   TString fName;
   UInt_t  fIndex;
   
   Member() : fName("None"),fIndex(0) {}
   Member(const Member &rhs) : fName(rhs.fName), fIndex(rhs.fIndex) {}
   Member(const char *name, UInt_t index) : fName(name), fIndex(index) {}
   
   virtual ~Member() {}
   ClassDef(Member,2);
};

void Member::Streamer(TBuffer &buf) {
   if (buf.IsReading()) {
      buf.ReadClassBuffer(Class(),this);
      cout << "Reading member : " << fName.Data() << " index: " << fIndex;
      cout << endl;
   } else {
      cout << "Writing member : " << fName.Data() << " index: " << fIndex;
      cout << endl;
      buf.WriteClassBuffer(Class(),this);
   }
}

class WithStreamerOnly {
public:
public:
   Member fOne;
   Member fTwo;
   
   WithStreamerOnly() : fOne("WithStreamerOnly Default one",0),fTwo("WithStreamerOnly Default two",0) {}
   WithStreamerOnly(UInt_t index) : fOne("WithStreamerOnly fOne",index),fTwo("WithStreamerOnly fTwo",index) {}
   
   void Streamer(TBuffer &buf) {
      if (buf.IsReading()) {
         cout << "Reading a WithStreamerOnly\n";
         buf.ReadClassBuffer(TClass::GetClass("WithStreamerOnly"),this);
      } else {
         buf.WriteClassBuffer(TClass::GetClass("WithStreamerOnly"),this);
      }
   }      
};

class WithAttachedStreamer {
public:
   Member fOne;
   Member fTwo;

   WithAttachedStreamer() : fOne("WithAttachedStreamer Default one",0),fTwo("WithAttachedStreamer Default two",0) {}
   WithAttachedStreamer(UInt_t index) : fOne("WithAttachedStreamer fOne",index),fTwo("WithAttachedStreamer fTwo",index) {}
};

class WithExplicitCannotSplit : WithAttachedStreamer {
public:
   WithExplicitCannotSplit() {}
   WithExplicitCannotSplit(UInt_t index) : WithAttachedStreamer(index) {}
};

void WithAttachedStreamerStreamer(TBuffer &buf,void *obj) {
   if (buf.IsReading()) {
      cout << "Reading a WithAttachedStreamer\n";
      buf.ReadClassBuffer(TClass::GetClass("WithAttachedStreamer"),obj);
   } else {
      buf.WriteClassBuffer(TClass::GetClass("WithAttachedStreamer"),obj);
   }
}      

RootStreamer(WithAttachedStreamer,WithAttachedStreamerStreamer);

class Content {
public:
   Member fOne;
   Member fTwo;
#ifdef NewMember
   UInt_t fData;
#endif
  
   virtual ~Content() {}
 
#ifdef NewMember
   
   Content() : fOne("Default one",0),fTwo("Default two",0), fData(0) {}
   Content(UInt_t index) : fOne("fOne",index),fTwo("fTwo",index), fData(index*10) {}
   
   ClassDef(Content,3);
#else
   Content() : fOne("Default one",0),fTwo("Default two",0) {}
   Content(UInt_t index) : fOne("fOne",index),fTwo("fTwo",index) {}

   ClassDef(Content,2);
#endif
};

class NeverWritten {
   UInt_t fValue;
#ifdef NewMember
   UInt_t fData;
public:
   NeverWritten() : fValue(0),fData(0) {}

   //   ClassDef(NeverWritten,3);
#else
public:
   NeverWritten() : fValue(0) {}
   //   ClassDef(NeverWritten,2);
#endif
};

class Holder {
public:
   Holder() : fPointer(0) {};
   ~Holder() { delete fPointer; }
   
   void Init() {
      fNormal.push_back(Content(1));
      fNormal.push_back(Content(2));
      fPointer = new std::vector<Content>;
      fPointer->push_back(Content(3));
      fPointer->push_back(Content(4));
      fHasCustomStreamer.push_back(Member("top-1",11));
      fHasCustomStreamer.push_back(Member("top-2",12));
      fNotSplit.push_back(Content(5));
      fNotSplit.push_back(Content(6));
      fAttachedStreamer.push_back(WithAttachedStreamer(7));
      fAttachedStreamer.push_back(WithAttachedStreamer(8));
      fStreamerOnly.push_back(WithStreamerOnly(9));
      fStreamerOnly.push_back(WithStreamerOnly(10));
      fCannotSplit.push_back(WithExplicitCannotSplit(11));
      fCannotSplit.push_back(WithExplicitCannotSplit(12));
      fStreamerOnlyNoReq.push_back(WithStreamerOnly(13));
      fStreamerOnlyNoReq.push_back(WithStreamerOnly(14));
   }
   
   std::vector<Content>  fNormal;
   std::vector<Content> *fPointer;
   std::vector<Member>   fHasCustomStreamer;
   std::vector<Content>  fNotSplit; //||
   std::vector<WithAttachedStreamer> fAttachedStreamer;
   std::vector<WithStreamerOnly> fStreamerOnly; //!
   std::vector<WithExplicitCannotSplit> fCannotSplit;
   std::vector<WithStreamerOnly> fStreamerOnlyNoReq; //

   std::vector<NeverWritten> fAlwaysEmpty;
};

#ifdef __MAKECINT__
#pragma link C++ class Member-;
#pragma link C++ class Content+;
#pragma link C++ class Holder+;
#pragma link C++ class WithAttachedStreamer+;
#pragma link C++ class WithExplicitCannotSplit+;
#pragma link C++ class WithStreamerOnly-;
#pragma link C++ class NeverWritten+;
#endif
