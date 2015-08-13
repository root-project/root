// Build SampleClasses.h with the command '.L SampleClasses.h+'
#ifndef Sample_classes
#define Sample_classes

#include "TObject.h"
#include "TClonesArray.h"
#include <vector>

#ifdef __MAKECINT__
#pragma link C++ class vector<EventData>+;
#pragma link C++ class vector<Particle>+;
#pragma link C++ class vector<ClassWithArray>+;
#endif

class ClassC : public TObject {
private:
   Float_t fPx;
   Int_t   fEv;
public:
   ClassC(Float_t fPx_, Int_t fEv_) : fPx(fPx_), fEv(fEv_) { }
   ClassC() : ClassC(0, 0) { }
   virtual ~ClassC() { }
   
   Float_t GetPx() const { return fPx; }
   Int_t   GetEv() const { return fEv; }
   void Set(Float_t fPx_, Int_t fEv_) { fPx = fPx_; fEv = fEv_; }
   
   ClassDef(ClassC, 1);
};

class ClassB : public TObject {
private:
   ClassC  fC;
   Float_t fPy;
public:
   ClassB(Float_t fPx_, Int_t fEv_, Float_t fPy_) : fC(fPx_, fEv_), fPy(fPy_) { }
   ClassB() : ClassB(0, 0, 0) { }
   virtual ~ClassB() { }
   
   ClassC  GetC() const { return fC; }
   Float_t GetPy() const { return fPy; }
   void Set(Float_t fPx_, Int_t fEv_, Float_t fPy_) {
      fC.Set(fPx_, fEv_);
      fPy = fPy_;
   }
   
   ClassDef(ClassB, 1);
};

class ClassWithArray : public TObject {
public:
   Int_t arr[10];
   
   ClassWithArray() { }
   virtual ~ClassWithArray() { }
   
   ClassDef(ClassWithArray, 1);
};

class ClassWithVector : public TObject {
public:
   std::vector<Int_t> vec;
   std::vector<Bool_t> vecBool;
   
   ClassWithVector() { }
   virtual ~ClassWithVector() { }
   
   ClassDef(ClassWithVector, 1);
};

class ClassWithClones : public TObject {
public:
   TClonesArray arr;

   ClassWithClones() : arr("Particle", 5) { }

   ClassDef(ClassWithClones, 1);
};

class Particle : public TObject {
public:
   Particle() { }
   double fPosX,fPosY,fPosZ;

   ClassDef(Particle,1);
};

class EventData : public TObject {
public:
   std::vector<Particle> fParticles;
   int fEventSize;

   void SetSize() {
      fEventSize = sizeof(EventData) + fParticles.size() * sizeof(Particle);
   }
   void Clear(const char* = "") {
      fParticles.clear();
   }
   void AddParticle(const Particle& p) { fParticles.push_back(p); }

   ClassDef(EventData,1);
};

#endif
