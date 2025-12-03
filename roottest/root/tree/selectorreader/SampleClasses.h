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
   ~ClassC() override { }
   
   Float_t GetPx() const { return fPx; }
   Int_t   GetEv() const { return fEv; }
   void Set(Float_t fPx_, Int_t fEv_) { fPx = fPx_; fEv = fEv_; }
   
   ClassDefOverride(ClassC, 1);
};

class ClassB : public TObject {
private:
   ClassC  fC;
   Float_t fPy;
public:
   ClassB(Float_t fPx_, Int_t fEv_, Float_t fPy_) : fC(fPx_, fEv_), fPy(fPy_) { }
   ClassB() : ClassB(0, 0, 0) { }
   ~ClassB() override { }
   
   ClassC  GetC() const { return fC; }
   Float_t GetPy() const { return fPy; }
   void Set(Float_t fPx_, Int_t fEv_, Float_t fPy_) {
      fC.Set(fPx_, fEv_);
      fPy = fPy_;
   }
   
   ClassDefOverride(ClassB, 1);
};

class ClassWithArray : public TObject {
public:
   Int_t arr[10];
   
   ClassWithArray() { }
   ~ClassWithArray() override { }
   
   ClassDefOverride(ClassWithArray, 1);
};

class ClassWithVector : public TObject {
public:
   std::vector<Int_t> vec;
   std::vector<Bool_t> vecBool;
   
   ClassWithVector() { }
   ~ClassWithVector() override { }
   
   ClassDefOverride(ClassWithVector, 1);
};

class ClassWithClones : public TObject {
public:
   TClonesArray arr;

   ClassWithClones() : arr("Particle", 5) { }

   ClassDefOverride(ClassWithClones, 1);
};

class Particle : public TObject {
public:
   Particle() { }
   double fPosX,fPosY,fPosZ;

   ClassDefOverride(Particle,1);
};

class EventData : public TObject {
public:
   std::vector<Particle> fParticles;
   int fEventSize;

   void SetSize() {
      fEventSize = sizeof(EventData) + fParticles.size() * sizeof(Particle);
   }
   void Clear(const char* = "") override {
      fParticles.clear();
   }
   void AddParticle(const Particle& p) { fParticles.push_back(p); }

   ClassDefOverride(EventData,1);
};

#endif
