#include "TObject.h"

class MyPos: public TObject {
public:
   MyPos(): fX(0), fY(0), fZ(0) {}
   MyPos(double x[3]): fX(x[0]), fY(x[1]), fZ(x[2]) {}

   double X() const { return fX; }
   double Y() const { return fY; }
   double Z() const { return fZ; }

private:
   double fX;
   double fY;
   double fZ;
   ClassDef(MyPos,1)
};

class MyParticle: public TObject {
public:
   MyParticle() { fP[0] = 0.; fP[1] = 0.; fP[2] = 0.; fP[3] = 0; }
   MyParticle(double p[4], const MyPos& pos): fPos(pos) {
      fP[0] = p[0];
      fP[1] = p[1];
      fP[2] = p[2];
      fP[3] = p[3];
   }
   const MyPos& Pos() const { return fPos; }
   const double* P() const { return fP; }

private:
   double fP[4];
   MyPos fPos;
   ClassDef(MyParticle,1)
};

class ParticleHolder: public TObject {
public:
   ParticleHolder(): fNParticles(0), fParticles(0) {}
   ~ParticleHolder() { delete [] fParticles; }
   void Clear(const Option_t* = "") {
      ClearAllP(fParticlesFixed, fNParticles);
      fNParticles = 0;
   }
   void SetN(int n) {
      fNParticles = n;
      delete [] fParticles;
      fParticles = new MyParticle[n];
   }
   void Set(int i, const MyParticle& p) {
      fParticlesFixed[i] = p;
      fParticles[i] = p;
   }
private:
   void ClearAllP(MyParticle* p, Int_t n) { while (n > 0) {p[--n] = MyParticle(); }}
   Int_t       fNParticles;
   MyParticle  fParticlesFixed[100];
   MyParticle *fParticles; //[fNParticles]
   ClassDef(ParticleHolder, 1)
};
