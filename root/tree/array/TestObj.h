/*
        TestObj.h
*/

#ifndef __TESTOBJ_H
#define __TESTOBJ_H

#include <TClass.h>
#include <TClonesArray.h>

#include <stdio.h>

class Particle {
                public:
        int t[2]; //index of tracks
        int nt;

        Particle() { Clear(); }

        void Init(int v)  { nt = 2+v; t[1] = t[0] = 1+v; }
        void Clear() { nt = 2; t[1] = t[0] = 1; }

        int print() {
           fprintf(stderr,"Particle at %p\n",this);
           fprintf(stderr,"   nt  ==%d\n",nt);
           fprintf(stderr,"   t[0]==%d\n",t[0]);
           fprintf(stderr,"   t[1]==%d\n",t[1]);
           return 0; 
        }
}; // Particle


class Reconstruction: public TObject {
                public:
        Particle p[2]; //particles of reconstrction
        int np;

        Reconstruction(int v=0) { Init(v); }

        void Init(int v) { np = 2+v; for (int i = 0; i < 2; ++i) p[i].Init(v); }

        virtual void Clear(const Option_t* opt = "") {
           Init(0);
           TObject::Clear(opt);
        }
        int print() {
           fprintf(stderr,"Reconstruction at %p\n",this);
           fprintf(stderr,"   np==%d\n",np);
           p[0].print();
           p[1].print();
           return 0;
        }
        
                private:
//      static const bool gInitted;
//      static bool StaticInit() { Class()->IgnoreTObjectStreamer(); }

        ClassDef(Reconstruction,1);
}; // Reconstruction


class AllReconstructions {
                public:
//      Reconstruction r[90];
        TClonesArray r; //reconstructions

        AllReconstructions(): r("Reconstruction",6) {}

        void Init(int v=0) { for (int i = 0; i < 6; ++i) new(r[i]) Reconstruction(v); }
        void Clear() { for (int i = 0; i < 6; ++i) r[i]->Clear(); }

        int print() {
           fprintf(stderr,"AllReconstructions at %p\n",this);
           for(int i = 0; i < 6; ++i) {
              Reconstruction *obj = (Reconstruction *)r[i];
              obj->print();
           }
           return 0;
        }

}; // AllReconostructions

#endif // __TESTOBJ_H
