#include "TVector3.h"
#include <vector>

class SRVertex {
public:
   float time;
   TVector3 vtx;
};

class SRElastic : public SRVertex
{
public:
   long stuff;
};

class SRVertexBranch 
{
public:
   std::vector<SRElastic> elastic;
};

class Holder {
public:
   SRVertexBranch vtx;
};

#include "TTree.h"

void execDupNames()
{
   TTree *t = new TTree("T","T");
   Holder h;
   t->Branch("rec.",&h);
}


