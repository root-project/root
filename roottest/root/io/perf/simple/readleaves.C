class MyClass 
{
public:
  int i0;
  int i1;
  int i2;
  int i3;
};

#include "TTree.h"

void read(TTree &tree,int r)
{
  for(int i=0; i<r; ++i) 
  {
     tree.GetEntry(0);
  }
}

void readleaves(int r = 1000)
{
   MyClass obj;
   TTree tree("tree","tree");
   tree.Branch("obj.",&obj);
   tree.Fill();
   read(tree,r);
}
