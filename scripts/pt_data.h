#include "TObject.h"
#include "TString.h"

class pt_data: public TObject {
 public:
 pt_data(): testname(0), testtime(0) {}
  int outlier;
  unsigned int testnumber;
  TString testname;
  TString testtime;  
  double heapalloc;
  double heapleak;
  double heappeak;
  double cputime;
  double meanTime;
  double varTime;
  double squareTime;
  double meanHeappeak;
  double varHeappeak;
  double squareHeappeak;
  double meanHeapalloc;
  double varHeapalloc;
  double squareHeapalloc;
  double meanHeapleak;
  double varHeapleak;
  double squareHeapleak;
  double z1,z2,z3,z4;
  ClassDef(pt_data,1)
}; 
    
