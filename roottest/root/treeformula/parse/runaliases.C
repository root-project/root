#include "TTree.h"
#include "TVectorF.h"

void part2() {
   fprintf(stdout,"Test part2\n");
   TTree *t=new TTree("test","test");
   TVectorF *v=new TVectorF(5);
   int arr[5][4];
   t->Branch("vec",&v);
   t->Branch("arr",&arr[0][0],"arr[5][4]/I");
   
   Int_t i;
   for (i=0;i<5;++i) {
      (*v)(i)=i;
      arr[i][0] = -1;
      arr[i][1] = -2;
      arr[i][2] = -3;
      arr[i][3] = -4;
   }
   arr[3][1] = 99;

   t->Fill();
   t->SetAlias("a2","vec.fElements[3]>1");
   t->SetAlias("a1","0+vec.fElements[3]");
   t->SetAlias("a","vec.fElements[3]");
   t->SetAlias("b1","0+arr[3]");
   t->SetAlias("b","arr[3]");
   
   
   fflush(stdout);
   t->Scan("vec.fElements[3]");
   fflush(stdout);
   t->Scan("a2");
   fflush(stdout);
   t->Scan("a1");
   fflush(stdout);
   t->Scan("a");
   fflush(stdout);
   
   t->Scan("vec.fElements[3][0]");
   fflush(stdout);
   t->Scan("a2[0]");
   fflush(stdout);
   t->Scan("a1[0]");
   fflush(stdout);
   t->Scan("a[0]");
   fflush(stdout);
   
   t->Scan("arr[3][1]");
   fflush(stdout);
   t->Scan("b1[1]");
   fflush(stdout);
   t->Scan("b[1]");
   fflush(stdout);
   t->Scan("arr.arr","","");
   fflush(stdout);
   t->Scan("arr.misspelt");
}


struct objs {
   objs() : one(1),two(2) {};
   int one;
   int two;
};

void part1(int debug = 0) {
   fprintf(stdout,"Test part1\n");
   TTree *t = new TTree;
   objs o;
   t->Branch("objs",&o,"one/I:two/I");
   t->Fill();
   if (debug) t->Print();
   t->Scan("objs.one");
   fflush(stdout);
   t->SetAlias("access","objs");
   t->Scan("access.one");
   fflush(stdout);
   t->SetAlias("ones","access.one");
   t->Scan("ones");
   fflush(stdout);
   t->SetAlias("add","ones+twos"); 
   t->Scan("add");
   fflush(stdout);
   t->SetAlias("twos","objs.two");
   t->Scan("add");
   fflush(stdout);
   t->SetAlias("twos","crap");
   t->Scan("add");
   fflush(stdout);
   t->SetAlias("crap","add");
   t->Scan("add");
   fflush(stdout);
}

void runaliases(int debug = 0) {
   // Fill out the code of the actual test
   part1(debug);
   part2();
}
