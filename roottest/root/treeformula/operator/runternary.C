#include "TTree.h"


int func1(int i) {
   return i*i;
}

int func2(int i,int j) {
   return i*j;
}

int funcchar(char *x) {
   return x[0];
}

bool ternaryfunc_test(TFormula *form, Int_t val, Int_t expected) {
   Int_t res = (Int_t)form->Eval(val);
   if (res != expected) {
      fprintf(stdout,"ternary test failed, got %d instead of %d\n",res,expected);
      return false;
   }
   return true;
}
   
void ternaryFunction() {
   TFormula *form = new TFormula("tern1","x?2:4");
   ternaryfunc_test(form, 0,4);
   ternaryfunc_test(form,-1,2);
   ternaryfunc_test(form, 1,2);
   
   form = new TFormula("tern2","x?((x+1&&2)?2:3):4");
   ternaryfunc_test(form, 0,4);
   ternaryfunc_test(form,-1,3);
   ternaryfunc_test(form, 1,2);
}

TTree* createtree() {

   TTree *tree = new TTree("T","T");
   float cond, first, second;
   string left,right;
   right = "a";
   left = "A";
   
   tree->Branch("cond", &cond);
   tree->Branch("first",&first);
   tree->Branch("second",&second);
   tree->Branch("right",&right);
   tree->Branch("left",&left);
   for(int j=0;j<5;++j) {
      cond = j;
      first = 100 + j;
      second = -j;
      right[0] = right[0] + 1;
      left[0] = left[0] + 1;
      tree->Fill();
      
   }
   tree->ResetBranchAddresses();
   return tree;
}

TTree* ternarytree() {
   TTree *tree = createtree();
   tree->Scan("cond:first:second:cond<2?first:second:left:right:cond<2?left:right","","col=:::20:::20");
   return tree;
}

int runternary() {
   ternaryFunction();
   ternarytree();
   return 0;
}
