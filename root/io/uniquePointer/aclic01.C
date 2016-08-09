#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <memory>
#include "TClass.h"
#include "TH1F.h"
#include "TFile.h"

class aclic01Class01{
private:
   vector<unique_ptr<int>> m0;
//    set<vector<unique_ptr<int>>> m1;
//    vector<vector<vector<vector<list<unique_ptr<TH1F>>>>>> m0;
//    vector<map<int,unique_ptr<TH1F>>> m1;
//    vector<map<int,TH1F*>> m2;
//    vector<unique_ptr<int>> ints;
//    vector<unique_ptr<double>> doubles;
//    vector<list<double>*> listdoubles2;
//    vector<unique_ptr<list<double>>> listdoubles;
};

class Dummy {
   vector<vector<vector<vector<list<TH1F*>>>>> m;
};

int checkDict(const char* name){
   int ret = 0;
   auto c = TClass::GetClass(name);
   if (c) {
      auto hasDict = c->HasDictionary();
      std::cout << "Class " << c->GetName() << " has "<< (hasDict ? "a":"*no*") << " dictionary\n";
      ret += hasDict ? 0 : 1;
   } else {
      std::cerr << "Class " << c->GetName() << " not found!\n";
      ret += 1;
   }
   return ret;
}


int aclic01(){

   TFile f("aclic01.root","RECREATE");
   aclic01Class01 a;
   f.WriteObject(&a,"aclic01Class01");
   f.Close();

   auto classNames = {"aclic01Class01",
                      "vector<int*>",
                      "vector<vector<vector<vector<list<TH1F*>>>>>"};
   int ret = 0;
   for (auto& className : classNames)
      ret += checkDict(className);
   return ret;

};
