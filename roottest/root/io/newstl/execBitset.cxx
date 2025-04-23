#include <iostream>

#include "TClass.h"
#include "TVirtualCollectionProxy.h"
#include <bitset>

#ifdef __ROOTCLING__
#pragma link C++ class std::bitset<64>+;
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <memory>

template <typename Coll> void runContentTest(TVirtualCollectionProxy *proxy, Coll &bs)
{
   if (!proxy){
      std::cerr << "proxy error\n";
      return;
   }

   if (proxy->GetProperties() & TVirtualCollectionProxy::kIsEmulated) {
      std::cerr << "proxy is emulated (this is an error)\n";
      return;
   }


   // assert some basic properties on proxy
   // assert(proxy->GetCollectionType() == ROOT::kSTLbitset);
   assert(proxy->GetIncrement() == 1);

   TVirtualCollectionProxy::TPushPop helper(proxy,(void*)&bs);
   if(proxy->Size() != bs.size()){
      std::cerr << "size error; got " << proxy->Size() << "\t expected " << bs.size() << "\n";
   }

   Coll bs2(bs);

   bool readout;
   std::cout << "Using 2 push and 2 objects\n";
   for(unsigned i=0;i<proxy->Size();++i) {
      TVirtualCollectionProxy::TPushPop helper2(proxy,(void*)&bs2);
      readout = *((bool*)(proxy->At(i)));
      std::cout << "# " << i << " " << readout << " vs " <<  bs[i] << "\n";
      assert( readout == (int)bs[i] );
      //std::cerr << TDataType::GetDataType(proxy->GetType())->AsString(proxy->At(i)) << "\n";
   }

   std::cout << "Using the same object (and hence same CollectionProxy Env).\n";
   for(unsigned i=0;i<proxy->Size();++i) {
      TVirtualCollectionProxy::TPushPop helper2(proxy,(void*)&bs);
      readout = *((bool*)(proxy->At(i)));
      std::cout << "# " << i << " " << readout << " vs " <<  bs[i] << "\n";
      assert( readout == (int)bs[i] );
      //std::cerr << TDataType::GetDataType(proxy->GetType())->AsString(proxy->At(i)) << "\n";
   }

   std::cout << "Relying on the single push.\n";
   for(unsigned i=0;i<proxy->Size();++i) {
      readout = *((bool*)(proxy->At(i)));
      std::cout << "# " << i << " " << readout << " vs " <<  bs[i] << "\n";
      assert( readout == (int)bs[i] );
      //std::cerr << TDataType::GetDataType(proxy->GetType())->AsString(proxy->At(i)) << "\n";
   }
   TDataType::GetDataType(proxy->GetType())->Print();
   std::cout << "Using GetDataType AsString.\n";
   for(unsigned i=0;i<proxy->Size();++i) {
      readout = *((bool*)(proxy->At(i)));
      std::cout << "# " << i << " " << readout << " vs " <<  bs[i] << "\n";
      assert( readout == (int)bs[i] );
      std::cerr << TDataType::GetDataType(proxy->GetType())->AsString(proxy->At(i)) << "\n";
   }

}

template <typename Coll> void runtest(Coll &bs)
{
   // now get the TCLASS and access bitset via the proxys
   TClass *cl  = TClass::GetClass(typeid(Coll));
   if(!cl) {
      std::cerr << "TClass error\n";
      return;
   }
   // cl->Dump();

   std::unique_ptr<TVirtualCollectionProxy> proxy(cl->GetCollectionProxy()->Generate());

   std::cout << "Run with a copy of the collection proxy\n";
   runContentTest(proxy.get(), bs);

   std::cout << "Run with the original collection proxy\n";
   runContentTest(cl->GetCollectionProxy(), bs);

}

void testbitset() {

   std::bitset<64> bs;
   for(int i=0;i<64;++i)
      bs[i]=i%2;

   runtest(bs);
}

void testvectorbool() {

   std::vector<bool> bs;
   for(int i=0;i<64;++i)
      bs.push_back(i%2);
   
   runtest(bs);
}

class Something {
public:
   std::bitset<5> fBits;

   void Print() {
     for(unsigned int i = 0; i < fBits.size(); ++i) {
       cout << "i: " << i << " val: ";
       if (fBits.test(i)) cout << "true";
       else cout << "false";
       cout << '\n';
     }
   }
   void Fill(int seed) {
     for(unsigned int i = 0; i < fBits.size(); ++i) {
       fBits.set(i,(i+seed)%2);
     }
   }
};

#include "TFile.h"

void trivialFile() {
   TFile *f = TFile::Open("bitset.root","RECREATE");
   Something s;
   s.Fill(0);
   s.Print();
   cout << "About to write\n";
   f->WriteObject(&s,"bits");
   Something *ptr;
   cout << "About to read\n";
   f->GetObject("bits",ptr);
   cout << "Finished reading\n";
   if(!ptr) cout << "Can't find the bitset object\n";
   else ptr->Print();
}

void execBitset() {
   trivialFile();
   testbitset();
   testvectorbool();
}
