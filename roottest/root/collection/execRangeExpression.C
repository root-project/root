
#include <iostream>
#include "TNamed.h"
#include "TClass.h"

#include "TObjArray.h"
#include "TList.h"
#include "THashList.h"
#include "TSortedList.h"
#include "TArrayL.h"
#include "TClonesArray.h"
#include "TExMap.h"
#include "THashTable.h"
#include "TMap.h"


template <typename coll> void testColl()
{
   std::cout << coll::Class()->GetName() << "\n";
   coll arr;
   arr.begin();
   arr.Add(new TNamed("1","title 1"));
   arr.Add(new TNamed("2","title 2"));
   for(auto i : arr) i->ls("noaddr");
}

void testClones()
{
   typedef TClonesArray coll;
   std::cout << coll::Class()->GetName() << "\n";
   coll arr("TNamed");
   new (arr[0]) TNamed("1","title 1");
   new (arr[1]) TNamed("2","title 2");
   for(auto i : arr) i->ls("noaddr");
}

template <typename coll> void testMap()
{
   std::cout << coll::Class()->GetName() << "\n";
   coll arr;
   TObject *val = new TNamed("1","title 1");
   arr.Add(val,val);
   val = new TNamed("2","title 2");
   arr.Add(val,val);

   for(auto i : arr) i->ls("noaddr");
}

void execRangeExpression() {

   cout << "Testing range expression for iterating through ROOT collections\n";
   testColl<TObjArray>();
   testColl<TList>();
   testColl<TSortedList>();
   testColl<THashList>();
   testClones();
   //testMap<TExMap>();
   testMap<TMap>();
   

   std::cout << "end\n";
}
