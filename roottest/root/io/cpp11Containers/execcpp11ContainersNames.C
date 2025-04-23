
std::string getCppForCling(const char* name){
   static int nsCounter=0;
   std::string instantiateClass = "namespace dummy_ns_";
   instantiateClass += std::to_string(nsCounter);
   instantiateClass += " { ";
   instantiateClass += name;
   instantiateClass += " a;}\n";
   nsCounter++;
   return instantiateClass;
}

int checkAndPrint(const char* name, const char* normName){

   // Check if it is sane C++
#ifdef __APPLE__
   if (strstr(name,"unordered_map")==nullptr && strstr(name,"unordered_multimap")==nullptr)
#endif
   gInterpreter->ProcessLine(getCppForCling(name).c_str());

   auto cl = TClass::GetClass(name);
   if (!cl){
      std::cerr << "Error: Could not find class " << name << " in the typesystem.\n";
      return 100;
   }

   auto tclassName = cl->GetName();
   std::cout << name << " -> " << tclassName << std::endl;

   if (0 != strcmp(normName, tclassName)){
      std::cerr << "Error: Expected norm name and constructed norm name differ:\n - " << normName << "\n - " << tclassName << std::endl<< std::endl;
      return 1;
   }

   return 0;
}

int execcpp11ContainersNames(){

   int retcode = 0;

   retcode += checkAndPrint("std::vector<int>","vector<int>");

   // inject aux code: allocators, hashes and so on.
   gInterpreter->ProcessLine("#include \"auxCode.h\"");

/*
Foewardlist:
template<
    class T,
    class Allocator = std::allocator<T>> class forward_list;
*/
   retcode += checkAndPrint("std::forward_list<int>","forward_list<int>");
   retcode += checkAndPrint("std::forward_list<TH1F,allocator<TH1F>>","forward_list<TH1F>");
   retcode += checkAndPrint("std::forward_list<TH1F,std::allocator<TH1F>>","forward_list<TH1F>");
   retcode += checkAndPrint("std::forward_list<TH1F,myAlloc<TH1F>>","forward_list<TH1F,myAlloc<TH1F> >");

/*
Unorderedset:
template < class Key,                        // unordered_set::key_type/value_type
           class Hash = hash<Key>,           // unordered_set::hasher
           class Pred = equal_to<Key>,       // unordered_set::key_equal
           class Alloc = allocator<Key>      // unordered_set::allocator_type
           > class unordered_set;
*/
   retcode += checkAndPrint("std::unordered_set<int>","unordered_set<int>");
   retcode += checkAndPrint("unordered_set<int>","unordered_set<int>");
   retcode += checkAndPrint("unordered_set<int,hash<int>,equal_to<int>,allocator<int>>","unordered_set<int>");
   retcode += checkAndPrint("unordered_set<int,myHash<int>,equal_to<int>,allocator<int>>","unordered_set<int,myHash<int> >");
   retcode += checkAndPrint("unordered_set<int,myHash<int>,myEqual_to<int>,allocator<int>>","unordered_set<int,myHash<int>,myEqual_to<int> >");
   retcode += checkAndPrint("unordered_set<int,myHash<int>,equal_to<int>,myAlloc<int>>","unordered_set<int,myHash<int>,equal_to<int>,myAlloc<int> >");
   retcode += checkAndPrint("vector<unordered_set<int,myHash<int>,equal_to<int>,myAlloc<int>>>","vector<unordered_set<int,myHash<int>,equal_to<int>,myAlloc<int> > >");
   retcode += checkAndPrint("std::unordered_set<TH1F>","unordered_set<TH1F>");
   retcode += checkAndPrint("unordered_set<TH1F>","unordered_set<TH1F>");
   retcode += checkAndPrint("unordered_set<TH1F,hash<TH1F>,equal_to<TH1F>,allocator<TH1F>>","unordered_set<TH1F>");
   retcode += checkAndPrint("unordered_set<TH1F,myHash<TH1F>,equal_to<TH1F>,allocator<TH1F>>","unordered_set<TH1F,myHash<TH1F> >");
   retcode += checkAndPrint("unordered_set<TH1F,myHash<TH1F>,myEqual_to<TH1F>,allocator<TH1F>>","unordered_set<TH1F,myHash<TH1F>,myEqual_to<TH1F> >");
   retcode += checkAndPrint("unordered_set<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F>>","unordered_set<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F> >");
   retcode += checkAndPrint("vector<unordered_set<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F>>>","vector<unordered_set<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F> > >");

/*
Unorderedmultiset:
template < class Key,                        // unordered_set::key_type/value_type
           class Hash = hash<Key>,           // unordered_set::hasher
           class Pred = equal_to<Key>,       // unordered_set::key_equal
           class Alloc = allocator<Key>      // unordered_set::allocator_type
           > class unordered_multiset;
*/
   retcode += checkAndPrint("std::unordered_multiset<int>","unordered_multiset<int>");
   retcode += checkAndPrint("unordered_multiset<int>","unordered_multiset<int>");
   retcode += checkAndPrint("unordered_multiset<int,hash<int>,equal_to<int>,allocator<int>>","unordered_multiset<int>");
   retcode += checkAndPrint("unordered_multiset<int,myHash<int>,equal_to<int>,allocator<int>>","unordered_multiset<int,myHash<int> >");
   retcode += checkAndPrint("unordered_multiset<int,myHash<int>,myEqual_to<int>,allocator<int>>","unordered_multiset<int,myHash<int>,myEqual_to<int> >");
   retcode += checkAndPrint("unordered_multiset<int,myHash<int>,equal_to<int>,myAlloc<int>>","unordered_multiset<int,myHash<int>,equal_to<int>,myAlloc<int> >");
   retcode += checkAndPrint("vector<unordered_multiset<int,myHash<int>,equal_to<int>,myAlloc<int>>>","vector<unordered_multiset<int,myHash<int>,equal_to<int>,myAlloc<int> > >");
   retcode += checkAndPrint("std::unordered_multiset<TH1F>","unordered_multiset<TH1F>");
   retcode += checkAndPrint("unordered_multiset<TH1F>","unordered_multiset<TH1F>");
   retcode += checkAndPrint("unordered_multiset<TH1F,hash<TH1F>,equal_to<TH1F>,allocator<TH1F>>","unordered_multiset<TH1F>");
   retcode += checkAndPrint("unordered_multiset<TH1F,myHash<TH1F>,equal_to<TH1F>,allocator<TH1F>>","unordered_multiset<TH1F,myHash<TH1F> >");
   retcode += checkAndPrint("unordered_multiset<TH1F,myHash<TH1F>,myEqual_to<TH1F>,allocator<TH1F>>","unordered_multiset<TH1F,myHash<TH1F>,myEqual_to<TH1F> >");
   retcode += checkAndPrint("unordered_multiset<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F>>","unordered_multiset<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F> >");
   retcode += checkAndPrint("vector<unordered_multiset<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F>>>","vector<unordered_multiset<TH1F,myHash<TH1F>,equal_to<TH1F>,myAlloc<TH1F> > >");

/*
Unorderedmap:
template < class Key,                                    // unordered_multimap::key_type
           class T,                                      // unordered_multimap::mapped_type
           class Hash = hash<Key>,                       // unordered_multimap::hasher
           class Pred = equal_to<Key>,                   // unordered_multimap::key_equal
           class Alloc = allocator< pair<const Key,T> >  // unordered_multimap::allocator_type
           > class unordered_map;
*/
   retcode += checkAndPrint("std::unordered_map<int,float>","unordered_map<int,float>");
   retcode += checkAndPrint("unordered_map<int,float>","unordered_map<int,float>");
   retcode += checkAndPrint("unordered_map<int,float,hash<int>,equal_to<int>,allocator<pair<const int,float>>>","unordered_map<int,float>");
   retcode += checkAndPrint("unordered_map<int,float,myHash<int>,equal_to<int>,allocator<pair<const int,float>>>","unordered_map<int,float,myHash<int> >");
   retcode += checkAndPrint("unordered_map<int,float,myHash<int>,myEqual_to<int>,allocator<pair<const int,float>>>","unordered_map<int,float,myHash<int>,myEqual_to<int> >");
   retcode += checkAndPrint("unordered_map<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float>>>","unordered_map<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float> > >");
   retcode += checkAndPrint("vector<unordered_map<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float>>>>","vector<unordered_map<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float> > > >");
   retcode += checkAndPrint("std::unordered_map<int,float>","unordered_map<int,float>");
   retcode += checkAndPrint("unordered_map<vector<TH1F>,float>","unordered_map<vector<TH1F>,float>");
   retcode += checkAndPrint("unordered_map<vector<TH1F>,float,hash<vector<TH1F> >,equal_to<vector<TH1F> >,allocator<pair<const vector<TH1F>,float>>>","unordered_map<vector<TH1F>,float>");
   retcode += checkAndPrint("unordered_map<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,allocator<pair<const vector<TH1F>,float>>>","unordered_map<vector<TH1F>,float,myHash<vector<TH1F> > >");
   retcode += checkAndPrint("unordered_map<vector<TH1F>,float,myHash<vector<TH1F> >,myEqual_to<vector<TH1F> >,allocator<pair<const vector<TH1F>,float> >>","unordered_map<vector<TH1F>,float,myHash<vector<TH1F> >,myEqual_to<vector<TH1F> > >");
   retcode += checkAndPrint("unordered_map<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float>>>","unordered_map<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float> > >");
   retcode += checkAndPrint("vector<unordered_map<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float>>>>","vector<unordered_map<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float> > > >");

/*
Unorderedmultimap:
template < class Key,                                    // unordered_multimap::key_type
           class T,                                      // unordered_multimap::mapped_type
           class Hash = hash<Key>,                       // unordered_multimap::hasher
           class Pred = equal_to<Key>,                   // unordered_multimap::key_equal
           class Alloc = allocator< pair<const Key,T> >  // unordered_multimap::allocator_type
           > class unordered_multimap;
*/
   retcode += checkAndPrint("std::unordered_multimap<int,float>","unordered_multimap<int,float>");
   retcode += checkAndPrint("unordered_multimap<int,float>","unordered_multimap<int,float>");
   retcode += checkAndPrint("unordered_multimap<int,float,hash<int>,equal_to<int>,allocator<pair<const int,float>>>","unordered_multimap<int,float>");
   retcode += checkAndPrint("unordered_multimap<int,float,myHash<int>,equal_to<int>,allocator<pair<const int,float>>>","unordered_multimap<int,float,myHash<int> >");
   retcode += checkAndPrint("unordered_multimap<int,float,myHash<int>,myEqual_to<int>,allocator<pair<const int,float>>>","unordered_multimap<int,float,myHash<int>,myEqual_to<int> >");
   retcode += checkAndPrint("unordered_multimap<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float>>>","unordered_multimap<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float> > >");
   retcode += checkAndPrint("vector<unordered_multimap<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float>>>>","vector<unordered_multimap<int,float,myHash<int>,equal_to<int>,myAlloc<pair<const int,float> > > >");
   retcode += checkAndPrint("std::unordered_multimap<int,float>","unordered_multimap<int,float>");
   retcode += checkAndPrint("unordered_multimap<vector<TH1F>,float>","unordered_multimap<vector<TH1F>,float>");
   retcode += checkAndPrint("unordered_multimap<vector<TH1F>,float,hash<vector<TH1F> >,equal_to<vector<TH1F> >,allocator<pair<const vector<TH1F>,float>>>","unordered_multimap<vector<TH1F>,float>");
   retcode += checkAndPrint("unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,allocator<pair<const vector<TH1F>,float>>>","unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> > >");
   retcode += checkAndPrint("unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> >,myEqual_to<vector<TH1F> >,allocator<pair<const vector<TH1F>,float>>>","unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> >,myEqual_to<vector<TH1F> > >");
   retcode += checkAndPrint("unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float>>>","unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float> > >");
   retcode += checkAndPrint("vector<unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float>>>>","vector<unordered_multimap<vector<TH1F>,float,myHash<vector<TH1F> >,equal_to<vector<TH1F> >,myAlloc<pair<const vector<TH1F>,float> > > >");

   return retcode;

}
