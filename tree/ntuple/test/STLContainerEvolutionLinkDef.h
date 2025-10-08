#ifdef __CLING__

#pragma link C++ class std::set<int>+;
#pragma link C++ class std::set<short int>+;
#pragma link C++ class std::set<std::pair<int, int>>+;
#pragma link C++ class std::set<std::pair<short int, short int>>+;

#pragma link C++ class std::unordered_set<int>+;
#pragma link C++ class std::unordered_set<short int>+;
#pragma link C++ class std::unordered_set<std::pair<int, int>>+;
#pragma link C++ class std::unordered_set<std::pair<short int, short int>>+;

#pragma link C++ class std::multiset<int>+;
#pragma link C++ class std::multiset<short int>+;
#pragma link C++ class std::multiset<std::pair<int, int>>+;
#pragma link C++ class std::multiset<std::pair<short int, short int>>+;

#pragma link C++ class std::unordered_multiset<int>+;
#pragma link C++ class std::unordered_multiset<short int>+;
#pragma link C++ class std::unordered_multiset<std::pair<int, int>>+;
#pragma link C++ class std::unordered_multiset<std::pair<short int, short int>>+;

#pragma link C++ class std::map<int, int>+;
#pragma link C++ class std::map<short int, short int>+;

#pragma link C++ class std::unordered_map<int, int>+;
#pragma link C++ class std::unordered_map<short int, short int>+;

#pragma link C++ class std::multimap<int, int>+;
#pragma link C++ class std::multimap<short int, short int>+;

#pragma link C++ class std::unordered_multimap<int, int>+;
#pragma link C++ class std::unordered_multimap<short int, short int>+;

#pragma link C++ class CollectionProxy<int>+;
#pragma link C++ class CollectionProxy<short int>+;
#pragma link C++ class CollectionProxy<std::pair<int, int>>+;
#pragma link C++ class CollectionProxy<std::pair<short int, short int>>+;

#endif
