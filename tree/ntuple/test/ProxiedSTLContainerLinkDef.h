#ifdef __CLING__

#pragma link C++ class std::set<std::int64_t>+;
#pragma link C++ class std::set<std::string>+;
#pragma link C++ class std::set<float>+;
#pragma link C++ class std::set<std::set<CustomStruct>>+;
#pragma link C++ class std::set<std::set<char>>+;
#pragma link C++ class std::set<std::tuple<int, char, CustomStruct>>+;
#pragma link C++ class std::set<std::array<float, 3>>+;

#pragma link C++ class std::unordered_set<std::int64_t>+;
#pragma link C++ class std::unordered_set<std::string>+;
#pragma link C++ class std::unordered_set<float>+;
#pragma link C++ class std::unordered_set<CustomStruct>+;
#pragma link C++ class std::unordered_set<std::vector<bool>>+;

#pragma link C++ class std::multiset<std::int64_t>+;
#pragma link C++ class std::multiset<float>+;
#pragma link C++ class std::multiset<CustomStruct>+;

#pragma link C++ class std::unordered_multiset<std::int64_t>+;
#pragma link C++ class std::unordered_multiset<float>+;
#pragma link C++ class std::unordered_multiset<CustomStruct>+;

#pragma link C++ class std::map<char, long>+;
#pragma link C++ class std::map<char, std::int64_t>+;
#pragma link C++ class std::map<int, std::string>+;
#pragma link C++ class std::map<int, std::vector<CustomStruct>>+;
#pragma link C++ class std::map<std::string, float>+;
#pragma link C++ class std::map<char, std::map<int, CustomStruct>>+;
#pragma link C++ class std::map<int, std::map<std::int64_t, float>>+;

#pragma link C++ class std::unordered_map<char, long>+;
#pragma link C++ class std::unordered_map<char, std::int64_t>+;
#pragma link C++ class std::unordered_map<char, std::string>+;
#pragma link C++ class std::unordered_map<int, CustomStruct>+;
#pragma link C++ class std::unordered_map<float, std::vector<bool>>+;

#pragma link C++ class std::multimap<char, std::int64_t>+;
#pragma link C++ class std::multimap<std::string, float>+;
#pragma link C++ class std::multimap<int, CustomStruct>+;

#pragma link C++ class std::unordered_multimap<char, std::int64_t>+;
#pragma link C++ class std::unordered_multimap<std::string, float>+;
#pragma link C++ class std::unordered_multimap<int, CustomStruct>+;

#endif
