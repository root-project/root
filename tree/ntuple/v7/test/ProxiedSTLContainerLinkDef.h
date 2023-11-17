#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class std::set<std::int64_t>+;
#pragma link C++ class std::set<std::string>+;
#pragma link C++ class std::set<float>+;
#pragma link C++ class std::set<std::set<CustomStruct>>+;
#pragma link C++ class std::set<std::set<char>>+;
#pragma link C++ class std::set<std::pair<int, CustomStruct>>+;

#pragma link C++ class std::unordered_set<std::int64_t>+;
#pragma link C++ class std::unordered_set<std::string>+;
#pragma link C++ class std::unordered_set<float>+;
#pragma link C++ class std::unordered_set<CustomStruct>+;
#pragma link C++ class std::unordered_set<std::vector<bool>>+;

#endif
