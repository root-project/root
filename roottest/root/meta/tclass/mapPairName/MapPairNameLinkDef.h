#pragma link C++ class MapPairKey+;
#pragma link C++ class MapPairHolder+;
#pragma link C++ class std::map<MapPairKey,std::string,MapPairLess>+;
#pragma link C++ class std::multimap<MapPairKey,std::string,MapPairLess>+;
#pragma link C++ class std::unordered_map<MapPairKey,std::vector<std::string>,MapPairHash,MapPairEqual>+;
#pragma link C++ class std::unordered_multimap<MapPairKey,std::vector<std::string>,MapPairHash,MapPairEqual>+;
