#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class Electron + ;
#pragma link C++ class std::set < Electron> + ;
#pragma link C++ class std::set < std::set < Electron>> + ;
#pragma link C++ class std::set < std::vector < Electron>> + ;

#pragma link C++ class Jet + ;

#endif
