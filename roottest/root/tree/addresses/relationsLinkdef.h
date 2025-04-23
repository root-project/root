#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class DataObject+;
#pragma link C++ class DataTObject+;
#pragma link C++ class IInterface+;
#pragma link C++ class Relation1D<int,float>+;
//#pragma link C++ class std::pair<int,float>+;
#pragma link C++ class std::vector<std::pair<int,float> >;
//the next line work around issue with CINT's typedef mechanism
//#pragma link C++ class std::vector<pair<int,float> >;
#pragma link C++ class IRelation<int,float>+;
#pragma link C++ class Relation<int,float>+;
#pragma link C++ class RelationBase<int,float>+;

#endif
