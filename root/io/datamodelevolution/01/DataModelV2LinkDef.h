#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ options=version(2) class ClassA2+;
#pragma link C++ options=version(3) class ClassB+;
#pragma link C++ options=version(3) class ClassC+;
#pragma link C++ options=version(3) class ClassD+;
#pragma link C++ options=version(2) class ClassAIns2+;
#pragma link C++ options=version(2) class ClassABase2+;
#pragma link C++ class std::vector<ClassA2>+;
#pragma link C++ class std::vector<std::pair<int,float> >+;
#pragma link C++ class std::vector<ClassA2*>+;
//#pragma link C++ class std::pair<int,float>+;

#pragma read sourceClass="ClassA" version="[2]" targetClass="ClassA2" source="ClassABase" target="ClassABase2"
#pragma read sourceClass="ClassAIns" version="[2]" targetClass="ClassAIns2"
#pragma read sourceClass="ClassABase" version="[2]" targetClass="ClassABase2"
#pragma read sourceClass="ClassC" version="[2]" targetClass="ClassC" source="ClassABase" target="ClassABase2"
#pragma read sourceClass="ClassB" version="[2]" targetClass="ClassB" source="ClassA" target="ClassA2"
//#pragma read sourceClass="pair<int,double>" version="[*]" targetClass="pair<int,float>"

#endif
