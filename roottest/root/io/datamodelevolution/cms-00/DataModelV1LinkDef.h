#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ options=version(2) class ClassA+;
#pragma link C++ options=version(2) class ClassB+;
#pragma link C++ options=version(2) class ClassC+;
#pragma link C++ options=version(2) class ClassAIns+;
#pragma link C++ options=version(2) class ClassAIns::Transient+;
#pragma link C++ options=version(2) class ClassABase+;
#pragma link C++ options=version(2) class ClassD+;
#pragma link C++ class std::vector<std::pair<int,double> >+;
#pragma link C++ class std::vector<ClassA>+;
#pragma link C++ class std::vector<ClassA*>+;
#pragma link C++ class std::vector<ClassB>+;
#pragma link C++ class std::vector<ClassB*>+;
#pragma link C++ class std::vector<ClassC>+;
#pragma link C++ class std::vector<ClassC*>+;
#pragma link C++ class std::vector<ClassD>+;
#pragma link C++ class std::vector<ClassD*>+;

#pragma link C++  options=version(2) class LHCb::Track+;
#pragma link C++  options=version(2) class LHCb::RefRight+;
#pragma link C++  options=version(2) class LHCb::RefLeft+;
#pragma link C++  options=version(2) class LHCb::Ref+;
#pragma link C++ class LHCb::Node+;
#pragma link C++ class vector<LHCb::Node*>+;

// #pragma readraw sourceClass="ClassA" version="[2]" targetClass="ClassA" source="int m_unit" target="m_unit" \
//  code="{ newObj->SetMdUnit( 10*onfile.m_unit) ; m_md_set = true; /* newObj->Print(); */ }"

#endif
