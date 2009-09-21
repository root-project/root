#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ options=version(3) class ClassA+;
#pragma link C++ options=version(3) class ClassB+;
#pragma link C++ options=version(3) class ClassC+;
#pragma link C++ options=version(3) class ClassD+;
#pragma link C++ options=version(3) class ClassAIns+;
#pragma link C++ options=version(2) class ClassABase+;
#pragma link C++ class std::vector<ClassA>+;
#pragma link C++ class std::vector<ClassB>+;
#pragma link C++ class std::vector<ClassC>+;
#pragma link C++ class std::vector<ClassD>+;
#pragma link C++ class std::vector<std::pair<int,float> >+;
#pragma link C++ class std::vector<std::pair<int,double> >+;
#pragma link C++ class std::vector<ClassA*>+;
#pragma link C++ class std::vector<ClassB*>+;
#pragma link C++ class std::vector<ClassC*>+;
#pragma link C++ class std::vector<ClassD*>+;

#pragma read sourceClass="ClassA" version="[2]" targetClass="ClassA" source="int m_unit" target="m_unit" \
  code="{ m_unit = 10*onfile.m_unit; }"

#pragma read sourceClass="ClassA" version="[2]" targetClass="ClassA" source="int m_unit" target="m_md_set" \
  code="{ newObj->m_d.m_unit = 10*onfile.m_unit; m_md_set = true; newObj->Print(); }"

//#pragma read sourceClass="ClassA" version="[2]" targetClass="ClassA" source="int m_unit; ClassAIns m_d" target="m_d" \
//  code="{ m_d = onfile.m_d; m_d.m_punit = 10*onfile.m_unit; }"


// #pragma link C++ class std::pair<int,double>+;

//#pragma read sourceClass="ClassA" version="[2]" targetClass="ClassA2" source="ClassABase" target="ClassABase2"
//#pragma read sourceClass="ClassAIns" version="[2]" targetClass="ClassAIns2"
//#pragma read sourceClass="ClassABase" version="[2]" targetClass="ClassABase2"
//#pragma read sourceClass="ClassC" version="[2]" targetClass="ClassC" source="ClassABase" target="ClassABase2"
//#pragma read sourceClass="ClassB" version="[2]" targetClass="ClassB" source="ClassA" target="ClassA2"
//#pragma read sourceClass="pair<int,double>" version="[*]" targetClass="pair<int,float>"

#endif
