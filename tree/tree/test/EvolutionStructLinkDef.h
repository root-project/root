#ifdef __ROOTCLING__

#pragma link C++ class EvolutionStruct_V2+;
#pragma link C++ class EvolutionStruct_V3+;
#pragma link C++ class std::vector<EvolutionStruct_V2>+;
#pragma link C++ class std::vector<EvolutionStruct_V3>+;

#pragma read sourceClass = "EvolutionStruct_V2" source = "float fOldMember" version = "[2]" targetClass = \
   "EvolutionStruct_V3" target = "fNewMember" code = "{ fNewMember = onfile.fOldMember; }"

#endif
