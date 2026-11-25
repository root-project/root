#ifdef __ROOTCLING__

#pragma link C++ class EvolutionStruct_V2+;
#pragma link C++ class EvolutionStruct_V3+;
#pragma link C++ class std::vector<EvolutionStruct_V2>+;
#pragma link C++ class std::vector<EvolutionStruct_V3>+;

#pragma read sourceClass = "EvolutionStruct_V2" source = "float fOldMember" version = "[2]" targetClass = \
   "EvolutionStruct_V3" target = "fNewMember" code = "{ fNewMember = onfile.fOldMember; }"

#pragma link C++ class EvolutionClonesItem_V2+;
#pragma link C++ class EvolutionClonesItem_V3+;
#pragma link C++ class EvolutionClonesHolder_V2+;
#pragma link C++ class EvolutionClonesHolder_V3+;

#pragma read sourceClass = "EvolutionClonesItem_V2" source = "float fPy; float fPz;" version = "[2]" targetClass = \
   "EvolutionClonesItem_V3" target = "fCombined" code = "{ fCombined = onfile.fPy * 100 + onfile.fPz; }"

#endif
