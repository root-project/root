#ifdef __ROOTCLING__

#pragma link C++ class NtplEvolv+;

#pragma read sourceClass="NtplEvolv" version="[1-]" source="int fA;" targetClass="NtplEvolv" target="fA" \
   code = "{ fA = onfile.fA + 13; }"

#endif
