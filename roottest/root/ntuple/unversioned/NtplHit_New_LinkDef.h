#ifdef __ROOTCLING__

#pragma link C++ class Hit+;

#pragma read sourceClass="Hit" targetClass="Hit" checksum="[2391364433]" \
   source="int fA; int fB; double fX; double fY" target="fA_r,fB_r,fX_r,fY_r" \
   code="{ fA_r = onfile.fA; fB_r = onfile.fB; fX_r = onfile.fX; fY_r = onfile.fY; }"

#endif
