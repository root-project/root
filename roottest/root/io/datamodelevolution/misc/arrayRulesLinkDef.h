#ifdef __MAKECINT__
#pragma link C++ class MyClass+;
#pragma read sourceClass="MyClass" targetClass="MyClass" source="" target="fCaches" code="{ newObj->fCaches[0] = 1; }"
#pragma read sourceClass="MyClass" targetClass="MyClass" source="float fOldCaches[3]" target="fCaches"  code="{ newObj->fCaches[0] = onfile.fOldCaches[0]; }"
#endif

