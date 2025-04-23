#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ class MyClass+;

#pragma read sourceClass="MyClass" version="[1-]" source="TObjArray fArray;" \
  targetClass="MyClass" target="fArray" \
  code="{ fArray = onfile.fArray; fArray.Compress(); }"

#endif
