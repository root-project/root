ROOT_COMPILE_MACRO(equal.C)

ROOT_ADD_ROOTTEST(cling-operator-EqualTest MACRO runEqualTest.C OUTREF EqualTest.ref-cling OUTCNV EqualTest_convert.sh COMPILEMACROS runEqualTest.C )
ROOT_ADD_ROOTTEST(cling-operator-ParanTest MACRO t01.C OUTREF t01.ref)

if(DEFINED X86_64)
    ROOT_ADD_ROOTTEST(cling-operator-ConversionOp MACRO runConversionOp.C OUTREF ConversionOp.ref64 OUTCNV ConversionOp_convert.sh) 
elseif(DEFINED X86)
    ROOT_ADD_ROOTTEST(cling-operator-ConversionOp MACRO runConversionOp.C OUTREF ConversionOp.ref32 OUTCNV ConversionOp_convert.sh)  
endif()
