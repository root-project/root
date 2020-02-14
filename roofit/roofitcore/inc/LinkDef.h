#include "roofit/roofitcore/inc/LinkDef1.h"
#include "roofit/roofitcore/inc/LinkDef2.h"
#include "roofit/roofitcore/inc/LinkDef3.h"
#include "roofit/roofitcore/inc/LinkDef4.h"

#pragma link C++ class RooSTLRefCountList<RooAbsArg>+;
#pragma link C++ class RooStringVar+ ;
#pragma read sourceClass="RooAbsString" targetClass="RooStringVar" version="[1]" source="Int_t _len; char *_value" target="_string" code="{_string.assign(onfile._value, onfile._len);}"
