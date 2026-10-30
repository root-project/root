// RUN: %rm TemplateAssert.h.pch
// RUN: %cling -x c++-header %S/Inputs/TemplateAssert.h -o TemplateAssert.h.pch
// RUN: cat %s | %cling -I%p -Xclang -include-pch -Xclang TemplateAssert.h.pch 2>&1 | FileCheck %s

#include "Inputs/TemplateAssert.h"

RequiresTrait<int> r;

// CHECK: static assertion failed
// CHECK-SAME: invalid type
r.Scale(1.0);

// CHECK: static assertion failed
// CHECK-SAME: invalid type
r.Scale(1.0);

.q
