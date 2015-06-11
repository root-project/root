#include "script.h"
//#include <new>
#include "nest/nested.h"
#include "nested_dir1.h"
#include "nested_dir2.h"

int function() {
   defined_in_dir1_nested();
   defined_in_dir2_nested();
   return defined_in_nested();
}
