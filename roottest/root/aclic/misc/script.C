#include "script.h"
//#include <new>
#include "nest/nested.h"
#include "nested_dir1.h"
#include "nested_dir2.h"
#include "tempHeader.h" // in temp/subdir3/tempHeader.h

int function() {
   defined_in_dir1_nested();
   defined_in_dir2_nested();
   defined_in_temp_subdir3();
   return defined_in_nested();
}
