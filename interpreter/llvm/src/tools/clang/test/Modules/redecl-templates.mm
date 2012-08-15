// template <typename T> int foo(T);
// int usesfootoo() { return foo((float)1); }
// template <typename T> int foo(T) { return 0; }

// class Redecl;
// class RedeclImp { Redecl* p; };
// class Redecl { public: int M; };

@__experimental_modules_import redecl_templates_left;
@__experimental_modules_import redecl_templates_right;

int call() {
   // L::AB b;

   // Redecl r;

  //usesfwddeclThenDefinedInLeftRight();
  //usesfwddeclThenDefinedInLeftRightToo();
  definedInLeft((double)0.);
//   L::definedInLeftSpecializedInRight((float)0);
//   L::definedInLeftSpecializedInRight((char)0);
   return 0;
}
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c++ -fmodule-cache-path %t -emit-module -fmodule-name=redecl_templates_left %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c++ -fmodule-cache-path %t -emit-module -fmodule-name=redecl_templates_right %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -w %s -emit-obj -o %t.obj
// RUN: echo 'int call(); int main(int,char*[]) { return call(); }' | %clang -x objective-c++ -fmodules -fmodule-cache-path %t - -c -o %t_main.obj
// RUN: %clang -lstdc++ %t_main.obj %t.obj
