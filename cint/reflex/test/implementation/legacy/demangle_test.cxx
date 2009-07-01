#include <assert.h>
#include <iostream>

using std::string;
using std::cout;
using std::endl;

#define demangle_assert(expected, actual) \
   if ((expected) != (actual)) { \
      cout << (expected) + " != " + (actual) << endl; \
      assert((expected) == (actual)); \
   }

bool
isalphanum(int i) {
   // Return true if char is alpha or digit.
   return isalpha(i) || isdigit(i);
}


std::string
normalize_if(const char* nam) {
   // Normalize a type name.
   std::string norm_name;
   char prev = 0;

   for (size_t i = 0; nam[i] != 0; i++) {
      char curr = nam[i];

      if (curr == ' ') {
         char next = 0;

         while (nam[i] != 0 && (next = nam[i + 1]) == ' ') {
            ++i;
         }

         if (!isalphanum(prev) || !isalpha(next)) {
            continue; // continue on non-word boundaries
         }
      } else if (curr == '>' && prev == '>' || curr == '(' && prev != ')') {
         norm_name += ' ';
      }
      norm_name += (prev = curr);
   }

   return norm_name;
} // normalize_if


std::string
normalize_switch(const char* nam) {
   // Normalize a type name.
   std::string norm_name;
   char prev = 0;

   for (size_t i = 0; nam[i] != 0; i++) {
      bool sep = false;
      char curr = nam[i];
      char next = 0;

      switch (curr) {
      case ' ':

         while (nam[i] != 0 && (next = nam[i + 1]) == ' ') {
            ++i;
         }

         if (!isalphanum(prev) || !isalpha(next)) {
            continue; // continue on non-word boundaries
         }

      case '>':
         sep = (prev == '>');
         break;

      case '(':
         sep = (prev != ')');
         break;

      } // switch

      if (sep) {
         norm_name += ' ';
      }
      norm_name += (prev = curr);
   }

   return norm_name;
} // normalize_switch


std::string
normalize(const char* name) {
   return normalize_if(name);
}


void
normalize_test() {
   demangle_assert(string(""), normalize(""));
   demangle_assert(string(""), normalize(" "));
   demangle_assert(string(""), normalize("  "));
   demangle_assert(string("x"), normalize("x"));
   demangle_assert(string("x"), normalize(" x"));
   demangle_assert(string("int"), normalize("int"));
   demangle_assert(string("int"), normalize("   int"));
   demangle_assert(string("int"), normalize("int  "));
   demangle_assert(string("int*"), normalize("int *"));
   demangle_assert(string("int**"), normalize("int **"));
   demangle_assert(string("int**"), normalize("int **  "));
   demangle_assert(string("int*&"), normalize("int *&"));
   demangle_assert(string("float (int)"), normalize("float (int)"));
   demangle_assert(string("float* (int)"), normalize("float *(int)"));
   demangle_assert(string("float* (int)"), normalize("float * (int)"));
   demangle_assert(string("float* (int)"), normalize("float*(int)"));
   demangle_assert(string("float** (int)"), normalize("float **(int)"));
   demangle_assert(string("float*& (int)"), normalize("float *&(int)"));
   demangle_assert(string("float*& (int)"), normalize("float * & (int)"));
   demangle_assert(string("float*& (int)"), normalize("float  *  &  (int)"));
   demangle_assert(string("float**& (int)"), normalize("float **&(int)"));
   demangle_assert(string("float (*)(int)"), normalize("float (*)(int)"));
   demangle_assert(string("float (*)(int)"), normalize("float(*)(int)"));
   demangle_assert(string("float (*)(int)"), normalize("  float  (  *  )  (  int  )  "));
   demangle_assert(string("float& (*)(int)"), normalize("float& (*)(int)"));
   demangle_assert(string("float**& (*)(int)"), normalize("float**& (*)(int)"));
   demangle_assert(string("a::b"), normalize("a::b"));
   demangle_assert(string("std::vector<bar,std::allocator<bar> >"), normalize("std::vector<bar,std::allocator<bar> >"));
   demangle_assert(string("std::vector<bar,std::allocator<bar> >"), normalize("std::vector<bar, std::allocator<bar>>"));
   demangle_assert(string("std::vector<bar,std::vector<bar,std::allocator<bar> > >"), normalize("std::vector<bar, std::vector<bar, std::allocator<bar>>>"));
   demangle_assert(string("float ()(int)"), normalize("float ()(int)"));
   demangle_assert(string("float ()(int)"), normalize("float ( )( int)"));
   demangle_assert(string("float ()(int)"), normalize("float (  )( int)"));
   demangle_assert(string("float* ()(int)"), normalize("float* ()(int)"));
   demangle_assert(string("float* ()(int)"), normalize("float * () (int)"));
   demangle_assert(string("float* ()(int)"), normalize("float* ( ) (int)"));
   demangle_assert(string("float* ()(int)"), normalize("float * (   )   (   int   )"));
   demangle_assert(string("unsigned int"), normalize("unsigned int"));
   demangle_assert(string("unsigned int"), normalize("unsigned    int"));
   demangle_assert(string("unsigned int"), normalize("   unsigned    int  "));
   demangle_assert(string("const char*"), normalize("   const  char   *"));
   demangle_assert(string("volatile const char*"), normalize(" volatile   const  char   *"));
} // normalize_test


int
main() {
   normalize_test();
   return 0;
}
