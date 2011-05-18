// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

#include "Reflex/Reflex.h"
#include <iostream>
#include <fstream>

#ifdef _WIN32
  # include <windows.h>
#elif defined(__linux) || defined(__APPLE__)
  # include <dlfcn.h>
#endif

using namespace ROOT::Reflex;
using namespace std;

enum Visibility { Public, Protected, Private };

void
generate_visibility(ostream& out,
                    const Member& m,
                    const string& indent,
                    Visibility& v) {
   if (m.IsPublic() && v != Public) {
      out << indent << "public:" << endl;  v = Public;
   } else if (m.IsProtected() && v != Protected) {
      out << indent << "protected:" << endl;  v = Protected;
   } else if (m.IsPrivate() && v != Private) {
      out << indent << "private:" << endl;  v = Private;
   }
}


void
generate_comment(ostream& out,
                 const Member& m) {
   if (m.Properties().HasProperty("comment")) {
      out << "  //" << m.Properties().PropertyAsString("comment");
   }
}


void
generate_class(ostream& out,
               const Type& cl,
               const string& indent = "") {
   out << indent << "class " << cl.Name();

   //...Bases
   if (cl.BaseSize() != 0) {
      out << " : ";

      for (size_t b = 0; b < cl.BaseSize(); b++) {
         Base ba = cl.BaseAt(b);

         if (ba.IsVirtual()) {
            out << "virtual ";
         }

         if (ba.IsPublic()) {
            out << "public ";
         }

         if (ba.IsPrivate()) {
            out << "private ";
         }
         out << ba.ToType().Name(SCOPED);

         if (b != cl.BaseSize() - 1) {
            out << ", ";
         }
      }
   }
   out << " {" << endl;
   Visibility curr_vis = Private;

   //...data members
   for (size_t d = 0; d < cl.DataMemberSize(); d++) {
      Member dm = cl.DataMemberAt(d);

      if (dm.IsArtificial()) {
         continue;
      }
      generate_visibility(out, dm, indent, curr_vis);
      out << indent + "  " << dm.TypeOf().Name(SCOPED | QUALIFIED) << " " << dm.Name() << ";";
      generate_comment(out, dm);
      out << endl;
   }

   //...methods
   for (size_t f = 0; f < cl.FunctionMemberSize(); f++) {
      Member fm = cl.FunctionMemberAt(f);

      if (fm.IsArtificial()) {
         continue;
      }
      generate_visibility(out, fm, indent, curr_vis);
      Type ft = fm.TypeOf();
      out << indent + "  ";

      if (!fm.IsConstructor() && !fm.IsDestructor()) {
         out << ft.ReturnType().Name(SCOPED) << " ";
      }

      if (fm.IsOperator()) {
         out << "operator ";
      }
      out << fm.Name() << " (";

      if (ft.FunctionParameterSize() == 0) {
         out << "void";
      } else {
         for (size_t p = 0; p < ft.FunctionParameterSize(); p++) {
            out << ft.FunctionParameterAt(p).Name(SCOPED | QUALIFIED);

            if (fm.FunctionParameterNameAt(p) != "") {
               out << " " << fm.FunctionParameterNameAt(p);
            }

            if (fm.FunctionParameterDefaultAt(p) != "") {
               out << " = " << fm.FunctionParameterDefaultAt(p);
            }

            if (p != ft.FunctionParameterSize() - 1) {
               out << ", ";
            }
         }
      }
      out << ");";
      generate_comment(out, fm);
      out << endl;
   }

   out << indent << "};" << endl;
} // generate_class


template <typename T>
struct NameSorter {
   bool
   operator ()(const T& one,
               const T& two) const {
      return one.Name(0) < two.Name(0);
   }


};

void
generate_namespace(ostream& out,
                   const Scope& ns,
                   const string& indent = "") {
   if (!ns.IsTopScope()) {
      out << indent << "namespace " << ns.Name() << " {" << endl;
   }


   // Sub-Namespaces
   std::vector<Scope> subscopes(ns.SubScope_Begin(), ns.SubScope_End());
   std::sort(subscopes.begin(), subscopes.end(), NameSorter<Scope>());

   for (size_t i = 0; i < subscopes.size(); i++) {
      Scope sc = subscopes[i];

      if (sc.IsNamespace()) {
         generate_namespace(out, sc, indent + "  ");
      }
      //one is enough, and we already generate classes as types below:
      // if ( sc.IsClass() ) generate_class(out, Type::ByName(sc.Name(SCOPED)), indent + "  ");
   }
   // Types----
   std::vector<Type> subtypes(ns.SubType_Begin(), ns.SubType_End());
   std::sort(subtypes.begin(), subtypes.end(), NameSorter<Type>());

   for (size_t t = 0; t < subtypes.size(); t++) {
      Type ty = subtypes[t];

      if (ty.IsClass()) {
         generate_class(out, ty, indent + "  ");
      }
   }

   if (!ns.IsTopScope()) {
      out << indent << "}" << endl;
   }
} // generate_namespace


int
main(int /*argc*/,
     char* argv[]) {
#ifdef _WIN32
   HMODULE libInstance = LoadLibrary("libtest_Class2DictRflx.dll");

   if (!libInstance) {
      std::cerr << "Could not load dictionary. " << std::endl << "Reason: " << GetLastError() << std::endl;
   }
#else
   void* libInstance = dlopen("libtest_Class2DictRflx.so", RTLD_LAZY);

   if (!libInstance) {
      std::cerr << "Could not load dictionary. " << std::endl << "Reason: " << dlerror() << std::endl;
   }
#endif

   std::string outfilename(argv[0]);
   outfilename += ".testout";
   ofstream outfile(outfilename.c_str());

   generate_namespace(outfile, Scope::GlobalScope());

   int ret = 0;
#if defined(_WIN32)
   ret = FreeLibrary(libInstance);

   if (ret == 0) {
      std::cerr << "Unload of dictionary library failed." << std::endl << "Reason: " << GetLastError() << std::endl;
   }
#else
   ret = dlclose(libInstance);

   if (ret == -1) {
      std::cerr << "Unload of dictionary library failed." << std::endl << "Reason: " << dlerror() << std::endl;
   }
#endif

   return 0;
} // main
