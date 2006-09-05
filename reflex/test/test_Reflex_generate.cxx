// @(#)root/reflex:$Name:  $:$Id: test_Reflex_generate.cxx,v 1.5 2006/08/11 06:32:00 roiser Exp $
// Author: Stefan Roiser 2004

#include "Reflex/Reflex.h"
#include <iostream>

#ifdef _WIN32
  #include<windows.h>
#elif defined(__linux) || defined (__APPLE__)
  #include<dlfcn.h>
#endif

using namespace ROOT::Reflex;
using namespace std;

ostream& out = cout;
enum Visibility { Public, Protected, Private }; 

void generate_visibility( const Member& m, const string& indent, Visibility& v ) {
  if ( m.IsPublic() && v != Public ) {
    out << indent << "public:" << endl;  v = Public;
  } else if ( m.IsProtected() && v != Protected ) {
    out << indent << "protected:" << endl;  v = Protected;
  } else if ( m.IsPrivate() && v != Private ) {
    out << indent << "private:" << endl;  v = Private;
  }
}
void generate_comment( const Member& m ) {
  if ( m.Properties().HasKey("comment") ) {
    cout << "  //" << m.Properties().PropertyAsString("comment");
  }
}

void generate_class( const Type& cl, const string& indent = "" ) {
  out << indent << "class " << cl.Name();
  //...Bases
  if (cl.BaseSize() != 0 ) {
    out << " : " ;
    for ( size_t b = 0; b < cl.BaseSize(); b++ ) {
      Base ba = cl.BaseAt(b);
      if ( ba.IsVirtual() ) out << "virtual ";
      if ( ba.IsPublic() ) out << "public ";
      if ( ba.IsPrivate() ) out << "private ";
      out << ba.ToType().Name(SCOPED);
      if ( b != cl.BaseSize()-1) out << ", ";
    }
  }
  out << " {" << endl;
  Visibility curr_vis = Private;
  //...data members
  for ( size_t d = 0; d < cl.DataMemberSize(); d++ ) {
    Member dm = cl.DataMemberAt(d);
    if ( dm.IsArtificial() ) continue;
    generate_visibility( dm, indent, curr_vis);
    out << indent + "  " << dm.TypeOf().Name(SCOPED|QUALIFIED) << " " << dm.Name() <<";" ;
    generate_comment( dm );
    out << endl;
  }
  //...methods
  for ( size_t f = 0; f < cl.FunctionMemberSize(); f++ ) {
    Member fm = cl.FunctionMemberAt(f);
    if ( fm.IsArtificial() ) continue;
    generate_visibility( fm, indent, curr_vis);
    Type ft = fm.TypeOf();
    out << indent + "  ";
    if ( ! fm.IsConstructor() && !fm.IsDestructor() ) out << ft.ReturnType().Name(SCOPED) << " ";
    if (  fm.IsOperator() ) out << "operator ";
    out << fm.Name() << " (";
    if ( ft.FunctionParameterSize() == 0 ) {
      out << "void";
    } else {
      for ( size_t p = 0 ; p < ft.FunctionParameterSize(); p++ ) {
        out << ft.FunctionParameterAt(p).Name(SCOPED|QUALIFIED);
        if ( fm.FunctionParameterNameAt(p) != "" ) out << " " << fm.FunctionParameterNameAt(p);
        if ( fm.FunctionParameterDefaultAt(p) != "" ) out << " = " << fm.FunctionParameterDefaultAt(p);
        if ( p != ft.FunctionParameterSize()-1 ) out << ", ";
      }
    }
    out << ");";
    generate_comment( fm );
    out << endl;
  }

  out << indent << "};" << endl;
}
void generate_namespace(const Scope& ns, const string& indent = "" ) {

  if ( ! ns.IsTopScope() ) out << indent << "namespace "<< ns.Name() << " {" << endl;

  // Sub-Namespaces
  for ( size_t i = 0; i < ns.SubScopeSize(); i++ ) {
    Scope sc = ns.SubScopeAt(i);
    if ( sc.IsNamespace() ) generate_namespace(sc, indent + "  ");
    if ( sc.IsClass() ) generate_class(Type::ByName(sc.Name(SCOPED)), indent + "  ");
  }
  // Types----
  for ( size_t t = 0; t < ns.SubTypeSize(); t++ ) {
    Type ty = ns.SubTypeAt(t);
    if ( ty.IsClass() ) generate_class(ty, indent + "  ");
  }

  if ( ! ns.IsTopScope() ) out << indent << "}" << endl;
}

int main() {

  std::cerr << "Hello World" << std::endl;

#ifdef _WIN32
  HMODULE libInstance = LoadLibrary("libtest_Class2DictRflx.dll");
  if ( ! libInstance )  std::cout << "Could not load dictionary. " << std::endl << "Reason: " << GetLastError() << std::endl;
#else
  void * libInstance = dlopen("libtest_Class2DictRflx.so", RTLD_LAZY);
  if ( ! libInstance )  std::cout << "Could not load dictionary. " << std::endl << "Reason: " << dlerror() << std::endl;
#endif

  generate_namespace( Scope::GlobalScope() );

  int ret = 0;
#if defined (_WIN32)
  ret = FreeLibrary(libInstance);
  if (ret == 0) std::cout << "Unload of dictionary library failed." << std::endl << "Reason: " << GetLastError() << std::endl;
#else
  ret = dlclose(libInstance);
  if (ret == -1) std::cout << "Unload of dictionary library failed." << std::endl << "Reason: " << dlerror() << std::endl;
#endif

  Reflex::Shutdown();

  return 0;
}

 
