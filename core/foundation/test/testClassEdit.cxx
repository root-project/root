#include "TClassEdit.h"
#include "TInterpreter.h"
#include "TStreamerInfo.h"

#include "gtest/gtest.h"

/*
/// Return type of the function, might be empty if the function declaration string did not provide it.
std::string fReturnType;

/// Name of the scope qualification of the function, possibly empty
std::string fScopeName;

/// Name of the function
std::string fFunctionName;

/// Template arguments of the function template specialization, if any; will contain one element "" for
/// `function<>()`
std::vector<std::string> fFunctionTemplateArguments;

/// Function parameters.
std::vector<std::string> fFunctionParameters;
*/

TEST(TClassEdit, SplitFunc)
{
   TClassEdit::FunctionSplitInfo fsi;

   TClassEdit::SplitFunction("", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("bar", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("bar", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("bar()", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("bar", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo bar", fsi);
   EXPECT_EQ("foo", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("bar", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo bar baz()", fsi);
   EXPECT_EQ("foo bar", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("baz", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo::baz()", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("foo", fsi.fScopeName);
   EXPECT_EQ("baz", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo<int>::baz()", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("foo<int>", fsi.fScopeName);
   EXPECT_EQ("baz", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("bar<foo>", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("bar", fsi.fFunctionName);
   EXPECT_EQ(1u, fsi.fFunctionTemplateArguments.size());
   EXPECT_EQ("foo", fsi.fFunctionTemplateArguments[0]);
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("bar<>()", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("bar", fsi.fFunctionName);
   EXPECT_EQ(1u, fsi.fFunctionTemplateArguments.size());
   EXPECT_EQ("", fsi.fFunctionTemplateArguments[0]);
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("bar<foo> foo<bar>", fsi);
   EXPECT_EQ("bar<foo>", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("foo", fsi.fFunctionName);
   EXPECT_EQ(1u, fsi.fFunctionTemplateArguments.size());
   EXPECT_EQ("bar", fsi.fFunctionTemplateArguments[0]);
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("abc<xyz>::bar<foo> cde::fgh<ijk>::foo<bar>", fsi);
   EXPECT_EQ("abc<xyz>::bar<foo>", fsi.fReturnType);
   EXPECT_EQ("cde::fgh<ijk>", fsi.fScopeName);
   EXPECT_EQ("foo", fsi.fFunctionName);
   EXPECT_EQ(1u, fsi.fFunctionTemplateArguments.size());
   EXPECT_EQ("bar", fsi.fFunctionTemplateArguments[0]);
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("ret fgh<(i > 2 ? 3 : 4)>()", fsi);
   EXPECT_EQ("ret", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("fgh", fsi.fFunctionName);
   EXPECT_EQ(1u, fsi.fFunctionTemplateArguments.size());
   EXPECT_EQ("(i > 2 ? 3 : 4)", fsi.fFunctionTemplateArguments[0]);
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("ret foo<a, b, c<d>, e = f<g>, h, ...>()", fsi);
   EXPECT_EQ("ret", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("foo", fsi.fFunctionName);
   EXPECT_EQ(6u, fsi.fFunctionTemplateArguments.size());
   EXPECT_EQ("a", fsi.fFunctionTemplateArguments[0]);
   EXPECT_EQ("b", fsi.fFunctionTemplateArguments[1]);
   EXPECT_EQ("c<d>", fsi.fFunctionTemplateArguments[2]);
   EXPECT_EQ("e = f<g>", fsi.fFunctionTemplateArguments[3]);
   EXPECT_EQ("h", fsi.fFunctionTemplateArguments[4]);
   EXPECT_EQ("...", fsi.fFunctionTemplateArguments[5]);
   EXPECT_TRUE(fsi.fFunctionParameters.empty());


   TClassEdit::SplitFunction("bar< foo(x::y), long long>                 "
                             "cde::fgh<ijk<x(y)>::bar>::foo<bar(a<b(c)>)>"
                             "(a::b<c::d<e::f(g::h)>*>::i j = k(l,m<n,p::o<q(r)>>))", fsi);
   EXPECT_EQ("bar< foo(x::y), long long>", fsi.fReturnType);
   EXPECT_EQ("cde::fgh<ijk<x(y)>::bar>", fsi.fScopeName);
   EXPECT_EQ("foo", fsi.fFunctionName);
   EXPECT_EQ(1u, fsi.fFunctionTemplateArguments.size());
   EXPECT_EQ("bar(a<b(c)>)", fsi.fFunctionTemplateArguments[0]);
   EXPECT_EQ(1u, fsi.fFunctionParameters.size());
   EXPECT_EQ("a::b<c::d<e::f(g::h)>*>::i j = k(l,m<n,p::o<q(r)>>)", fsi.fFunctionParameters[0]);

   TClassEdit::SplitFunction("someNamespace::someClass< std::function<void (sometype&)> >::FunctionName(std::function<void (someothertype&)>)", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("someNamespace::someClass< std::function<void (sometype&)> >", fsi.fScopeName);
   EXPECT_EQ("FunctionName", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_EQ(1u, fsi.fFunctionParameters.size());
   EXPECT_EQ("std::function<void (someothertype&)>", fsi.fFunctionParameters[0]);   
}

TEST(TClassEdit, SplitFuncErrors)
{
   TClassEdit::FunctionSplitInfo fsi;

   TClassEdit::SplitFunction("foo:", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("foo:", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction(":foo", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ(":foo", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction(":::", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ(":", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("a:::b", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("a", fsi.fScopeName);
   EXPECT_EQ(":b", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("a:b:c:d", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("a:b:c:d", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo(", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("foo", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("(foo", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo)", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("foo)", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction(")foo", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ(")foo", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo<", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("foo", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("<foo", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction("foo>", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ("foo>", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());

   TClassEdit::SplitFunction(">foo", fsi);
   EXPECT_EQ("", fsi.fReturnType);
   EXPECT_EQ("", fsi.fScopeName);
   EXPECT_EQ(">foo", fsi.fFunctionName);
   EXPECT_TRUE(fsi.fFunctionTemplateArguments.empty());
   EXPECT_TRUE(fsi.fFunctionParameters.empty());
}

// ROOT-9926
TEST(TClassEdit, GetNameForIO)
{
   const std::vector<std::pair<std::string, std::string>> names{{"T*", "unique_ptr<const T>"},
                                                                {"T*", "unique_ptr<const T*>"},
                                                                {"T*", "unique_ptr<const T* const*>"},
                                                                {"T*", "unique_ptr<T * const>"},
                                                                {"T*", "unique_ptr<T * const**const**&* const>"},
                                                                {"vector<T*>", "vector<unique_ptr<T>>"},
                                                                {"vector<const T*>", "vector<unique_ptr<const T>>"}};
   for (auto &&namesp : names) {
      EXPECT_EQ(namesp.first, TClassEdit::GetNameForIO(namesp.second.c_str()))
         << "Failure in transforming typename " << namesp.first << " into " << namesp.second;
   }
}

// ROOT-10574, https://github.com/root-project/root/issues/17295
TEST(TClassEdit, ResolveTypedef)
{
   gInterpreter->Declare("struct testPoint{}; typedef struct testPoint testPoint;");
   std::string non_existent = TClassEdit::ResolveTypedef("testPointAA");
   EXPECT_STREQ("testPointAA", non_existent.c_str());
   EXPECT_STRNE("::testPoint", TClassEdit::ResolveTypedef("::testPointXX").c_str());
   gInterpreter->Declare("typedef const int mytype_t;");
   gInterpreter->Declare("typedef const int cmytype_t;");
   EXPECT_STREQ("const int", TClassEdit::ResolveTypedef("mytype_t").c_str());
   EXPECT_STREQ("const int", TClassEdit::ResolveTypedef("cmytype_t").c_str());
   // #18833
   const char* type_18833 = "pair<TAttMarker*,TGraph*(  *  )(const std::string&,const std::string&,TH1F*) >";
   EXPECT_STREQ(type_18833, TClassEdit::ResolveTypedef(type_18833).c_str());
}

// ROOT-11000
TEST(TClassEdit, DefComp)
{
   EXPECT_FALSE(TClassEdit::IsDefComp("std::less<>", "std::string"));
}

TEST(TClassEdit, DefAlloc)
{
   // https://github.com/root-project/root/issues/6607
   EXPECT_TRUE(TClassEdit::IsDefAlloc("class std::allocator<float>", "float"));

   // Space handling issues (part of https://github.com/root-project/root/issues/18654)
   EXPECT_TRUE(TClassEdit::IsDefAlloc("std::allocator<std::pair<K,V>>", "K", "V"));
   EXPECT_TRUE(TClassEdit::IsDefAlloc("std::allocator<   std::pair<K,V>  >", "K", "V"));
   EXPECT_TRUE(TClassEdit::IsDefAlloc("std::allocator<std::pair<K,V>  const  >", "K", "V"));
}


TEST(TClassEdit, GetNormalizedName)
{
   std::string n;
   
   // https://github.com/root-project/root/issues/6607
   TClassEdit::GetNormalizedName(n, "std::vector<float, class std::allocator<float>>");
   EXPECT_STREQ("vector<float>", n.c_str());

   // https://github.com/root-project/root/issues/18643
   n.clear();
   TClassEdit::GetNormalizedName(n, "_Atomic(map<string, TObjArray* >*)");
   EXPECT_STREQ("_Atomic(map<string,TObjArray*>*)", n.c_str());

   n.clear();
   EXPECT_THROW(TClassEdit::GetNormalizedName(n, "_Atomic(map<string, TObjArray* >*"), std::runtime_error);

}

// https://github.com/root-project/root/issues/18654
TEST(TClassEdit, UnorderedMapNameNormalization)
{
   // These two should normalise to map<string,char>.
   // When this did not work, df104_CSVDataSource-py crashed while querying the classes
   std::string in_cxx11{
      "std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, char, "
      "std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, "
      "std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, "
      "std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, "
      "char> > >"};
   std::string in{"std::unordered_map<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, char, "
                  "std::hash<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, "
                  "std::equal_to<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, "
                  "std::allocator<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > "
                  "const, char> > >"};
   const auto target = "unordered_map<string,char>";

   std::string out;
   TClassEdit::GetNormalizedName(out, in);
   EXPECT_STREQ(target, out.c_str());

   TClassEdit::GetNormalizedName(out, in_cxx11);
   EXPECT_STREQ(target, out.c_str());
}

TEST(TClassEdit, SplitType)
{
   // https://github.com/root-project/root/issues/16119
   TClassEdit::TSplitType t1("std::conditional<(1 < 32), int, float>");
   EXPECT_STREQ("std::conditional", t1.fElements[0].c_str());
   EXPECT_STREQ("(1<32)", t1.fElements[1].c_str());
   EXPECT_STREQ("int", t1.fElements[2].c_str());
   EXPECT_STREQ("float", t1.fElements[3].c_str());

   gInterpreter->ProcessLine(".L file_16199.C+");
   auto c = TClass::GetClass("o2::dataformats::AbstractRef<25,5,2>");
   auto si = (TStreamerInfo*) c->GetStreamerInfo();
   si->ls("noaddr");
}
