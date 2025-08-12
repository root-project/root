#include <stdio.h>
#include <TClassEdit.h>
#include <TClass.h>
#include <cstdint>

// See also roottest/meta/naming/execCheckNaming.C

// The role of ResolveTypedef is to remove typedef and should be given
// an almost normalized name.  The main purpose of this test is to
// insure that nothing is removed and no stray space is added.
// However, (at least for now), it is allowed to remove some spaces
// but does not have to remove them (this is the job of ShortType
// or the name normalization routine).

namespace A1 { namespace B2 { namespace C3 { typedef int what; } } }
namespace NS { typedef int IntNS_t; }
namespace SG { typedef std::uint32_t sgkey_t; }
namespace RT {
namespace EX {
struct ClusterSize {
   using ValueType = std::uint64_t;
   ValueType fValue;
};
using ClusterSize_t = ClusterSize;
}
}

struct PackedParameters {
   SG::sgkey_t  m_sgkey;
};

class Object
{
public:
   struct Inside {};
   typedef long value_t;
};

namespace NS {
   template <typename T, typename Q = Object> class Inner
   {

   };
   typedef ::Object Object;
}

template <typename T> class Wrapper
{
public:
   typedef T value_t;
   typedef const T const_value_t;
   typedef NS::Inner<T,Object> Point_t;
   typedef NS::Inner<T>        PointDefault_t;

   T fOne;
   value_t fTwo;
};

class cl { public: template <class T> class inner; };
typedef cl cl_t;
typedef int SomeTypedefName_t;

bool testing(const char *expected, const std::string &result)
{
   static int count = 0;
   ++count;

   if (result == expected) {
      printf("Test %d The result is correct: %s\n",count, expected);
      return true;
   } else {
      printf("Error test %d the result is %s instead of %s\n",count,result.c_str(),expected);
      return false;
   }
}

int execResolveTypedef()
{
   testing("const int",TClassEdit::ResolveTypedef("const int"));
   testing("const int",TClassEdit::ResolveTypedef("const Int_t"));
   testing("const Long64_t",TClassEdit::ResolveTypedef("const long long"));
   testing("const Long64_t",TClassEdit::ResolveTypedef("const Long64_t"));

   testing("const int&",TClassEdit::ResolveTypedef("const int&"));
   testing("const int&",TClassEdit::ResolveTypedef("const Int_t&"));
   testing("const Long64_t&",TClassEdit::ResolveTypedef("const long long&"));
   testing("const Long64_t&",TClassEdit::ResolveTypedef("const Long64_t&"));

   testing("const int*",TClassEdit::ResolveTypedef("const int*"));
   testing("const int*",TClassEdit::ResolveTypedef("const Int_t*"));
   testing("const Long64_t*",TClassEdit::ResolveTypedef("const long long*"));
   testing("const Long64_t*",TClassEdit::ResolveTypedef("const Long64_t*"));

   testing("const int *&",TClassEdit::ResolveTypedef("const int *&"));
   testing("const int*&",TClassEdit::ResolveTypedef("const Int_t*&"));
   testing("const Long64_t*&",TClassEdit::ResolveTypedef("const long long*&"));
   testing("const Long64_t*&",TClassEdit::ResolveTypedef("const Long64_t*&"));

   testing("const int*const&",TClassEdit::ResolveTypedef("const int*const&"));
   testing("const int* const &",TClassEdit::ResolveTypedef("const Int_t* const &"));
   testing("const Long64_t* const  &",TClassEdit::ResolveTypedef("const long long* const  &"));
   testing("const Long64_t*const&",TClassEdit::ResolveTypedef("const Long64_t*const&"));


   testing("int",TClassEdit::ResolveTypedef("A1::B2::C3::what"));
   testing("const int",TClassEdit::ResolveTypedef("const NS::IntNS_t"));

   testing("long",TClassEdit::ResolveTypedef("Object::value_t"));
   testing("long",TClassEdit::ResolveTypedef("Wrapper<long>::value_t"));
   testing("long",TClassEdit::ResolveTypedef("Wrapper<Long_t>::value_t"));
   testing("long",TClassEdit::ResolveTypedef("NS::Object::value_t"));

   testing("const long",TClassEdit::ResolveTypedef("Wrapper<long>::const_value_t"));
   testing("const long",TClassEdit::ResolveTypedef("Wrapper<Long_t>::const_value_t"));
   testing("const long",TClassEdit::ResolveTypedef("Wrapper<const long>::value_t"));
   testing("const long",TClassEdit::ResolveTypedef("Wrapper<const Long_t>::value_t"));

   testing("Object::Inside",TClassEdit::ResolveTypedef("NS::Object::Inside"));
   testing("Object",TClassEdit::ResolveTypedef("NS::Object"));

   // Known failure: the Double32_t is not yet propagated to the template's typedef :(
   // testing("Double32_t",TClassEdit::ResolveTypedef("Wrapper<Double32_t>::value_t"));

   testing("NS::Inner<long,Object>",TClassEdit::ResolveTypedef("Wrapper<long>::Point_t"));
   // 10.
   testing("NS::Inner<long,Object>",TClassEdit::ResolveTypedef("Wrapper<long>::PointDefault_t"));
   testing("NS::Inner<long,Object>",TClassEdit::ResolveTypedef("Wrapper<Long_t>::Point_t"));
   testing("NS::Inner<long,Object>",TClassEdit::ResolveTypedef("Wrapper<Long_t>::PointDefault_t"));

   testing("vector<long>",TClassEdit::ResolveTypedef("vector<Long_t>",true));
   testing("long",TClassEdit::ResolveTypedef("vector<Long_t>::value_type",true));

   testing("pair<vector<long>,int>",TClassEdit::ResolveTypedef("pair<vector<Long_t>,vector<Int_t>::value_type>",true));
   testing("pair<vector<long>,long>",TClassEdit::ResolveTypedef("pair<vector<Long_t>,vector<Long_t>::value_type>",true));

   testing("pair<pair<long,vector<int> >,long>",TClassEdit::ResolveTypedef("pair<pair<Long_t,vector<int>>,vector<Long_t>::value_type>",true));
   testing("pair<pair<long,vector<int> >,long>",TClassEdit::ResolveTypedef("pair<pair<Long_t,vector<int> >,vector<Long_t>::value_type>",true));

   testing("pair<vector<long>,long*>",TClassEdit::ResolveTypedef("pair<vector<long>,vector<long>::value_type*>",true));
   testing("pair<vector<long>,long*>",TClassEdit::ResolveTypedef("pair<vector<Long_t>,vector<Long_t>::value_type*>",true));

   testing("int",TClassEdit::ResolveTypedef("Int_t"));
   testing("Long64_t",TClassEdit::ResolveTypedef("Long64_t"));
   testing("Long64_t",TClassEdit::ResolveTypedef("long long"));
   testing("vec<Long64_t>",TClassEdit::ResolveTypedef("vec<Long64_t>"));
   testing("vec<Long64_t>",TClassEdit::ResolveTypedef("vec<long long>"));
   testing("Long64_t",TClassEdit::ResolveTypedef("vector<Long64_t>::value_type"));
   testing("Long64_t",TClassEdit::ResolveTypedef("vector<long long>::value_type"));

   testing("testing_iterator<pair<const unsigned int,TGLPhysicalShape*> >::_Base_ptr*",TClassEdit::ResolveTypedef("testing_iterator<pair<const unsigned int,TGLPhysicalShape*> >::_Base_ptr*"));
   testing("testing_iterator<pair<const unsigned int,TGLPhysicalShape*> >::_Base_ptr*",TClassEdit::ResolveTypedef("testing_iterator<pair<const UInt_t,TGLPhysicalShape*> >::_Base_ptr*"));

   // Known failure: the Long64_t is not yet propagated to the template's typedef :(
   // testing("NS::Inner<Long64_t,Object>",TClassEdit::ResolveTypedef("Wrapper<Long64_t>::Point_t"));

   testing(                           "!=<const Roo*,const Roo*,std::vector<RooFunction> >",
           TClassEdit::ResolveTypedef("!=<const Roo*, const Roo*, std::vector<RooFunction> >"));

   // TClassEdit::ResolveTypedef's job is *not* (yet?) to clean up the spaces.
   //testing(                           "!=<const Roo*,const Roo*,std::vector<RooFunction> >",
   //        TClassEdit::ResolveTypedef("!=<const Roo *, const Roo *, std::vector<RooFunction >>"));

   //testing("!=<const RooStats::HistFactory::PreprocessFunction*,const RooStats::HistFactory::PreprocessFunction*,std::vector<RooStats::HistFactory::PreprocessFunction> >",
   //        TClassEdit::ResolveTypedef("!=<const RooStats::HistFactory::PreprocessFunction*,const RooStats::HistFactory::PreprocessFunction*,std::vector<RooStats::HistFactory::PreprocessFunction> >"));
   //testing(                           "!=<const RooStats::HistFactory::PreprocessFunction*,const RooStats::HistFactory::PreprocessFunction*,std::vector<RooStats::HistFactory::PreprocessFunction> >",
   //        TClassEdit::ResolveTypedef("!=<const RooStats::HistFactory::PreprocessFunction *, const RooStats::HistFactory::PreprocessFunction *, std::vector<RooStats::HistFactory::PreprocessFunction >>"));

   testing("vec<const int>",TClassEdit::ResolveTypedef("vec< const int>"));
   testing("vec<const int>",TClassEdit::ResolveTypedef("vec<  const int>"));
   testing("vec<const int>",TClassEdit::ResolveTypedef("vec< const Int_t>"));
   testing("vec<const int>",TClassEdit::ResolveTypedef("vec< Int_t  const>"));
   testing("vec<const int>",TClassEdit::ResolveTypedef("vec<int const>"));
   testing("unknown::wrapper<int>",TClassEdit::ResolveTypedef("unknown::wrapper<Int_t>"));
   testing("std::pair<char,unknown::wrapper<int> >",TClassEdit::ResolveTypedef("std::pair<Char_t,unknown::wrapper<Int_t>>"));

   printf("Starting GetNormalizedName tests\n");

   std::string output;
   TClassEdit::GetNormalizedName(output,"Wrapper<long>::Point_t");
   testing("NS::Inner<long,Object>",output);

   TClassEdit::GetNormalizedName(output,"Wrapper<Long_t>::Point_t");
   testing("NS::Inner<long,Object>",output);

   TClassEdit::GetNormalizedName(output,"Wrapper<long long>::Point_t");
   testing("NS::Inner<Long64_t,Object>",output);

   TClassEdit::GetNormalizedName(output,"Wrapper<long long>::PointDefault_t");
   testing("NS::Inner<Long64_t,Object>",output);

   TClassEdit::GetNormalizedName(output,"Wrapper<Long64_t>::Point_t");
   testing("NS::Inner<Long64_t,Object>",output);

   TClassEdit::GetNormalizedName(output,"Wrapper<Long64_t>::PointDefault_t");
   testing("NS::Inner<Long64_t,Object>",output);

   TClassEdit::GetNormalizedName(output,"pair<vector<Long_t>,vector<Long_t>::value_type>");
   testing("pair<vector<long>,long>",output);

   TClassEdit::GetNormalizedName(output,"NS::Inner<Int_t>");
   testing("NS::Inner<int,Object>",output);

   TClassEdit::GetNormalizedName(output,"Wrapper<NS::Inner<Int_t> >");
   testing("Wrapper<NS::Inner<int,Object> >",output);

   TClassEdit::GetNormalizedName(output,"vector2<NS::Inner<Int_t> >");
   testing("vector2<NS::Inner<int,Object> >",output);

   TClassEdit::GetNormalizedName(output,"vector<NS::Inner<Int_t> >");
   testing("vector<NS::Inner<int,Object> >",output);

   testing("RootPCtempObj<const TObject*const>",TClassEdit::ResolveTypedef("RootPCtempObj<TObject const*const>"));
   testing("vector<const cl::inner<const long>*const>",TClassEdit::ResolveTypedef("vector<cl_t::inner<long const> const*const>"));
   testing("const cl::inner<const long>*const",TClassEdit::ResolveTypedef("cl_t::inner<long const> const*const"));
   testing("vector<const cl::inner<const long>*const>",TClassEdit::ResolveTypedef("vector<cl_t::inner<const long> const*const>"));
   testing("const cl::inner<const long>*const",TClassEdit::ResolveTypedef("cl_t::inner<const long> const*const"));
   testing("vector<const cl::inner<const long>*const>",TClassEdit::ResolveTypedef("vector<const cl_t::inner<const long> *const>"));
   testing("const cl::inner<const long>*const",TClassEdit::ResolveTypedef("const cl_t::inner<const long> *const"));

   testing("::int",TClassEdit::ResolveTypedef("::SomeTypedefName_t"));
   testing("::Unknown",TClassEdit::ResolveTypedef("::Unknown"));
   testing("::SomeTypedefName_tSF",TClassEdit::ResolveTypedef("::SomeTypedefName_tSF")); // the last 2 characters used to be ignored.
   testing("::int",TClassEdit::ResolveTypedef("::Int_t"));
   // Add an example like pair<...::type_t,int>

   testing("unsigned int", TClassEdit::ResolveTypedef("SG::sgkey_t"));
   testing("unsigned int", TClass::GetClass("PackedParameters")->GetDataMember("m_sgkey")->GetTrueTypeName());
   testing("SG::sgkey_t", TClass::GetClass("PackedParameters")->GetDataMember("m_sgkey")->GetFullTypeName());
   testing("RT::EX::ClusterSize", TClassEdit::ResolveTypedef("RT::EX::ClusterSize_t"));
   return 0;
}
