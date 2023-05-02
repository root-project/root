// Boost include(s):
//#include <boost/multi_index_container.hpp>
//#include <boost/multi_index/mem_fun.hpp>
//#include <boost/multi_index/ordered_index.hpp>
//#include <boost/multi_index/composite_key.hpp>

// Dummy structures:
struct DummyCounter {};

/// Class used in a tricky container
class ComplexElement {
public:
   int level() const;
};

class DefaultTag {};

/// To simplify the expression a bit
//using namespace boost::multi_index;


template <typename A, typename B, typename C = std::allocator<A> > class multi_index_container {};
template <typename A, typename B> class indexed_by {};
template <typename KeyFromValue, typename Compare =std::less<typename KeyFromValue::result_type> > class ordered_unique {};

//template <typename TagList, typename KeyFromValue, typename Compare> class ordered_unique {};
//template <typename KeyFromValue, typename Compare /* =std::less<typename KeyFromValue::result_type> */ > class ordered_unique<DefaultTag,KeyFromValue,Compare> {};
//template <typename KeyFromValue> class ordered_unique<DefaultTag,KeyFromValue,std::less<typename KeyFromValue::result_type> > {};

template <typename A> class identity { public: typedef A result_type; };
template <typename A> class tag {};
template <typename A, typename B> class composite_key {};
template<class Class, typename Type, Type (Class::*PtrToMemberFunction)()const> struct const_mem_fun {};

/// Base class for our complex class
typedef multi_index_container< ComplexElement*,
                               indexed_by< ordered_unique< ::identity< ComplexElement > >,
                                           ordered_unique< tag< DummyCounter >,
                                                           composite_key< ComplexElement*,
                                                                          const_mem_fun< ComplexElement,
                                                                                         int,
                                                                                         &ComplexElement::level >
                                                         >
                              > > > ClassFail;

/// Variadic template.
template <class Index, class... Config>
class TempVariadic {};

template <class Index, int... Config>
class TempValueVariadic {};

#include "cmsExample01.h"

class cl { public: template <class T> class inner; };
typedef cl cl_t;

bool test(const std::string &input)
{
   std::string output;
   TClassEdit::GetNormalizedName(output,input);

   if (input != output) {
      fprintf(stdout,"discrepancy:\n\texpected: %s\n\tgot: %s\n",input.c_str(),output.c_str());
      return false;
   }
   return true;
}

bool test(const std::string &input,const char *expected)
{
   std::string output;
   TClassEdit::GetNormalizedName(output,input);

   if (output != expected) {
      fprintf(stdout,"discrepancy:\n\texpected: %s\n\tgot: %s\n",expected,output.c_str());
      return false;
   }
   return true;
}

int execClassEditNormalize() {

   if (!test("TempValueVariadic<indexed_by<int,float>,1,2,3>")) return 62;

   if (!test("TempVariadic<indexed_by<int,float>,double,long,char>")) return 61;

   if (!test("std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::list<int, std::allocator<int> > >","pair<const string,list<int> >")) return 60;
   //if (!test("std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const, std::__1::list<int, std::__1::allocator<int> > >","pair<const string,list<int> >")) return 59;

   if (!test("basic_string<char,char_traits<char>,allocator<char> >","string")) return 58;
   if (!test("const basic_string<char,char_traits<char>,allocator<char> >","const string")) return 57;
   if (!test("basic_string<char,char_traits<char>,allocator<char> > const","const string")) return 56;

   if (!test("vector<cl2_t::inner<long const> const*const>","vector<const cl2_t::inner<const long>*const>")) return 55;
   if (!test("cl2_t::inner<long const> const*const","const cl2_t::inner<const long>*const")) return 54;
   if (!test("vector<cl_t::inner<long const> const*const>","vector<const cl::inner<const long>*const>")) return 53;
   if (!test("cl_t::inner<long const> const*const","const cl::inner<const long>*const")) return 52;

   if (!test("RootPCtempObj<TObject * const *>","RootPCtempObj<TObject*const*>")) return 51;
   if (!test("RootPCtempObj<TObject*const*>","RootPCtempObj<TObject*const*>")) return 50;
   if (!test("RootPCtempObj<TObject const*const>","RootPCtempObj<const TObject*const>")) return 49;
   if (!test("RootPCtempObj<const TObject*const>","RootPCtempObj<const TObject*const>")) return 48;
   if (!test("RootPCtempObj<TObject*const>","RootPCtempObj<TObject*const>")) return 47;
   if (!test("TObject const*const","const TObject*const")) return 46;
   if (!test("const TObject*const","const TObject*const")) return 45;
   if (!test("TObject*const","TObject*const")) return 44;
   if (!test("v3<Long_t const,int const>","v3<const long,const int>")) return 43;
   if (!test("v3<const Long_t,int>","v3<const long,int>")) return 42;
   if (!test("v3<Long_t const,int>","v3<const long,int>")) return 41;
   if (!test("v3<Long_t,const int>","v3<long,const int>")) return 40;
   if (!test("v3<Long_t,int const>","v3<long,const int>")) return 39;
   if (!test("v3<Long_t,int>","v3<long,int>")) return 38;

   if (!test("const int *const* const","const int*const*const")) return 37;
   if (!test("const int **const","const int**const")) return 36;
   if (!test("const int  * const","const int*const")) return 35;
   if (!test("const int *const","const int*const")) return 34;

   if (!test("int const*const* const","const int*const*const")) return 33;
   if (!test("int const**const","const int**const")) return 32;
   if (!test("int const * const","const int*const")) return 31;
   if (!test("int const*const","const int*const")) return 30;


   if (!test("int const","const int")) return 29;
   if (!test("Int_t const","const int")) return 28;
   if (!test("const int","const int")) return 27;
   if (!test("const Int_t","const int")) return 26;
   if (!test("const string","const string")) return 25;
   if (!test("string const","const string")) return 24;
   if (!test("pair<string const,Data<int> >","pair<const string,Data<int> >")) return 23;
   if (!test("pair<vector<int> const,Data<int> >","pair<const vector<int>,Data<int> >")) return 22;

    if (!test("edm::ValueMap<std::vector<edm::Ref<std::vector<reco::PFCandidate>,"
                           "reco::PFCandidate,"
                           "edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > >"
                "::value_type",
             "vector<edm::Ref<vector<reco::PFCandidate>,"
                    "reco::PFCandidate,"
                    "edm::refhelper::FindUsingAdvance<vector<reco::PFCandidate>,reco::PFCandidate> > >")) return 21;

   if (!test("const std::string","const string")) return 20;
   if (!test("pair<const std::string,int>","pair<const string,int>")) return 19;
   if (!test("allocator<pair<const std::string,int> >","allocator<pair<const string,int> >")) return 18;

   if (!test("vector<Int_t, my_allocator<int> >","vector<int,my_allocator<int> >")) return 17;
   if (!test("vector<Int_t, allocator<int> >","vector<int>")) return 17;
   if (!test("vector<int, allocator<int> >","vector<int>")) return 16;

   if (!test("Int_t","int")) return 15;
   if (!test("Double32_t","Double32_t")) return 14;

   if (!test("templ<Int_t>","templ<int>")) return 13;
   if (!test("templ<Long64_t>","templ<Long64_t>")) return 12;
   if (!test("templ<long long>","templ<Long64_t>")) return 11;
   if (!test("templ<Double32_t>","templ<Double32_t>")) return 10;

   std::string input7 = "tempname<&ComplexElement::level>";
   if (!test(input7)) return 9;

   std::string input6 = "boost::multi_index::const_mem_fun<ComplexElement,int,&ComplexElement::level>";
   if (!test(input6)) return 8;


   std::string input5 = "boost::multi_index::composite_key<ComplexElement*,boost::multi_index::const_mem_fun<ComplexElement,int,&ComplexElement::level>,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type>";
   if (!test(input5)) return 7;


   std::string input4 = "boost::multi_index::composite_key<ComplexElement*,boost::multi_index::const_mem_fun<ComplexElement,int,&ComplexElement::level>,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type>";
   if (!test(input4)) return 6;

   std::string input3 = "boost::multi_index::tag<DummyCounter,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na>";
   if (!test(input3)) return 5;

   std::string input2 = "boost::multi_index::ordered_unique<boost::multi_index::tag<DummyCounter,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na>,boost::multi_index::composite_key<ComplexElement*,boost::multi_index::const_mem_fun<ComplexElement,int,&ComplexElement::level>,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type>,mpl_::na>";
   if (!test(input2)) return 4;

   std::string input1 = "boost::multi_index::ordered_unique<boost::multi_index::identity<ComplexElement>,mpl_::na,mpl_::na>";
   if (!test(input1)) return 3;

   std::string input0 = "boost::multi_index::indexed_by<boost::multi_index::ordered_unique<boost::multi_index::identity<ComplexElement>,mpl_::na,mpl_::na>,boost::multi_index::ordered_unique<boost::multi_index::tag<DummyCounter,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na>,boost::multi_index::composite_key<ComplexElement*,boost::multi_index::const_mem_fun<ComplexElement,int,&ComplexElement::level>,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type>,mpl_::na>,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na>";
   if (!test(input0)) return 2;

   std::string input_full = "boost::multi_index::multi_index_container<ComplexElement*,boost::multi_index::indexed_by<boost::multi_index::ordered_unique<boost::multi_index::identity<ComplexElement>,mpl_::na,mpl_::na>,boost::multi_index::ordered_unique<boost::multi_index::tag<DummyCounter,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na>,boost::multi_index::composite_key<ComplexElement*,boost::multi_index::const_mem_fun<ComplexElement,int,&ComplexElement::level>,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type,boost::tuples::null_type>,mpl_::na>,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na,mpl_::na>,allocator<ComplexElement*> >";

   if (!test(input_full)) return 1;
   return 0;
}
