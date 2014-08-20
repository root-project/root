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
                               indexed_by< ordered_unique< identity< ComplexElement > >,
                                           ordered_unique< tag< DummyCounter >,
                                                           composite_key< ComplexElement*,
                                                                          const_mem_fun< ComplexElement,
                                                                                         int,
                                                                                         &ComplexElement::level >
                                                         >
                              > > > ClassFail;

bool test(const std::string &input)
{
   std::string output;
   TClassEdit::GetNormalizedName(output,input.c_str());

   if (input != output) {
      fprintf(stdout,"discrepancy:\n%s\n%s\n",input.c_str(),output.c_str());
      return false;
   }
   return true;
}

int execClassEditNormalize() {

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
