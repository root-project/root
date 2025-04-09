// $Id: BaseInfo.icc,v 1.9 2008-12-15 15:12:39 ssnyder Exp $
/**
 * @file  SGTools/BaseInfo.icc
 * @author scott snyder
 * @date Nov 2005
 * @brief Provide an interface for finding inheritance information
 *        at run time.
 *        Implementation file.
 */


#include <type_traits>


namespace SG {


//===========================================================================
// Inheritance data representation classes.
//


struct NoBase {};

template <class T> struct Virtual {};

template <class T>
struct Bases
{
  typedef NoBase Base1;
  typedef NoBase Base2;
  typedef NoBase Base3;
};




//===========================================================================
// Internal implementation class for @a BaseInfo.
// This is used as a singleton, and should be accessed using the @a BaseInfo
// wrapper class.
//

/**
 * @brief Internal implementation class for @a BaseInfo.
 */
template <class T>
class BaseInfoImpl
  : public BaseInfoBase
{
public:
  /**
   * @brief Constructor.
   */
  BaseInfoImpl();


  /**
   * @brief Add information about base class @a B (for @a T).
   * @param is_virtual True if the derivation from @a B to @a T
   *                   is via virtual derivation.
   */
  template <class B>
  void add_base (bool is_virtual)
  {
    // Make sure the bib for the base class exists.
    (void)BaseInfo<B>::baseinfo();

    // Add the information for this base.
    //    this->add_info (typeid(B),
    //                converter<B>, converterTo<B>, is_virtual);
  }


};


//===========================================================================
// Initialization.
// Here we walk the class hierarchy, calling @a add_base for each base.


/**
 * @brief Generic initializer for base @a B.
 */
template <class B>
struct BaseInfo_init
{
  template <class T>
  static void init (BaseInfoImpl<T>& c, bool is_virtual)
  {
    // Here, we initialize the @a BaseInfo for @a T for base class @a B.
    // First, we add the information for this base to the instance.
    c.template add_base<B>(is_virtual);

    // Then we recurse on each possible base.
    BaseInfo_init<typename Bases<B>::Base1>::init (c, is_virtual);
     //BaseInfo_init<typename Bases<B>::Base2>::init (c, is_virtual);
     //BaseInfo_init<typename Bases<B>::Base3>::init (c, is_virtual);
  }
};


/**
 * @brief Dummy initializer.
 */
template <>
struct BaseInfo_init<NoBase>
{
  template <class T>
  static void init (BaseInfoImpl<T>& /*c*/, bool /*is_virtual*/)
  {
    // This gets called when there is no base in a slot
    // (signaled by the use of @a NoBase).
    // This ends the recursion.
  }
};



/**
 * @brief Constructor.
 */
template <class T>
BaseInfoImpl<T>::BaseInfoImpl ()
  : BaseInfoBase (typeid(T))
{
  // This starts the walk over the bases.
  // We start with @a T itself.
  // The virtual flag is initially false.
  BaseInfo_init<T>::init (*this, false);
}



/**
 * @brief Return the non-templated @c BaseInfoBase object for this type.
 */
template <class T>
const BaseInfoBase* BaseInfo<T>::baseinfo()
{
  return instance();
}


/**
 * @brief Return a reference to the (singleton) implementation object
 *        for this class.
 */
template <class T>
const BaseInfoImpl<T>* BaseInfo<T>::instance()
{
  BaseInfoImpl<T>* inst = s_instance.instance;
  if (inst)
    inst->maybeInit();
  return inst;
}


/**
 * @brief Constructor to get the singleton instance set up.
 */
template <class T>
BaseInfo<T>::instance_holder::instance_holder()
{
  fprintf(stderr,"BaseInfo<T>::instance_holder::instance_holder()\n");
  static BaseInfoImpl<T> inst;
  instance = &inst;
}

/// Declare the static member of @c BaseInfo.
template <class T>
typename BaseInfo<T>::instance_holder BaseInfo<T>::s_instance;


/**
 * @brief Helper to get @c BaseInfo initialized.
 */
template <class T>
struct RegisterBaseInit
{
  RegisterBaseInit()
#ifdef __GNUC__
    // Force this function to appear as a symbol in the output file,
    // even in an optimized build where it's always inlined.
    // Otherwise, we get complaints from cling that it can't find the symbol
    // (as of root 6.04).
    //    __attribute__ ((used))
#endif
  ;
};


template <class T>
RegisterBaseInit<T>::RegisterBaseInit()
{
  BaseInfoBase::addInit(&typeid(T), BaseInfo<T>::baseinfo);
}


template <class T> struct BaseInit {
  static RegisterBaseInit<T> s_regbase;
};
template <class T> RegisterBaseInit<T> BaseInit<T>::s_regbase;







} // namespace SG

