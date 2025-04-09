#ifndef SGTOOLS_BASEINFO_H
#define SGTOOLS_BASEINFO_H

typedef unsigned int CLID;
static const CLID CLID_NULL = 0;

//#include "GaudiKernel/ClassID.h"
#include <vector>
#include <typeinfo>


#define SG_BASE(D, B) SG_BASES1(D, B)


#define SG_BASES1(D, B)          \
  namespace SG        {          \
    template<> struct Bases<D >{ \
      typedef B Base1;           \
      typedef NoBase Base2;      \
      typedef NoBase Base3;      \
    };                           \
    template struct BaseInit<D >; \
} struct sg_dummy // to swallow semicolon


namespace SG {


/**
 * @brief Helper metafunction to get base class types.
 *
 * For a class @c T,
 *@code
 *   SG::BaseType<SG::Bases<T>::Base1>::type
 @endcode
 * gives the type of @c T's first base.  Also,
 *@code
 *   SG::BaseType<SG::Bases<T>::Base1>::is_virtual
 @endcode
 * tells whether the derivation is virtual
 * (either @c true_type or @c false_type).
 *
 * Note that @c SG::Bases\<T>::Base1 is not the actual type
 * of @c T's first base if virtual derivation was used.
 */
template <class T>
struct BaseType;


// Forward declaration.
template <class T>
class BaseInfoImpl;


struct BaseInfoBaseImpl;


//===========================================================================
// Copy conversion declarations.
//


/**
 * @brief The non-template portion of the @a BaseInfo implementation.
 */
class BaseInfoBase
{
public:

  /// Type for an initialization function.
  typedef const BaseInfoBase* init_func_t();


  /**
   * @brief Register an initialization function.
   * @param tinfo The @c std::type_info for the class being registered.
   * @param init_func Function to initialize @c BaseInfo for the class.
   */
  static void addInit (const std::type_info* tinfo,
                       init_func_t* init_func) {};


  /**
   * @brief Run initializations for this class, if needed.
   */
  void maybeInit() {};


protected:
  /**
   * @brief Constructor.
   * @param tinfo The @c std::type_info for this class.
   */
  BaseInfoBase (const std::type_info& tinfo) {};


  /**
   * @brief Destructor.
   */
  ~BaseInfoBase() {};


private:


  BaseInfoBase (const BaseInfoBase&);
  BaseInfoBase& operator= (const BaseInfoBase&);
};


//===========================================================================
// The templated @c BaseInfo class.
//

/**
 * @brief Provide an interface for finding inheritance information
 *        at run time.  See the file comments for full details.
 */
template <class T>
class BaseInfo
{
public:

   BaseInfo() { fprintf(stderr,"Creating a BaseInfo\n"); }

  /**
   * @brief Return the non-templated @c BaseInfoBase object for this type.
   */
  static const BaseInfoBase* baseinfo ();


private:
  /// Return a reference to the (singleton) implementation object
  /// for this class.
  static const BaseInfoImpl<T>* instance();

  /// This holds the singleton implementation object instance.
  struct instance_holder
  {
    instance_holder();
    BaseInfoImpl<T>* instance;
  };
  static instance_holder s_instance;
};


} // namespace SG


#include "ROOT-7775/BaseInfo.i.h"


#endif // not SGTOOLS_BASEINFO_H

