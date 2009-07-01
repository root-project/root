// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

#ifndef Reflex_DictSelection
#define Reflex_DictSelection

#include "Reflex/Kernel.h"

/**
 * @file  DictSelection.h
 * @author scott snyder
 * @author Stefan Roiser (minor changes, mainly documentation)
 * @date Aug 2005
 * @brief Definitions for selection classes to provide metadata
 *        for SEAL dictionary generation.
 *
 * When generating dictionary information for a class,
 * one sometimes wants to specify additional information
 * beyond the class definition itself, for example, to specify
 * that certain members are to be treated as transient by the persistency
 * system.  This can be done by associating a dictionary selection class
 * with the class for which dictionary information is being generated.
 * The contents of this selection class encode the additional information.
 * Below, we first discuss how to associate a selection class
 * with your class; then we list the current Set of information
 * which may appear inside the selection class.
 *
 * The simplest case is for the case of a non-template class @c C.
 * By default, the Name of the selection class is then
 * @c Reflex::selection::C.  If you have such a class, it will be found
 * automatically.  If @c C is in a namespace, @c NS::C, then
 * the selection class should be in the same namespace: @c Reflex::selection::NS::C.
 * Examples:
 *
 * @code
 *   namespace N {
 *     class C { ... };
 *   }
 *  namespace Reflex {
 *    namespace selection {
 *      namespace N {
 *        class C { ... };
 *      }
 *    }
 *  }
 *
   @endcode
 *
 * If, however, we're dealing with a template class, @c C\<T>, then
 * things are trickier, since one needs to be sure that the
 * selection class gets properly instantiated.  As before, the dictionary
 * generator will look for a class @c Reflex::selection::C\<T> (with the same
 * template arguments as @c C).  This will only succeed, however,
 * if the selection class has otherwise been used.  Example:
 *
 * @code
 *
 * template <class T>
 * class C { ... };
 *
 *  namespace Reflex {
 *    namespace selection {
 *      template <class T>
 *      class C { ... };
 *    }
 *  }
 *
 * struct foo { C<int> x; };
 *
 * // Without this, the selection class won't be fully instantiated.
 * struct foo_selection { Reflex::selection::C<int> x; }
   @endcode
 *
 * What one would really like is a way to ensure that the selection class
 * gets instantiated whenever the class its describing does.  That does
 * not seem to be possible without modifying that class (at least not
 * without changes to gccxml).  The following idiom seems to work:
 *
 * @code
 *
 * template <class T> class Reflex::selection::C; // forward declaration
 *
 * template <class T>
 * class C
 * {
 *   ...
 *   typedef typename Reflex::selection::C<T>::self DictSelection;
 * };
 *
 *     namespace Reflex {
 *       namespace selection {
 *         template <class T>
 *         class C
 *         {
 *           typedef DictSelection<C> self;
 *           ...
 *         };
 *       }
 *     }
 *
   @endcode
 *
 * Note that if you instead use
 *
 * @code
 *
 *   typedef Reflex::selection::C<T> DictSelection;
 *
   @endcode
 *
 * then @c Reflex::selection::C\<T> will not be fully instantiated.
 *
 * We turn now to declarations the may be present in the selection class.
 * Below, we'll call the class being described by the selection class @c C.
 *
 * @Reflex::selection::AUTOSELECT
 *
 *   This can be useful for automatically including classes which @c C depends upon.
 *
 *   @code
 *   template <class T> class Reflex::selection::C; // forward declaration
 *
 *   // class C<T> depends on std::vector<T>.
 *   template <class T>
 *   class C
 *   {
 *   public:
 *     std::vector<T> fX;
 *     typedef typename Reflex::selection::C<T>::self DictSelection;
 *   };
 *
 *     namespace Reflex {
 *       namespace selection {
 *         template <class T>
 *         class C
 *         {
 *           typedef DictSelection<C> self;
 *           AUTOSELECT fX;
 *         };
 *       }
 *     }
 *
 * // The above declarations mark both C<T> and std::vector<T>
 * // as autoselect.  This means that dictionary information for them
 * // will be emitted wherever it's needed --- no need to list them
 * // in selection.xml.
 *
   @endcode
 *
 * @Reflex::selection::TRANSIENT
 *
 *   This declaration marks the corresponding MemberAt in @c C with
 *   the same Name as transient.  This allows the transient flag
 *   to be listed once in the class header, rather than having
 *   to list it in selection.xml (in possibly many places if @c C
 *   is a template class).  Example:
 *
 *   @code
 *
 *   class C
 *   {
 *   public:
 *     int fX;
 *     int fY;  // This shouldn't be saved.
 *   };
 *
 *     namespace Reflex {
 *       namespace selection {
 *         class C
 *         {
 *           TRANSIENT fY; // Don't save C::fY.
 *         };
 *       }
 *     }
 *
   @endcode
 *
 * @Reflex::selection::TEMPLATE_DEFAULTS<T1, T2, ...>
 *
 *   (The Name of the MemberAt used does not matter.)
 *   Declares default template arguments for @c C.  Up to 15 arguments
 *   may be listed.  If a given position cannot be defaulted, then
 *   use @c Reflex::selection::NODEFAULT.
 *
 *   If this declaration has been made, then any defaulted template
 *   arguments will be suppressed in the external representations
 *   of the class Name (such as seen by the persistency service).
 *   This can be used to add a new template argument to a class
 *   without breaking backwards compatibility.
 *   Example:
 *
 *   @code
 *   template <class T, class U> class Reflex::selection::C; // forward declaration
 *
 *   template <class T, class U=int>
 *   class C
 *   {
 *   public:
 *     ...
 *     typedef typename Reflex::selection::C<T>::self DictSelection;
 *   };
 *
 *     namespace Reflex {
 *       namespace selection {
 *         template <class T, class U>
 *         class C
 *         {
 *           typedef DictSelection<C> self;
 *
 *           TEMPLATE_DEFAULTS<NODEFAULT, int> dummy;
 *         };
 *       }
 *     }
 * // With the above, then C<T,int> will be represented externally
 * // as just `C<T>'.
 *
   @endcode
 */


namespace Reflex {
namespace Selection {
/*
 * @brief turn of autoselection of the class
 *
 * By default classes which appear in the Selection namespace will be selected
 * for dictionary generation. If a class has a member of type NO_SELF_AUTOSELECT
 * no dictionary information for this class will be generated.
 */
class RFLX_API NO_SELF_AUTOSELECT {};


/*
 * @brief Mark a MemberAt as being transient.
 *
 * This should be used in a selection class.  This marks the corresponding
 * MemberAt as being transient.  See the header comments for examples.
 */
class RFLX_API TRANSIENT {};


/*
 * @brief Mark the At of a (data)MemberAt as autoselected.
 *
 * This should be used in a selection class. The Name of the MemberAt shall be the same
 * as the MemberAt in the original class and will be automatically
 * selected to have dictionary information generated wherever it's
 * needed.  See the header comments for examples.
 */
class RFLX_API AUTOSELECT {};


/*
 * @brief Placeholder for @c TEMPLATE_DEFAULTS.
 *
 * This is used in the @c TEMPLATE_DEFAULTS template argument list
 * for positions where template arguments cannot be defaulted.
 */
struct RFLX_API NODEFAULT {};


/*
 * @brief Declare template argument defaults.
 *
 * This should be used in a selection class.  The template arguments
 * of this class give the template argument defaults for the class
 * being described.  If the class is used with defaulted template
 * arguments, then these arguments will be omitted from external
 * representations.  See the header comments for examples.
 */
template <class T1 = NODEFAULT,
          class T2 = NODEFAULT,
          class T3 = NODEFAULT,
          class T4 = NODEFAULT,
          class T5 = NODEFAULT,
          class T6 = NODEFAULT,
          class T7 = NODEFAULT,
          class T8 = NODEFAULT,
          class T9 = NODEFAULT,
          class T10 = NODEFAULT,
          class T11 = NODEFAULT,
          class T12 = NODEFAULT,
          class T13 = NODEFAULT,
          class T14 = NODEFAULT,
          class T15 = NODEFAULT>
struct TEMPLATE_DEFAULTS {
   typedef NODEFAULT nodefault;
   typedef T1 t1;
   typedef T2 t2;
   typedef T3 t3;
   typedef T4 t4;
   typedef T5 t5;
   typedef T6 t6;
   typedef T7 t7;
   typedef T8 t8;
   typedef T9 t9;
   typedef T10 t10;
   typedef T11 t11;
   typedef T12 t12;
   typedef T13 t13;
   typedef T14 t14;
   typedef T15 t15;
};

}    // namespace Selection

} // namespace Reflex


#endif // Reflex_DictSelection
