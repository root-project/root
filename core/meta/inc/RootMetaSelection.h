/* This file contains the syntax to implement the Root Meta Selection, which is  
 * an evolution of the Reflex::selection technique in ROOT 5.
*/


// // user header
// template <class T, class U = int> class C {
// private:
//    C<T, float>* fX; // example for a "dependent" dictionary
// };
// 
// // selection header, to be exposed to genreflex
// namespace ROOT {
//    namespace Selection {
//       
//       template <class T, class U = int> class C:
//       public HideLastDefaultTemplateArguments<1>,
//       DoNotSelect {
//          Dict<kSelected + kTransient> fX;
//       };
//    }
// }


namespace ROOT{
   namespace Meta{
      namespace Selection{

// Indentation to the left: we know that everything is in this namespace and we
// don't sacrifice 12 chars to underline this.

template <unsigned int I> class KeepFirstTemplateArguments{};

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
Warning("RootMetaSelection", "Strongly typed enum still absent despite the move to c++11.");
#endif
#endif

enum EClassMemberAttributes { kMemberNullProperty, kTransient, kAutoSelected };
enum EClassAttributes { kClassNullProperty, kNonSplittable};

template <EClassAttributes classAttributes1=kClassNullProperty,
          EClassAttributes classAttributes2=kClassNullProperty,
          EClassAttributes classAttributes3=kClassNullProperty,
          EClassAttributes classAttributes4=kClassNullProperty,
          EClassAttributes classAttributes5=kClassNullProperty> class ClassAttributes{};


#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
Warning("RootMetaSelection", "Variadic template still absent despite the move to c++11");
#endif
#endif

template <EClassMemberAttributes classMemberAttributes1=kMemberNullProperty,
          EClassMemberAttributes classMemberAttributes2=kMemberNullProperty,
          EClassMemberAttributes classMemberAttributes3=kMemberNullProperty,
          EClassMemberAttributes classMemberAttributes4=kMemberNullProperty,
          EClassMemberAttributes classMemberAttributes5=kMemberNullProperty> class MemberAttributes{};

      }
   }   
}