#ifndef __CLASSES_DICT__SELECTION_H_
#define __CLASSES_DICT__SELECTION_H_

#include "RootMetaSelection.h"

// Simple class
class classVanilla{};

// Simple templated class
template <class A> class classTemplateVanilla {};
classTemplateVanilla<char> t0;

// Class with transient member
class classTransientMember{
private:
   int transientMember;
};

// Class which gets autoselected since a property is specified for one of its
// instances which is a data member of classTestAutoselect
class classAutoselected{};

// Class used to test autoselection of classes
class classTestAutoselect{
private:
   classAutoselected autoselected;
};

// Class used to test the specification of attributes via traits
class classWithAttributes{};

// Templated class used to test:
// 1. The presence of a transient member
// 2. An autoselected class
// 3. Class properties
class classAutoselectedFromTemplateElaborate1{};
class classAutoselectedFromTemplateElaborate2{};
template <class A> class classTemplateElaborate {
private:
   classAutoselectedFromTemplateElaborate1 autoselectedMember;
   classVanilla transientMember;
   classAutoselectedFromTemplateElaborate2 autoselectedMemberAndTransientMemberMember;
};

classTemplateElaborate<char> classTemplateElaborate1;


// Template class used to test suppression of template arguments
template < typename T,
           typename U=int,
           typename X=classVanilla > class classRemoveTemplateArgs{};
classRemoveTemplateArgs<float> t1;
           
// Template class used to test suppression of template arguments, but in a
// namespace
namespace testNs {
  template < typename T,
  typename U=int,
  typename X=classVanilla > class classRemoveTemplateArgs{};
}
// testNs::classRemoveTemplateArgs<float*> t2;
// testNs::classRemoveTemplateArgs<int*> t3;
// testNs::classRemoveTemplateArgs<classRemoveTemplateArgs<classAutoselected> > t4;

template<typename T, typename U, typename V, typename Z=int> class A{};
template<typename T, typename U=int> class B{};

A<B<double,double>,int,float > testNested;

// The selection namespace
namespace ROOT{
   namespace Meta {
      namespace Selection{

         class classVanilla{};

         template <typename A> class classTemplateVanilla{};
         classTemplateVanilla<char> t0;

         template <typename A> class classTemplateElaborate:
            ClassAttributes <kNonSplittable>{
               MemberAttributes<kAutoSelected> autoselectedMember;
               MemberAttributes<kTransient> transientMember;
               MemberAttributes<kTransient + kAutoSelected> autoselectedMemberAndTransientMemberMember;
            };
         classTemplateElaborate<char> classTemplateElaborate_inst1;

         class classTestAutoselect{
            MemberAttributes<kAutoSelected> autoselected;
         };

         class classTransientMember{
            MemberAttributes<kTransient> transientMember;
         };

         class classWithAttributes : ClassAttributes <kNonSplittable> {};

         template < typename T, typename U=int, typename X=classVanilla >
            class classRemoveTemplateArgs : KeepFirstTemplateArguments<2> {};

         template<typename T, typename U, typename V, typename Z=int> class A: KeepFirstTemplateArguments<3> {};
         template<typename T, typename U=int> class B: KeepFirstTemplateArguments<1> {};
            
         namespace testNs{
            template < typename T, typename U=int, typename X=classVanilla >
               class classRemoveTemplateArgs : KeepFirstTemplateArguments<1> {};
         }

      }
   }
}

#endif