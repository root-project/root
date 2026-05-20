#ifndef __TEST_SELECT_NO_INSTANCE_H_
#define __TEST_SELECT_NO_INSTANCE_H_

#include "RootMetaSelection.h"

class MyClass{};
class MyClass2{};

template< class T, class BASE = char* >
class MyDataVector{
private:
   MyClass m_isMostDerived;
   MyClass2 m_isNonSplit;

};

namespace ROOT { namespace Meta { namespace Selection {

template< class T, class BASE >
class MyDataVector : KeepFirstTemplateArguments< 1 >, SelectNoInstance {

   /// Declare the automatically created variable transient
   MemberAttributes< kTransient + kAutoSelected > m_isMostDerived;
   MemberAttributes< kNonSplittable+ kAutoSelected > m_isNonSplit;
};

MyDataVector<float,bool> dummy;

}}}

#endif
