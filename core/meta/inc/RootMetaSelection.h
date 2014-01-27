#ifndef __ROOTMETASELECTION_H__
#define __ROOTMETASELECTION_H__

namespace ROOT{
   namespace Meta{
      namespace Selection{

// Indentation to the left: we know that everything is in this namespace and we
// don't sacrifice 12 chars to underline this.

template <unsigned int I> class KeepFirstTemplateArguments{};

enum EClassAttributes { kClassNullProperty   = 0,
                        kNonSplittable       = 2};
                              
template <unsigned int classAttributes = kClassNullProperty> class ClassAttributes{};
                              


enum EClassMemberAttributes { kMemberNullProperty = 0,
                              kTransient          = 2,
                              kAutoSelected       = 4};

template <unsigned int memberAttributes = kMemberNullProperty > class MemberAttributes{};

      }
   }   
}
#endif //__ROOTMETASELECTION_H__