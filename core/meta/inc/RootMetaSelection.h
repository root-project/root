#ifndef ROOT_Meta_Selection
#define ROOT_Meta_Selection

namespace ROOT{
   namespace Meta{
      namespace Selection{

// Indentation to the left: we know that everything is in this namespace and we
// don't sacrifice 12 chars to underline this.

template <unsigned int I> class KeepFirstTemplateArguments{};

///\brief Describes the attributes of a class
enum EClassAttributes {
      ///\brief Indicates absence of properties
      kClassNullProperty   = 0,
      ///\brief The class cannot be split
      kNonSplittable       = 2};

///\brief Used to specify attributes of classes in the "DictSelection" syntax                        
template <unsigned int classAttributes = kClassNullProperty> class ClassAttributes{};
                              

///\brief Describes attributes of a data member
enum EClassMemberAttributes {
      ///\brief Indicates absence of properties
      kMemberNullProperty = 0,
      ///\brief The data member is transient
      kTransient          = 2,
      ///\brief Select the type of the member
      kAutoSelected       = 4};

///\brief Used to specify attributes of data members in the "DictSelection" syntax
template <unsigned int memberAttributes = kMemberNullProperty > class MemberAttributes{};

      }
   }   
}

#endif 

