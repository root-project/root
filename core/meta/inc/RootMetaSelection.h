#ifndef ROOT_Meta_Selection
#define ROOT_Meta_Selection

namespace ROOT {
   namespace Meta {
      namespace Selection {

         ///\brief Used to specify the number of arguments to be kept
         template <unsigned int I> class KeepFirstTemplateArguments {};

         ///\brief Used to avoid to select all instances of a template
         class SelectNoInstance {};

         ///\brief Describes the attributes of a class
         enum EClassAttributes {
            ///\brief Indicates absence of properties
            kClassNullProperty   = 0
         };

         ///\brief Used to specify attributes of classes in the "DictSelection" syntax
         template <unsigned int classAttributes = kClassNullProperty> class ClassAttributes {};

         ///\brief Describes attributes of a data member
         enum EClassMemberAttributes {
            ///\brief Indicates absence of properties
            kMemberNullProperty = 0,
            ///\brief The data member is transient
            kTransient          = 2,
            ///\brief Select the type of the member
            kAutoSelected       = 4,
            ///\brief Exclude the type of the member
            kNoAutoSelected     = 8,
            ///\brief The class cannot be split
            kNonSplittable      = 16
         };

         ///\brief Used to specify attributes of data members in the "DictSelection" syntax
         template <unsigned int memberAttributes = kMemberNullProperty > class MemberAttributes {};

      }
   }
}

#endif

