#ifndef POOL_EMBEDDED_OBJECTS_H
#define POOL_EMBEDDED_OBJECTS_H
#include "TObject.h"
#include <vector>

class Embedded_objects {
public:
   /** @class EmbeddedClasses
    *
    * Example of the use of an embedded classes.
    *
    * @author  M.Frank
    * @version 1.0
    */
   class EmbeddedClasses;
   typedef EmbeddedClasses EmbeddedTypedef; 
   
   class EmbeddedClasses {
   public:
      
      /** @class Embedded1
       * A nested embedded class.
       * @author M.Frank
       */
      class Embedded1 {
      public:
         /// Integer aggregate
         int i;
         /// Standard Constructor
         Embedded1();
         /// Standard Destructor
         virtual ~Embedded1();
      };
      /** @class Embedded1
       * A nested embedded class with an aggregate.
       * @author M.Frank
       */
      class Embedded2 {
      public:
         /// Double aggregate
         double d;
         /// Aggregate nested class in nested class
         Embedded1 m_embed1;
         /// Standard Constructor
         Embedded2();
         /// Standard Destructor
         virtual ~Embedded2();
      };
      /** @class Embedded1
       * A nested embedded class with an aggregate.
       * @author M.Frank
       */
      class Embedded3 {
      public:
         /// Aggregate nested class in nested class
         Embedded2 m_embed2;
         /// Float aggregate
         float f;
         /// Standard Constructor
         Embedded3();
         /// Standard Destructor
         virtual ~Embedded3();
      };
      /// Aggregate nested class in nested class + pointer to nested class
      Embedded3  /* m_emb3, */ *m_pemb3;
      /// Aggregate nested class in nested class + pointer to nested class
      Embedded2  /* m_emb2, */ *m_pemb2;
      /// Aggregate nested class in nested class + pointer to nested class
      Embedded1  /* m_emb1, */ *m_pemb1;
      /// Standard Constructor
      EmbeddedClasses();
      /// Standard Destructor
      virtual ~EmbeddedClasses();
   };
   EmbeddedClasses m_embedded;
   EmbeddedClasses::Embedded1 m_emb1;
   EmbeddedClasses::Embedded2 m_emb2;
   EmbeddedClasses::Embedded3 m_emb3;

   EmbeddedTypedef::Embedded3 m_emb4;
   Embedded_objects::EmbeddedClasses::Embedded3 m_emb5;
   Embedded_objects::EmbeddedTypedef::Embedded3 m_emb6;
   typedef std::vector<int> vecint;
   vecint::iterator m_iter; //!

   Embedded_objects();
   virtual ~Embedded_objects();
   void initData(int i);
   void dump() const;
};

class Normal_objects {
public:
   Embedded_objects emb;
   int i, j, k, l, m;
   Normal_objects();
   virtual ~Normal_objects();
   void initData(int i);
   void dump() const;
};


#endif // POOL_EMBEDDED_OBJECTS_H
