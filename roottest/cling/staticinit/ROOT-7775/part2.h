#ifndef PART2_H
#define PART2_H


namespace SG {
   template <typename T> struct Wrapper;

   template <class T>
   class BaseInfoImpl {};

   template <class T>
   class BaseInfo
   {
   public:
      static const BaseInfoImpl<T>& instance();

      /// This holds the singleton implementation object instance.
      struct instance_holder
      {
         instance_holder();
         BaseInfoImpl<T>* instance;
      };
      static instance_holder s_instance;
   };


   template <class T>
   typename BaseInfo<T>::instance_holder BaseInfo<T>::s_instance;

   template <class T>
   struct RegisterBaseInit
   {
      RegisterBaseInit(){ fprintf(stderr,"Creating RegisterBaseInit<T>\n"); }
   };

   template <class T> struct BaseInit {
      static RegisterBaseInit<T> s_regbase;
   };
   template <class T> RegisterBaseInit<T> BaseInit<T>::s_regbase;

   template <> struct Wrapper<int>
   {
   public:
      Wrapper() { fprintf(stderr,"Creating Wrapper<int>\n"); }
   };
   template struct BaseInit<int >;
   
} struct sd_dummy;


#endif