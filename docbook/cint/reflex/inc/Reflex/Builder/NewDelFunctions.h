// $Id: NewDelFunctions.h 2134 2007-11-30 18:07:51Z pcanal $

#ifndef Reflex_NewDelFunctions
#define Reflex_NewDelFunctions

/**
 * @file  NewDelFunctions.h
 */


namespace Reflex {
typedef void* (*NewFunc_t)(void*);
typedef void* (*NewArrFunc_t)(long size, void* arena);
typedef void (*DelFunc_t)(void*);
typedef void (*DelArrFunc_t)(void*);
typedef void (*DesFunc_t)(void*);

struct RFLX_API NewDelFunctions {
   NewFunc_t fNew;                   //pointer to a function newing one object.
   NewArrFunc_t fNewArray;           //pointer to a function newing an array of objects.
   DelFunc_t fDelete;                //pointer to a function deleting one object.
   DelArrFunc_t fDeleteArray;        //pointer to a function deleting an array of objects.
   DesFunc_t fDestructor;            //pointer to a function call an object's destructor.
};

template <class T> struct NewDelFunctionsT: public NewDelFunctions {
   static void*
   new_T(void* p) { return p ? new (p) T : new T; }

   static void*
   new_p_T(void* p) { return p ? new (p) T : ::new T; }

   static void*
   new_np_T(void* p) { return p ? ::new (p) T : new T; }

   static void*
   newArray_T(long size,
              void* p) { return p ? new (p) T[size] : new T[size]; }

   static void*
   newArray_p_T(long size,
                void* p) { return p ? new (p) T[size] : ::new T[size]; }

   static void*
   newArray_np_T(long size,
                 void* p) { return p ? ::new (p) T[size] : new T[size]; }

   static void
   delete_T(void* p) { delete (T*) p; }

   static void
   deleteArray_T(void* p) { delete[] (T*) p; }

   static void
   destruct_T(void* p) { ((T*) p)->~T(); }

   NewDelFunctionsT() {
      fNew = new_T;
      fNewArray = newArray_T;
      fDelete = delete_T;
      fDeleteArray = deleteArray_T;
      fDestructor = destruct_T;
   }


};


} // namespace reflex

#endif
