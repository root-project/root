// $Id: NewDelFunctions.h,v 1.5 2006/03/13 15:49:50 roiser Exp $

#ifndef ROOT_Reflex_NewDelFunctions
#define ROOT_Reflex_NewFelFunctions

/**
 * @file  NewDelFunctions.h
 */

namespace ROOT {

   namespace Reflex {
   
      typedef void* (*NewFunc_t)( void* );
      typedef void* (*NewArrFunc_t)( size_t size, void *arena );
      typedef void  (*DelFunc_t)( void* );
      typedef void  (*DelArrFunc_t)( void* );
      typedef void  (*DesFunc_t)( void* ); 
    
      struct RFLX_API NewDelFunctions {
         NewFunc_t    fNew;             //pointer to a function newing one object.
         NewArrFunc_t fNewArray;        //pointer to a function newing an array of objects.
         DelFunc_t    fDelete;          //pointer to a function deleting one object.
         DelArrFunc_t fDeleteArray;     //pointer to a function deleting an array of objects.
         DesFunc_t    fDestructor;      //pointer to a function call an object's destructor.
      };
    
      template <class T>  struct NewDelFunctionsT : public NewDelFunctions {
         static void* new_T(void* p) { return p ? new(p) T : new T; }
         static void* new_p_T(void* p) { return p ? new(p) T : ::new T; }
         static void* new_np_T(void* p) { return p ? ::new(p) T : new T; }
         static void* newArray_T(size_t size, void* p) { return p ? new (p) T[size] : new T[size]; }
         static void* newArray_p_T(size_t size, void* p) { return p ? new (p) T[size] : ::new T[size]; }
         static void* newArray_np_T(size_t size, void* p) { return p ? ::new (p) T[size] : new T[size]; }
         static void  delete_T(void *p) { delete (T*)p; }
         static void  deleteArray_T(void* p) { delete [] (T*)p; }
         static void  destruct_T(void* p) { ((T*)p)->~T(); }
         NewDelFunctionsT() {
            fNew         = new_T;
            fNewArray    = newArray_T;
            fDelete      = delete_T;
            fDeleteArray = deleteArray_T;
            fDestructor  = destruct_T;
         }      
      };


   } // namespace reflex
      
} // namespace ROOT

#endif
