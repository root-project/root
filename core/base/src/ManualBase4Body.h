#include "TInterpreter.h"

static int  G__ManualBase4_101_0_97(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   // Wrapper function for TDirectory::WriteObject
   // We need to emulate:
   //    return WriteObjectAny(obj,TClass::GetClass(typeid(T)),name,option);

   // Here find the class name 
   ClassInfo_t* ti = gInterpreter->ClassInfo_Factory( &libp->para[0] );

   switch(libp->paran) {
   case 3:
      G__letint(result7,105,(long)((TDirectory*)(G__getstructoffset()))->WriteObjectAny((const void*)G__int(libp->para[0]),gInterpreter->ClassInfo_FullName( ti ),(const char*)G__int(libp->para[1])
,(Option_t*)G__int(libp->para[2])));
      break;
   case 2:
      G__letint(result7,105,(long)((TDirectory*)(G__getstructoffset()))->WriteObjectAny((const void*)G__int(libp->para[0]),gInterpreter->ClassInfo_FullName( ti ),(const char*)G__int(libp->para[1])));
      break;
   }
   gInterpreter->ClassInfo_Delete( ti );
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ManualBase4_101_0_98(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash) {
   // We need to emulate:
   //     ptr = (T*)GetObjectChecked(namecycle,TClass::GetClass(typeid(T)));

   // Here find the class name 
   ClassInfo_t* ti = gInterpreter->ClassInfo_Factory( &libp->para[1] );

   G__setnull(result7);
   TDirectory *directory = ((TDirectory*)(G__getstructoffset()));
   const char* namecycle = (const char*)G__int(libp->para[0]);
   void *ptr = directory->GetObjectChecked( namecycle, gInterpreter->ClassInfo_FullName( ti ) );
   void **ptrarg;
   if ( libp->para[1].ref ) {
      ptrarg = (void**)libp->para[1].ref;
   } else {
      ptrarg = (void**)(&G__Mlong(libp->para[1]));
   }
   *ptrarg = ptr;

   gInterpreter->ClassInfo_Delete( ti );
   return(1 || funcname || hash || result7 || libp) ;
}

/* Setting up global function */
static int G__ManualBase4__0_211(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
   // We need to emulate: template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj)

   // Here find the class name 
   ClassInfo_t* ti = gInterpreter->ClassInfo_Factory( &libp->para[1] );

   TBuffer & buf( *(TBuffer*) libp->para[0].ref );

   TClass *cl = TBuffer::GetClass(gInterpreter->ClassInfo_FullName( ti ));
   void * obj = buf.ReadObjectAny(cl);

   void **ptr = libp->para[1].ref ? (void**) libp->para[1].ref : (void**) (&G__Mlong(libp->para[1])) ;

   *ptr = obj;

   result7->ref = (long) (&buf);
   result7->obj.i = (long) (&buf);

   gInterpreter->ClassInfo_Delete( ti );
   return(1 || funcname || hash || result7 || libp) ;
}

static int G__ManualBase4__0_212(G__value* result7, G__CONST char* funcname, struct G__param* libp, int hash)
{
   // We need to emulate template <class Tmpl> TBuffer &operator<<(TBuffer &buf, const Tmpl *obj)

   // Here find the class name 
   ClassInfo_t* ti = gInterpreter->ClassInfo_Factory( &libp->para[1] );

   TBuffer & buf( *(TBuffer*) libp->para[0].ref );

   const void *obj = (const void*) G__int(libp->para[1]);

   TClass *cl = (obj) ? TBuffer::GetClass(gInterpreter->ClassInfo_FullName( ti )) : 0;
   buf.WriteObjectAny(obj, cl);

   result7->ref = (long) (&buf);
   result7->obj.i = (long) (&buf);

   gInterpreter->ClassInfo_Delete( ti );
   return(1 || funcname || hash || result7 || libp) ;
}

