class Z {
public:
   static long GimeAddressPtr( void* obj ) {
      return reinterpret_cast< long >( obj );
   }

   static long GimeAddressPtrRef( void*& obj ) {
      return reinterpret_cast< long >( obj );
   }

   static long SetAddressPtrRef( void*& obj ) {
      obj = (void*)0x1234;
      return 21;
   }

   static long GimeAddressPtrPtr( void** obj ) {
      return reinterpret_cast< long >( obj );
   }

   static long SetAddressPtrPtr( void** obj ) {
      (*(long**)obj) = (long*)0x4321;
      return 42;
   }

   static long GimeAddressObject( TObject* obj ) {
      return reinterpret_cast< long >( obj );
   }
};

class Z_ {
public:
   Z_& GimeZ_( Z_& z ) { return z; }

public:
   int myint;
};
