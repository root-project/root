class Z {
public:
   static long GimeAddressPtr( void* obj ) {
      return reinterpret_cast< long >( obj );
   }

   static long GimeAddressPtrRef( void*& obj ) {
      return reinterpret_cast< long >( obj );
   }

   static long GimeAddressPtrPtr( void** obj ) {
      return reinterpret_cast< long >( obj );
   }

   static long GimeAddressObject( TObject* obj ) {
      return reinterpret_cast< long >( obj );
   }
};
