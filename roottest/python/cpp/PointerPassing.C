class Z {
public:
   static Z* getZ(int idx) {
      static Z z[2];
      return &(z[idx]);
   }
   static unsigned long GimeAddressPtr( void* obj ) {
      return reinterpret_cast< unsigned long >( obj );
   }

   static unsigned long GimeAddressPtrRef( void*& obj ) {
      return reinterpret_cast< unsigned long >( obj );
   }

   static unsigned long SetAddressPtrRef( void*& obj ) {
      obj = (void*)0x1234;
      return 21;
   }

   static unsigned long GimeAddressPtrPtr( void** obj ) {
      return reinterpret_cast< unsigned long >( obj );
   }

   static unsigned long SetAddressPtrPtr( void** obj ) {
      (*(long**)obj) = (long*)0x4321;
      return 42;
   }

   static unsigned long GimeAddressObject( TObject* obj ) {
      return reinterpret_cast< unsigned long >( obj );
   }

   static bool checkAddressOfZ(Z*& pZ) {
      bool ret = (pZ == getZ(0));
      pZ = getZ(1);
      return ret;
   }
   int Grow = {};
};

class Z_ {
public:
   Z_& GimeZ_( Z_& z ) { return z; }

public:
   int myint;
};
