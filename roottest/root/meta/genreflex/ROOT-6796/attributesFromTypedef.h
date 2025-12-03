// This is a reproducer for ROOT-6796

class RootType {};
namespace pool {
   class IClassHandler  {
   public:
   typedef RootType TypeH;
   };
} // End namespace pool
