/* ensure that the shadow for Comp overrides ia() or we'll have an
   ambiguous ICh::ia() vs Obj::ia().
   From Jean-François Bastien.
 */

namespace VirtFuncOverrider {
class IA {
public:
   virtual ~IA() {}

   virtual void ia() = 0;
};

class Obj: public virtual IA {
public:
   virtual ~Obj() {}

   virtual void
   ia() {}

   virtual void
   obj() {}

};

class ICh: public virtual IA {
public:
   virtual ~ICh() {}

   virtual void
   ia() {}

   virtual void
   iCh() {}

};

class Comp: public Obj,
   public virtual ICh {
public:
   virtual ~Comp() {}

   virtual void
   ia() {}

   virtual void
   comp() {}

};

} // namespace VirtFuncOverrider
