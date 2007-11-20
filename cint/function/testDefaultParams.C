{
   // One object of type Base and one of type derived
   Base    *base    = new Base();
   Derived *derived = new Derived();
   
   // Another of type Base pointing to a Derived 
   Base    *base_der = derived;

   // Default parameter of the base funtion
   base->FunctionX();

   // Default parameter of the derived funtion
   derived->FunctionX();

   // Default parameter of the base funtion again!!!
   // Note: they are evaluated according to their static
   // type since in this case it's "Base *base_der"
   base_der->FunctionX();

   delete base;
   delete derived;
}
