To see the problem do the following:

   rootcint -f baddict.C -DBAD mycl.C LinkDef.h

then build the library, for example

   KCC --no_exceptions -o myclbad.so -I$ROOTSYS/include  baddict.C -DBAD

Then in root:

   root [] .L myclbad.so
   root [] .typedef MyClass<Toy>::value_type
   List of typedefs
   myclbad.so        1 typedef MyClass<Toy>::value_type MyClass<Toy>::value_type
   root [] gROOT->GetClass("MyClass<Toy>::value_type")
   (const class TClass*)0x0

The correct behavior can be demonstrated as:

   rootcint -f gooddict.C mycl.C LinkDef.h
   KCC --no_exceptions -o myclgood.so -I$ROOTSYS/include  gooddict.C 

   root [] .L myclgood.so
   root [] .typedef MyClass<Toy>::value_type
   List of typedefs
   myclgood.so       1 typedef ConstLink<Toy> MyClass<Toy>::value_type
   root [] gROOT->GetClass("MyClass<Toy>::value_type");
   (const class TClass*)0x87829a8

