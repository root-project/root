How to read and write custom classes in TTree
=============================================

This example shows how to use ROOT as a toolkit to write and read a TTree containing custom classes.

# Starting steps

1. Build the project, it requires ROOT

```
cmake -B build -S . 
cmake --build build 
```

2. Run the executable

```
./build/treeExample
```

3. Inspect the output rootfile, and the source code. 

# Files

This example shows how to use ROOT as a library in a C++ project. The C++ main function is defined in the file main.cpp. The code that writes the TTree onto disk is contained in the file writeTree.cxx. The files readTree.cxx and readTreeDF.cxx show two different ROOT interfaces to read a TTree. The files data2tree.* contain the code for the custom class that fills the TTree.

## Definition of custom class

The TTree can be seen as a collection of objects (branches), with a number of attributes (leaves). The name of the branch corresponds to the name of the instantiated object. The name of the leaves corresponds to the name of the attributes.

The class that is present in the TTree is declared in the file data2tree.hpp, and the methods defined in the file data2tree.cpp. 

To be able to read and write objects of a particular user-defined type, ROOT I/O needs to know some information about the class/struct, e.g. the class members and their types, offset of each data member, etc.  This information is contained in a ROOT dictionary; see [I/O of custom classes](https://root.cern/manual/io_custom_classes/#generating-dictionaries) for more information.

The linkdef file contains some instructions for ROOT, to specify which classes will require a dictionary:

```
#pragma link C++ class myFancyClass+;
```

If for example `std::vector` of such class is also used in the TTree as well, it should be notified to ROOT as another separated instruction:

```
#pragma link C++ class std::vector<myFancyClass>+;
```

# Links: 
* ROOT documentation: https://root.cern/manual/io_custom_classes/
