How to read and write custom classes in TTree
=============================================

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

The TTree can be seen as a collection of objects (branches), with a number of attributes (leaves). The name of the branch corresponds to the name of the instantianated object. The name of the leaves corresponds to the name of the attributes.

The attributes of 

The class that is present in the TTree is declared in the .hpp, and the methods defined in the .cpp. 

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
