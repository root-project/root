
## Platform Support

Temporarily for version 6.00/00, ROOT has a reduced set of supported
platforms. Most notably Windows is not supported until at least 6.02.
6.00/00 supports only

-   Linux 32 bit and 64 bit, i32 and x86-64 and x32 (see below).
-   OSX 64 bit on x86-64.

More platforms are expected to be available later; the lack of support
stems from Cling and Clang/LLVM not being ported to these platforms yet.

To aleviate the pain for Windows users who want to try ROOT 6 we provide [a recipe](http://root.cern.ch/drupal/content/running-root-6-unsupported-systems-using-cernvm) on how to run ROOT 6 in a VM on Windows.

Building ROOT also requires a C++11 compatible compiler, so one needs to either have installed gcc >= 4.8 or Clang >= 3.4. On most lecagy platforms these newer compilers are available via a special install. See [the build prerequisites](http://root.cern.ch/drupal/content/build-prerequisites) page.

Despite that, an additional platform as been added: the [x32
psAPI](https://sites.google.com/site/x32abi/), called linuxx32gcc. It is
a regular x86-64 ABI but with shorter pointers (4 bytes instead of 8).
This reduces the addressable memory per process to 4GB - but that is
usally sufficient. The advantages are reduced memory consumption (due to
the smaller pointers) and increased performance compared to 32 bit
applications due to the availability of the 64 bit instructions. The
Clang developers mailing list archive [contains a good
comparison](http://clang-developers.42468.n3.nabble.com/Re-PATCH-add-x32-psABI-support-td4024297.html).

To build and run binaries compiled in x32, toolchain support is needed.
That is available in the in binutils (2.22), GCC (4.8), glibc (2.16),
Linux kernel (3.4) and even GDB (7.5). These versions are not available
in regular distributions yet (except for [this beta Gentoo
distro](http://dev.gentoo.org/~vapier/x32/stage3-amd64-x32-20120605.tar.xz)
built in x32); once they are, building and running x86-64 and x32
side-by-side will be possible.

## Build System
ROOT 6.00/00 can be built either using the classic "./configure;make" method or using CMake.
The CMake system has been completed for this version and should be functionally equivalent
to the classic one. The [detailed instructions](http://root.cern.ch/drupal/content/building-root-cmake) are available.
In upcoming releases the classic method will slowly be deprecated in favor of the CMake one.

## Core Libraries

### Cling vs CINT

Cling follows the C++ standard much more strictly than CINT. In
particular some code that used to run with CINT will either issue new
warnings or new compilation errors. For example when CINT was parsing
Namespace::%Symbol it would not only apply the C++ search rules but also
search in the outer scopes and for this example could actually return
::%Symbol instead of (as Cling now does) issuing a compilation error.

#### Template class names
Cling no longer supports refering to a class template instantiation of a
class template that has all default template parameter without the \<\>.
With:

``` {.cpp}
   template <typename T = int> class templt {};
```

With Cling (and any standard compliant compiler), using `*templt<>*` is
allowed (but `*templt*` is not).

#### Namespace prefix of template parameters
Given `namespace N { class A; template <typename T> class B;}`, the name
`N::B<N::A>` is no longer "shortened" to `N::B<A>`. This affects the forward
and backward compatibility of files.

#### Implicit dynamic up-casts
CINT would perform automatic upcasts to derived classes under certain contexts:

``` {.cpp}
   TH1* h1 = hpx
   TH1F* h1f = h1;
```

Cling does not allow this anymore. We might add this feature later if demand exists ([ROOT-4802](https://sft.its.cern.ch/jira/browse/ROOT-4802)).

#### Using symbols that are only available at runtime: load libFoo; foo()
CINT was processing macros line by line; Cling compiles code.
When calling a function (or in general using a symbol) that is provided by a library loaded at runtime,
Cling will in some cases report an unresolved symbol:

``` {.cpp}
   #include "Event.h"
   void dynload() {
      gSystem->Load("libEvent");
      new Event();
   }
```

You will currently have to provide a rootmap file for libEvent (which also requires include
guards for Event.h). This might get fixed in a later version ([ROOT-4691](https://sft.its.cern.ch/jira/browse/ROOT-4691)).

#### Using identifiers that are only available at runtime: gROOT->LoadMacro("foo.h"); foo()
CINT was processing macros line by line; Cling compiles code.
During this compilation, Cling will not see identifiers provided by `gROOT->LoadMacro()`.
While this will covered by dynamic scopes, they are currently too limited to handle this.
Please `#include` the header instead.

### TInterpreter

### TInterpreter

`TInterpreter::GetCurrentMacroName()` has a slightly different behavior in ROOT
6 than in ROOT 5. In ROOT 5 it was a synonym for `__FILE__` (so please use
`__FILE__` to get the old value); in ROOT 6 it returns the currently executing
(.x) source file. For nested execution it returns the innermost one.

The following are no longer supported and are now only issuing error
messages:

-   Getgvp
-   Setgvp
-   SetRTLD\_NOW
-   SetRTLD\_LAZY

Use of Setgvp and Getgvp should be looked at carefully as they were use
to control the behavior of the multiplexing CINT wrappers that were both
wrapper the calls to operator new with and without placement and the
constructor (and similarly for operator delete and the destructor).
Cling does not support such a multiplexing wrapper and alternatively
interface must be used (See TClass::New and TClass::Destructor for
example).

The following interfaces are not yet available:

- DeleteVariable

They might be re-implemented in a later version.

### rootcling, rootcint and genreflex

rootcling is the successor to rootcint and it preserves its old commandline
arguments and provides more (see help). The executable rootcint still exists
but it is just a redirection to rootcling.
The following *backward incompatibilities* are present between rootcint in
version 5 and rootcling in version 6:

-   rootcling does not support multiline \#pragma without a line
    continuation as rootcint did (rootcint support \#pragma link with a
    line continutation only in ROOT v5.34/02 and above).
-   rootcling no longer re-\#defines the private and protected keywords
    to public. In particular this means that code compiled as part of
    the dictionary no longer has access to protected and private members
    of a class (except where allowed by the C++ standard).
    In particular, this means that the code used for read rules (\#pragma read)
    can no longer access private member unless they have been mentioned as a
    target of the rule.
-   rootcling no longer considers a friend declaration to be a
    declaration for the friended function. In particular this means that
    rootcling may now issue:

``` {.cpp}
       Error: in this version of ROOT, the option '!' used in a linkdef file
              implies the actual existence of customized operators.
              The following declaration is now required:
          TBuffer &operator<<(TBuffer &,const THit *);
```

   if the operator `<<` declaration is missing.

To steer the parsing done during the execution of rootcling, a new
macro: *\_\_ROOTCLING\_\_* is now defined during the parsing. The macros
*\_\_CINT\_\_* and *\_\_MAKECINT\_\_* are defined only when looking for
`#pragma` statement.

The genreflex executable is still available, it preserves the same command
line options as in version 5 and provides new ones. Old selection XML files
are transparently usable by the new implementation. On the other hand some
of the new functionalities can be hardly backported to version 5. See
genreflex help for all the details about commandline switches and selection
files. It is important to note that it is not anymore a python script
interfaced to GCCXML but rather a wrapper around rootcling.

In order to specify the classes selection, three methods are available:

1)   Selection XML file. This file can be read by both genreflex and rootcling.
2)   LinkDef file. This file can be read by rootcling.
3)   Selection namespace (also called *dictselection*). Available both for
     rootcling and genreflex. See the documentation of the
     `ROOT::Meta::Selection` namespace for all the details.

*Warning*

It is important to note that during the parsing of the header files,
rootcint no longer defines *\_\_CINT\_\_* and genreflex no longer defines
*\_\_GCCXML\_\_*.  This means in particular that data members that were
made transient by hiding them from rootcint or genreflex now *must* be
*explicitly* marked as transient.  For rootcint or rootcling use:

``` {.cpp}
   sometype datamember; //! The exclamation mark signals the transientness
```

and for genreflex

``` {.cpp}
   <class name="edm::Particle" >
      <field name="m_calculated" transient="true" />
   </class>
```

or

``` {.cpp}
   <class name="edm::Particle" >
      <field name="m_calculated" persistent="false" />
   </class>

```

This new set of executables are responsible also for the generation of pcm
files. These are crucial for the functioning of ROOT and must reside in the
same directory of the libraries which contain the compiled dictionaries.

### rlibmap

The tools used to generate rootmap files are rootcling and genreflex. The
rlibmap tool is not present any more in ROOT starting from release 6.00.00.

### Rootmap files

To enhance the set of functionalities offered by ROOT and its new interpreter,
the format of the rootmaps evolved. Rootmap in the old format cannot be
produced anymore but only read. The new rootmaps can be still be concatenated.
A rootmap file now contains:

-   One (or more) section for forward declarations. These are real C++
    forward declarations of templates and namespaces. This is needed for Cling
    to be able to parse templates' instantiations and for some autoloading
    functionalities.
-   One (or more) libraries sections. These sections describe the ensamble of
    the autoload keys related to one or more shared libraries. An autoload key
    can be a class name, a namespace name, a typedef or alias or a header file name.
-   Single line comments, which start with a "#" character.

At ROOT startup, a check is performed on autoload keys. If the same key (which is not a template instantiation) refers to two different libraries (or sets of libraries) a warning is issued.
A typical Rootmap file look like:
``` {.cpp}
{ decls }
fwd declaration 1;
fwd declaration 2;
[...]
fwd declaration N;

[ libraryName1 libraryName2 ... ]
class className1
class className2
...
typedef typedefName1
typedef typedefName2
...
header headerName1
header headerName2
...

```


### TROOT

The list returned by `GetListOfTypes` is no longer filled when the dictionary
are loaded but instead are filled on demand, when the user explicitly (directly
or indirectly) request each typedef.  In particular this means that

``` {.cpp}
   gROOT->GetListOfTypes()->ls(); // or Print()
```

no longer prints the list of all available typedef but instead list only the
typedefs that have been previously accessed throught the list (plus the builtins
types).

### ACliC

ACLiC has the following backward incompatibilities:

-   Since rootcling no longer re-\#defines the private and protected
    keyword to public, the code compiled by ACLIC no longer has access
    to protected and private members of a class (except where allowed by
    the C++ standard).

### Collection

New collection `TListOfTypes` that implements on demand creation
of the `TDataType` describing a typedef.

### TUnixSystem

-   Simplify `Setenv` coding.
-   Implement `Unsetenv` using the system function `unsetenv`.

### TMacOSXSystem

-   The file descriptors' management improved/fixed.

### TColor

-   5 new predefined palettes with 255 colors are available vis
    `gStyle->SetPalette(n)`:

    -   n = 51 a Deep Sea palette is used.
    -   n = 52 a Grey Scale palette is used.
    -   n = 53 a Dark Body Radiator palette is used.
    -   n = 54 a two-color hue palette palette is used.(dark blue
        through neutral gray to bright yellow)
    -   n = 55 a Rain Bow palette is used.

    ![New 255 colors palettes](pal255.png)

-   Add the optional parameter "alpha" to `TColor::SetPalette` and
    `TStyle::SetPalette`. The default value is 1. (opaque palette). Any
    value between 0. and 1 define the level of transparency. 0. being
    fully transparent and 1. fully opaque.

-   In `SaveAS` implement the possibility to save an object as C code
    independant from ROOT. It is enough to save the object in a file
    with the extension ".cc". The code as to save each objet should be
    implement in each `SavePrimitive` function like in `TF1`.

-   In `SetPalette` predefined palettes were redefined even if it was
    not necessary.

### TAttText

-   Mnemonic constants are available:

    ``` {.cpp}
    kHAlignLeft   = 10, kHAlignCenter = 20, kHAlignRight = 30,
    kVAlignBottom = 1,  kVAlignCenter = 2,  kVAlignTop   = 3
    ```

    They allow to write:

    ``` {.cpp}
    object->SetTextAlign(kHAlignLeft+kVAlignTop);
    ```

-   New method `SetTextColorAlpha(ci, alpha)` to set the color index
    `ci` with a transparency percentage.

### TAttFill

-   Provide symbolic values for different styles.
-   New method `SetFillColorAlpha(ci, alpha)` to set the color index
    `ci` with a transparency percentage.

### TAttMarker

-   New method `SetMarkerColorAlpha(ci, alpha)` to set the color index
    `ci` with a transparency percentage.

### TAttLine

-   New method `SetLineColorAlpha(ci, alpha)` to set the color index
    `ci` with a transparency percentage.

### TAttAxis

-   `SetLabelColor` and `SetLabelColor` have an optional parameter
    `alpha` allowing to change the transparency of the axis labels and
     axis body.
