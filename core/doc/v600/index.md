## Platform Support

Temporarily for version 6.00/00, ROOT has a reduced set of supported
platforms. Most notably Windows is not supported until at least 6.02.
6.00/00 supports only

-   Linux 32bit and 64 bit, i32 and x86-64 and x32 (see below). Likely
    also PPC but that is untested, please report.
-   MacOS on Intel CPUs.

More platforms are expected to be available later; the lack of support
stems from cling not being ported to these platforms yet.

Despite that, an additional platform as been added: the [x32
psAPI](https://sites.google.com/site/x32abi/), called linuxx32gcc. It is
a regular x86-64 ABI but with shorter pointers (4 bytes instead of 8).
This reduces the addressable memory per process to 4GB - but that is
usally sufficient. The advantages are reduced memory consumption (due to
the smaller pointers) and increased performance compared to 32bit
applications due to the availability of the 64bit instructions. The
clang developers mailing list archive [contains a good
comparison.](http://clang-developers.42468.n3.nabble.com/Re-PATCH-add-x32-psABI-support-td4024297.html)

To build and run binaries compiled in x32, toolchain support is needed.
That is available in the in binutils (2.22), GCC (4.7), glibc (2.16),
Linux kernel (3.4) and even GDB (7.5). These versions are not available
in regular distributions yet (except for [this beta Gentoo
distro](http://dev.gentoo.org/~vapier/x32/stage3-amd64-x32-20120605.tar.xz)
built in x32); once they are, building and running x86-64 and x32
side-by-side will be possible.

## Core Libraries

### Cling vs CINT

Cling follows the C++ standard much more strictly than CINT. In
particular some code that used to run with CINT will either issue new
warnings or new compilations error. For example when CINT was parsing
Namespace::Symbol it would not only apply the C++ search rules but also
search in the outer scopes and for this example could actually return
::Symbol instead of (as Cling now does) issuing a compilation error.
Cling no longer supports refering to a class template instantiation of a
class template that has all default template parameter without the \<\>.
With:

``` {.cpp}
   template <typename T = int> class vec {};
```

With Cling (and any standard compliant compiler), using `*vec<>*` is
allowed (but `*vec*` is not).

### Cling not yet implemented features

In this release Cling does not support the following (but we intend to
re-introduce them soon) features:

-   Unloading of script and shared libraries
-   Discovery of symbols that are declared but not implemented being
    delayed until run-time. I.e.
    
``` {.cpp}
       void foo();
       void run() { dlopen("foo.so"); foo(); }
```

does not work in this release.

### TInterpreter

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

### rootcling

rootcling is the successor to rootcint and genreflex. It is used to
implement backward compatibility wrappers for both of them with the
following *backward incompatibilities.*

-   rootcling does not support multiline \#pragma without a line
    continuation as rootcint did (rootcint support \#pragma link with a
    line continutation only in ROOT v5.34/02 and above).
-   rootcling no longer re-\#defines the private and protected keywords
    to public. In particular this means that code compiled as part of
    the dictionary no longer has access to protected and private members
    of a class (except where allowed by the C++ standard).
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

### ACliC

ACLiC has the following backward incompatibilities:

-   Since rootcling no longer re-\#defines the private and protected
    keyword to public, the code compiled by ACLIC no longer has access
    to protected and private members of a class (except where allowed by
    the C++ standard).

### TUnixSystem

-   Simplify `Setenv` coding.
-   Implement `Unsetenv` using the system function `unsetenv`.

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

