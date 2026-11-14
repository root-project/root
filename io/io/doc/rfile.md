# RFile

RFile is a minimal and modern interface to ROOT files, design for ease of use from C++ and Python.

## Core API

The core API of RFile is represented by the methods:

- `Get`
- `Put`
- `Overwrite`
- `ListKeys`

## Notes about signatures and overloads

To strike a balance between expandability and convenience, methods such as Put and Overwrite accept some defaulted
arguments but don't use an option struct parameter as of today.

In case we need to add more parameters to those methods in the future, our plan is to create a new overload
that accepts an option struct and use that as the *only* optional parameter for the new method, instead of adding
more defaulted parameters.

## Interaction with objects stored in the file

All object interaction happens via string-based *paths*. A path is a filesystem-like sequence of slash-separated
names, such as: `my/dir/object`. Only the forward slash is accepted as a separator.

A valid object path must:
  - not be empty
  - not contain the character '.'
  - not contain ASCII control characters or whitespace characters (including tab or newline).
  - not contain more than RFile::kMaxPathNesting path fragments (i.e. more than RFile::kMaxPathNesting - 1 '/')
  - not end with a '/'

In addition, when *writing* an object to RFile, the character ';' is also banned (it is treated specially for reading,
as it separates the object name from its cycle).

### Paths uniqueness

By construction, RFile cannot store an object whose name contains a `/`, in the sense that it will always consider
the slash character as a directory separator.

This way, when reading back objects, there is no ambiguity as to which object is referred to by any specific path:
`"a/b/c"` will always mean "object `c` inside directory `b` inside directory `a`" and never, say, "object `b/c` inside
directory `a`.
Since the ROOT file format, and TFile in particular, doesn't forbid such ambiguity, RFile may be used to read a file
containing such ambiguous object. In this case, no object with a `/` in its name will be able to be read by RFile.

### No "RPath" in the signatures

An option to represent the object paths through a `ROOT::RPath` (or similar) bespoke class was considered, but
ultimately rejected at least for direct use in the RFile's API. Here is the rationale: a dedicated class would be
useful to strongly distinguish the strings representing paths from those representing other things (e.g. object titles)
in signatures, but only if this RPath class is not implicitly convertible to string. This would however come at the cost
of increased friction when using the RFile methods, which contrasts the main pillar of RFile (simplicity of use).
In practice, one would have to write:

```c++
auto myObj = file->Get<MyClass>(ROOT::RPath("my/object"));
// instead of
auto myObj = file->Get<MyClass>("my/object");
```

and in Python:

```python
myObj = file.Get(ROOT.RPath("my/object"))
```

which adds quite a bit of noise to this simple operations. The API design is such that there should be no ambiguity
between object paths and other strings (currently only titles).

That said, another benefit that RPath would provide is functionality for convenient path manipulation, most notably
pushing/popping of path fragments. This is a useful thing to have and we foresee that we may implement it in the future
in the form of either an RPath class that is implicitly convertible to string (and, crucially, does not appear in the
RFile methods' signature) or a set of free/static functions.

An example idea might be:
```c++
auto path = ROOT::RPath("my");
path = path.Push("dir");
file->Put(path.Push("obj1"), myObj1); // stored in my/dir/obj1
file->Put(path.Push("obj2"), myObj2); // stored in my/dir/obj2
path = path.Pop();
file->Put(path.Push("obj3"), myObj3); // stored in my/obj3
```

Another might be:
```c++
std::string path = "my";
auto dirPath = RFile::Child(path, "dir");
file->Put(RFile::Child(dirPath, "obj1"), myObj1); // stored in my/dir/obj1
file->Put(RFile::Child(dirPath, "obj2"), myObj2); // stored in my/dir/obj2
dirPath = RFile::Parent(dirPath);
file->Put(RFile::Child(dirPath, "obj3"), myObj3); // stored in my/obj3
```

## Directories

RFile stores objects in directories internally, however it does not contain a "directory object" in its API.
`RFile::Get<T>("dir")` where `dir` is a directory will return `nullptr` for any `T`.
