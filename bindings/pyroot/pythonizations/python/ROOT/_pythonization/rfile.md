\defgroup Py_RFile RFile
\ingroup Python
\brief Modern, minimalistic interface for reading and writing ROOT files in Python.

# RFile: Modern ROOT File I/O in Python

`ROOT::Experimental::RFile` is ROOT's modern, lightweight file I/O interface.
Unlike `TFile`, which relies on global directory state (`gDirectory`) and implicit object registration,
`RFile` provides:
- **Value semantics and clear ownership**: Objects read from `RFile` are owned by the application and survive after the file is closed.
- **Hierarchical path-based access**: Use intuitive string paths like `"folder/subfolder/my_obj"`.
- **Pythonic ergonomics**: Full context manager support, automatic C++ type deduction, and Python iterable key exploration.

\note `RFile` is part of `ROOT.Experimental` and is currently in active development.

---

## Quickstart

~~~{.py}
import ROOT

# 1. Writing objects to an RFile
with ROOT.Experimental.RFile.Recreate("sample.root") as f:
    h = ROOT.TH1D("h1", "Sample Histogram", 50, -5, 5)
    h.FillRandom("gaus", 2000)

    # Write under root or nested paths
    f.Put("h1", h)
    f.Put("analysis/h1_copy", h)

# 2. Reading objects from an RFile
with ROOT.Experimental.RFile.Open("sample.root") as f:
    # Read by path
    hist = f.Get("h1")
    print(f"Retrieved {hist.GetName()} with {hist.GetEntries()} entries.")

    # Inspect all keys in the file
    for key in f.ListKeys():
        print(f"Key: {key.GetPath()} [{key.GetClassName()}], cycle {key.GetCycle()}")

# hist remains valid here, even though the file is now closed!
print(f"Hist entries outside context: {hist.GetEntries()}")
~~~

---

## Opening and Creating Files

Direct instantiation with `ROOT.Experimental.RFile()` is disallowed to avoid undefined states.
Instead, use one of the factory methods:

| Method | Description |
| :--- | :--- |
| `RFile.Open(filename)` | Opens an existing file in read-only mode. |
| `RFile.Recreate(filename)` | Creates a new file for writing, overwriting any existing file. |
| `RFile.Update(filename)` | Opens an existing file for reading and writing. |

### Context Manager (`with` statement)

Using `RFile` as a context manager ensures that the file is safely flushed and closed upon exiting the block, even if an exception occurs:

~~~{.py}
with ROOT.Experimental.RFile.Open("data.root") as f:
    obj = f.Get("my_tree")
    # File is automatically closed when leaving this block
~~~

To manually persist changes without closing, call `f.Flush()`.

---

## Writing Objects (`Put`)

### Automatic Type Deduction

`file.Put(name, obj)` automatically identifies the C++ class using `obj.__cpp_name__`:

~~~{.py}
with ROOT.Experimental.RFile.Recreate("output.root") as f:
    hist = ROOT.TH1F("h", "Hist", 10, 0, 10)
    f.Put("my_hist", hist)

    # Python str is automatically stored as std::string
    f.Put("metadata/description", "Run 2026 dataset")
~~~

### Explicit Type Specialization

For collection types (such as `std::vector`), specify the C++ template type via indexing:

~~~{.py}
with ROOT.Experimental.RFile.Recreate("output.root") as f:
    # Explicitly specify std::vector<int>
    f.Put["std::vector<int>"]("data/numbers", [1, 2, 3, 4, 5])
~~~

---

## Reading Objects (`Get`)

### Automatic Binding

Calling `file.Get(name)` reads the object with its concrete type automatically bound:

~~~{.py}
with ROOT.Experimental.RFile.Open("output.root") as f:
    obj = f.Get("my_hist")
    if obj is None:
        print("Object not found!")
    else:
        print(type(obj))  # <class 'ROOT.TH1F'>
~~~

### Explicit Type Specialization

You can also pass an explicit type to `Get`:

~~~{.py}
with ROOT.Experimental.RFile.Open("output.root") as f:
    hist = f.Get[ROOT.TH1]("my_hist")
~~~

\warning If the requested type does not match the stored object, `Get[T]` returns `None`.

---

## Inspecting and Listing Keys

### `GetKeyInfo(path)`

Queries metadata for a specific key without loading the object into memory:

~~~{.py}
with ROOT.Experimental.RFile.Open("output.root") as f:
    key = f.GetKeyInfo("my_hist")
    if key:
        print(f"Path: {key.GetPath()}")
        print(f"Class: {key.GetClassName()}")
        print(f"Cycle: {key.GetCycle()}")
~~~

### `ListKeys(basePath="", **kwargs)`

Returns a Python iterable over `RKeyInfo` items. You can customize the listing using keyword arguments:

~~~{.py}
with ROOT.Experimental.RFile.Open("output.root") as f:
    # 1. Iterate over all leaf objects recursively (default)
    for key in f.ListKeys():
        print(key.GetPath(), key.GetClassName())

    # 2. Include directory keys
    for key in f.ListKeys("", listDirs=True):
        print("Entry:", key.GetPath())

    # 3. List only immediate children of a subfolder
    for key in f.ListKeys("folder", listRecursive=False):
        print("Child:", key.GetPath())

    # 4. List only directories under a path
    for key in f.ListKeys("analysis", listDirs=True, listObjects=False):
        print("Directory:", key.GetPath())
~~~
