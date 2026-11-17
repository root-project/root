\defgroup roofit_dev_docs_hs3 Implement JSON I/O for a RooFit class with RooFit HS3
\ingroup roofit_dev_docs
\date March 2023
\author Carsten Burgard
\brief Guide on how to add JSON IO for you RooFit classes.

## RooFitHS3 Library
_Contains facilities to serialize and deserialize RooWorkspaces to and from JSON._
#### Note: This library is still at an experimental stage.

### Purpose

When using `RooFit`, statistical models can be conveniently handled and
stored as a `RooWorkspace`. However, for the sake of interoperability
with other statistical frameworks, and also ease of manipulation, it
may be useful to store statistical models in text form. This library
sets out to achieve exactly that, exporting to and importing from JSON.

### Usage

The main class providing import from and export to JSON is the
RooJSONFactoryWSTool.

### Open-world philosophy

One of the most challenging aspects of providing serialization and
deserialization for `RooFit` is the fact that `RooFit` follows an
"open-world" philosophy with respect to the functions and pdfs it can
handle. Over the years, `RooFit` has also accumulated a significant
number of different pre-implemented functions and pdfs. What is more,
you can easily create your own `RooFit` function by inheriting from
`RooAbsReal` or your own `RooFit` pdf by inheriting from
`RooAbsPdf`. This means that feature-complete serialization and
deserialization to and from JSON will probably never be fully
achieved. However, this may not impede your usage of this library, as
it was written in such a way as to allow users (that is, people like
you) to easily add missing importers and exporters for existing
`RooFit` classes as well as custom implementations you might be using.

### Writing your own importers and exporters

In order to implement your own importer or exporter, you can inherit
from the corresponding base classes `RooFit::JSONIO::Importer`
or `RooFit::JSONIO::Exporter`, respectively. You can find
[simple examples](https://github.com/root-project/root/blob/master/roofit/hs3/src/JSONFactories_RooFitCore.cxx)
as well as
[more complicated ones](https://github.com/root-project/root/blob/master/roofit/hs3/src/JSONFactories_HistFactory.cxx)
in `ROOT`.

Any importer should take the following form:

``` {.cpp}
    using RooFit::Detail::JSONNode;

    class MyClassFactory : public RooFit::JSONIO::Importer {
    public:
       bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
       {
          std::string name(RooJSONFactoryWSTool::name(p));

          // check if the required keys are available in the JSON
          if (!p.has_child("<class member key #1>")) {
             RooJSONFactoryWSTool::error("missing key '<class member key #1>' of '" + name + "'");
          }
          if (!p.has_child("<class member key #2>")) {
             RooJSONFactoryWSTool::error("missing key '<class member key #2>' of '" + name + "'");
          }

          std::string member1(p["<class member key #1>"].val());
          int member2(p["<class member key #2>"].val_int());

          MyClass theobj(name.c_str(), member1, member2);
          tool->wsImport(theobj);
          return true;
       }
    };
```

The same `importArg` function is used for functions (inheriting from
`RooAbsReal`) and pdfs (inheriting from `RooAbsPdf`). If the importer
can't handle a given `JSON` node, it should return `false` so that the
next importer is tried. The functions `importFunction` and `importPdf`
of older `ROOT` versions are still called for backwards compatibility,
so existing importers that override them keep working, but new code
should override `importArg`.

Once your importer implementation exists, you need to register it with the tool using a line like the following:

``` {.cpp}
    RooFit::JSONIO::registerImporter<MyClassFactory>("<json key>", true);
```

This is a shorthand for passing an instance owned by a `std::unique_ptr`:

``` {.cpp}
    RooFit::JSONIO::registerImporter("<json key>", std::make_unique<MyClassFactory>(), true);
```

As there can be several importers for the same `json` key, the last
(boolean) argument determines whether your new importer should be
added at the top of the priority list (`true`) or at the bottom
(`false`). If the import fails, other importers are attempted.

The implementation of an exporter works in a very similar fashion:

``` {.cpp}
    class MyClassStreamer : public RooFit::JSONIO::Exporter {
    public:
       std::string const &key() const override
       {
          const static std::string keystring = "<json key>";
          return keystring;
       }
       bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
       {
          const MyClass *theObj = static_cast<const MyClass *>(func);
          elem["type"] << key();

          auto &member1 = elem["<class member key #1>"];
          member1 << theObj->getMember1();

          auto &member2 = elem["<class member key #2>"];
          member2 << theObj->getMember2();

          return true;
       }
    };
```

Also this needs to be registered with the tool

``` {.cpp}
    RooFit::JSONIO::registerExporter<MyClassStreamer>(MyClass::Class(), true);
```

or equivalently

``` {.cpp}
    RooFit::JSONIO::registerExporter(MyClass::Class(), std::make_unique<MyClassStreamer>(), true);
```

If the `TClass` of your class is not available where you register the
exporter, you can also pass the class name as a string instead of
`MyClass::Class()`. If `exportObject` returns `false`, the next
exporter is tried.

For more complicated cases where members are lists of elements, the
methods `is_seq()`, `set_seq()`, `is_map()`, `set_map()` and
ranged-based for-loops via `children()` might come in handy.

### Contributing

If you encounter a missing importer or exporter, please consider
filing a feature request via the
[issue tracker](https://github.com/root-project/root/issues/new?assignees=&labels=new+feature&template=feature_request.md).

If you don't want to wait for one of the dev's to pick up your request
an process it, you can use the above instructions to write your own.
If you wrote an importer or exporter for a `RooFit` class that is part
of `ROOT`, please consider contributing your implementation
to `ROOT` such that it can help other users also missing this importer
or exporter. You can fork `ROOT`, commit your changes, and file a pull
request to `ROOT` for your work to be included in the next release!

