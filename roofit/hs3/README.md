## RooFitHS3 Library
_Contains facilities to serialize and deserialize RooWorkspaces to and from JSON and YML._
#### Note: This library is still at an experimental stage.

### Purpose

When using `RooFit`, statistical models can be conveniently handled and
stored as a `RooWorkspace`. However, for the sake of interoperability
with other statistical frameworks, and also ease of manipulation, it
may be useful to store statistical models in text form. This library
sets out to achieve exactly that, exporting to and importing from JSON
and YML.

### Backend

The default backend for this is the `nlohmann` JSON implementation,
which ships with ROOT as a builtin dependency and will import from and
export to JSON. Alternatively, the RapidYAML (`RYML`) implementation
can be used to also import from and export to YML. This implementation
can be selected at compile time with the `cmake` flag
`roofit_hs3_ryml`.

### Usage

The main class providing import from and export to JSON and YML is the
`RooJSONFactoryWSTool`. For basic usage, please refer to the
[class documentation](https://root.cern.ch/doc/master/RooJSONFactoryWSTool_8h.html).

### Open-world philosophy

One of the most challenging aspects of providing serialization and
deserialization for `RooFit` is the fact that `RooFit` follows an
"open-world" philosophy with respect to the functions and pdfs it can
handle. Over the years, `RooFit` has also accumulated a significant
number of different pre-implemented functions and pdfs. What is more,
you can easily create your own `RooFit` function by inheriting from
`RooAbsReal` or your own `RooFit` pdf by inheriting from
`RooAbsPdf`. This means that feature-complete serialization and
deserialization to and from JSON and YML will probably never be fully
achieved. However, this may not impede your usage of this library, as
it was written in such a way as to allow users (that is, people like
you) to easily add missing importers and exporters for existing
`RooFit` classes as well as custom implementations you might be using.

### Native and proxy-based importers and exporters

`RooFitHS3` allows to different types of importers and exporters:
*Native* implementations, and *proxy-based* ones.  If for a certain
class several implementations are provided, the native
implementation(s) take precedence.

### Writing your own importers and exporters: Proxy-based

Proxy-based implementations can be added very easily and without
actually writing any `C++` code -- you only need to add a short item
to a list in a `JSON` file, namely the
[export keys](https://github.com/root-project/root/blob/master/etc/RooFitHS3_wsexportkeys.json)
for an exporter, or the
[factory expressions](https://github.com/root-project/root/blob/master/etc/RooFitHS3_wsfactoryexpressions.json)
for an importer.

This works in the following way: Every `RooFit` class performs
dependency tracking via proxies, which have names. This can be
exploited to perform the mapping of proxy names to `json` keys upon
export. In the other direction, the `RooWorkspace` has a factory
interface that allows to call any constructor via a string
interface. Hence:
 - If a `RooFit` class has no other members aside from proxies, it can
   be exported using a set of `export keys`.
 - If all relevant members to a `RooFit` class are passed as
   constructor arguments, it can be imported using a `factory
   expression`.

For the importer, an entry in the
[factory expressions](https://github.com/root-project/root/blob/master/etc/RooFitHS3_wsfactoryexpressions.json)
needs to be added as follows:

```json
    "<json-key>": {
        "class": "<C++ class name>",
        "arguments": [
            "<json-key of constructor argument #1>",
            "<json-key of constructor argument #2>",
             ...
        ]
    }
```

Similarly, for the exporter, an entry in the
[export keys](https://github.com/root-project/root/blob/master/etc/RooFitHS3_wsexportkeys.json)
needs to be added as follows:

```json
    "<C++ class name>": {
        "type": "<json-key>",
        "proxies": {
            "<name of proxy>": "<json-key of this element>",
            "<name of proxy>": "<json-key of this element>",
            ...
        }
    }
```


If you don't want to edit the central `json` files containing the
factory expressions or export keys, you can also put your custom
export keys or factory expressions into a different json file and load
that using `RooJSONFactoryWSTool::loadExportKeys(const std::string
&fname)` and `RooJSONFactoryWSTool::loadFactoryExpressions(const
std::string &fname)`.

If either the importer or the exporter cannot be created with factory
expressions and export keys, you can instead write a custom `C++`
class to perform the import and export for you.

### Writing your own importers and exporters: Custom `C++` code

In order to implement your own importer or exporter, you can inherit
from the corresponding base classes `RooJSONFactoryWSTool::Importer`
or `RooJSONFactoryWSTool::Exporter`, respectively. You can find
[simple examples](https://github.com/root-project/root/blob/master/roofit/hs3/src/JSONFactories_RooFitCore.cxx)
as well as
[more complicated ones](https://github.com/root-project/root/blob/master/roofit/hs3/src/JSONFactories_HistFactory.cxx)
in `ROOT`.

Any importer should take the following form:

```C++
    class MyClassFactory : public RooJSONFactoryWSTool::Importer {
    public:
       bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
          tool->workspace()->import(theobj, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
          return true;
       }
    };
```

If the class you are trying to import inherits from `RooAbsPdf` rather
than from `RooAbsReal`, you should define `importPdf` instead of
`importFunction`, with the same signature.

Once your importer implementation exists, you need to register it with the tool using a line like the following:

```C++
    RooJSONFactoryWSTool::registerImporter("<json key>", new MyClassFactory(), true);
```

As there can be several importers for the same `json` key, the last
(boolean) argument determines whether your new importer should be
added at the top of the priority list (`true`) or at the bottom
(`false`). If the import fails, other importers are attempted.

The implementation of an exporter works in a very similar fashion:

```C++
    class MyClassStreamer : public RooJSONFactoryWSTool::Exporter {
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

```C++
    RooJSONFactoryWSTool::registerExporter(MyClass::Class(), new MyClassStreamer(), true);
```

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
of `ROOT`, either via `export keys` and `factory expressions`, or via
native `C++` classes, please consider contributing your implementation
to `ROOT` such that it can help other users also missing this importer
or exporter. You can fork `ROOT`, commit your changes, and file a pull
request to `ROOT` for your work to be included in the next release!

