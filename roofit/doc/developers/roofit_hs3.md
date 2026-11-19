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

### Native and proxy-based importers and exporters

`RooFitHS3` allows to different types of importers and exporters:
*Native* implementations, and *proxy-based* ones.  If for a certain
class several implementations are provided, the native
implementation(s) take precedence.

### Writing your own importers and exporters: Proxy-based

Proxy-based implementations can be added very easily and without
actually writing any `C++` logic -- you only need to add a short item
to a `JSON` catalog, namely the
[export keys](https://github.com/root-project/root/blob/master/roofit/hs3/src/RooFitHS3_wsexportkeys.cxx)
for an exporter, or the
[factory expressions](https://github.com/root-project/root/blob/master/roofit/hs3/src/RooFitHS3_wsfactoryexpressions.cxx)
for an importer. The built-in catalogs are `JSON` documents embedded as
raw string literals in these two source files. They are compiled into
the `RooFitHS3` library and loaded automatically when the library is
loaded, so there are no separate `JSON` files to install or find at
runtime.

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
[factory expressions](https://github.com/root-project/root/blob/master/roofit/hs3/src/RooFitHS3_wsfactoryexpressions.cxx)
needs to be added as follows:

``` {.json}
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
[export keys](https://github.com/root-project/root/blob/master/roofit/hs3/src/RooFitHS3_wsexportkeys.cxx)
needs to be added as follows:

``` {.json}
    "<C++ class name>": {
        "type": "<json-key>",
        "proxies": {
            "<name of proxy>": "<json-key of this element>",
            "<name of proxy>": "<json-key of this element>",
            ...
        }
    }
```


To contribute a catalog entry to `ROOT`, edit the embedded catalog in
`roofit/hs3/src/RooFitHS3_wsexportkeys.cxx` or
`roofit/hs3/src/RooFitHS3_wsfactoryexpressions.cxx` and rebuild `ROOT`.
If you don't want to change `ROOT` itself, you can put your custom
export keys or factory expressions into your own `JSON` file (with the
same format as the embedded catalogs) and add them to the built-in ones
at runtime using `RooFit::JSONIO::loadExportKeys(const std::string
&fname)` and `RooFit::JSONIO::loadFactoryExpressions(const
std::string &fname)`. Both functions also have an overload that reads
from a `std::istream`. You can inspect the currently known entries with
`RooFit::JSONIO::printExportKeys()` and
`RooFit::JSONIO::printFactoryExpressions()`.

If either the importer or the exporter cannot be created with factory
expressions and export keys, you can instead write a custom `C++`
class to perform the import and export for you.

### Writing your own importers and exporters: Custom `C++` code

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
exporter is tried, and finally the export keys.

For more complicated cases where members are lists of elements, the
methods `is_seq()`, `set_seq()`, `is_map()`, `set_map()` and
ranged-based for-loops via `children()` might come in handy.

### Conformance testing

The HS3 conformance test suite checks that RooFit imports and evaluates a
set of frozen HS3 files like the reference implementation. It consists of
two repositories: the fixtures in
[HS3TestSuite](https://github.com/hep-statistics-serialization-standard/HS3TestSuite)
and the Python runner in
[HS3TestSuiteCode](https://github.com/hep-statistics-serialization-standard/HS3TestSuiteCode).

The suite is enabled with the CMake option `-Dtest_roofit_hs3testsuite=ON`,
which requires `-Dtesting=ON`, `-Dpyroot=ON` and the `jsonschema` Python
package. Both repositories are fetched at configuration time, so the option
requires network access; alternatively, point
`FETCHCONTENT_SOURCE_DIR_HS3TESTSUITE` and
`FETCHCONTENT_SOURCE_DIR_HS3TESTSUITECODE` to local checkouts. The option is
enabled in the ROOT CI. Run the suite with

``` {.sh}
    ctest -R roofit-hs3-hs3-suite --output-on-failure
```

Each check of each fixture is reported as a separate test case by
`roofit/hs3/test/test_hs3_suite.py`, which uses
`roofit/hs3/test/hs3testsuite_roofit_backend.py` to run the checks with
RooFit.

Both repositories are pinned to explicit commits in
`roofit/hs3/test/CMakeLists.txt`. Checks that are known to fail with the
current RooFit are listed in
`roofit/hs3/test/hs3testsuite_expected_failures.txt`, one
`<test_id>::<check_id>` per line, together with a one-line reason. They are
run as expected failures: if one of them starts to pass, or if an entry
matches no check, the test fails, so that the list can only shrink. When you
fix a conformance issue in RooFit, remove the corresponding entries in the
same pull request.

To bump the pinned commits, update `hs3testsuite_tag` and
`hs3testsuitecode_tag` in
`roofit/hs3/test/CMakeLists.txt`, re-run the suite and update the list of
expected failures accordingly. Make sure the runner and the fixtures are
compatible, and adapt the RooFit backend if the backend interface of the
runner has changed.

### Published HS3 documents

Experiments publish HS3 documents on [HEPData](https://www.hepdata.net),
where they are listed as resources of the type "HS3" (search for
`analysis:hs3`). These are the documents that users of RooFit actually have,
and most of them were written by older versions of ROOT, so they show whether
RooFit still reads what it once wrote. Being real analyses, they are also
realistic benchmarks.

The test `roofit/hs3/test/test_hs3_hepdata.py` runs these documents through
the runner of the HS3 conformance test suite, with the RooFit backend of the
suite and the fixtures in `roofit/hs3/test/hs3hepdata`. The fixtures have the
layout of the suite, but without the HS3 documents: those are downloaded from
HEPData when the test runs, and `manifest.json` pins them by their hashes.
`hepdata.json` lists where each document is found. For every document, the
test checks

- with the suite: that the document is imported with its pdf and data set,
  and that 2·(NLL − NLL(reference)) has the expected values at fixed points
  along the parameters of interest (`expected.json`);
- `round_trip`: that RooFit exports the imported workspace, and imports the
  exported document again;
- `fit_<data>`, only if the fits are enabled: the best-fit values, the
  uncertainties from Hesse and the profile likelihood intervals (Minos) of
  the parameters of interest and of the other free parameters without a
  constraint (typically the normalization factors of backgrounds), and a
  profile likelihood scan of the first parameters of interest
  (`expected_fit.json`). The likelihood includes the external constraints of
  the ModelConfig, e.g. the regularization of an unfolding. The HS3 test
  suite has no check kind for fits, so this file and its check are specific
  to this test. A model without parameters of interest, e.g. for a
  background-only fit, is scanned and fitted in its normalization factors.

Every fit of `expected_fit.json` says how it is run: `target` (the pdf and
the data set), `parameter_point` (the parameter point of the document that is
loaded before the fit: some documents keep the result of a conditional fit,
with most parameters fixed, in `default_values`), optionally `strategy` (the
Minuit strategy, 1 by default) and `fixed_parameters` (parameters that the fit
holds at a value, e.g. the signal strength of a background-only fit), and the
parameters. A fixture can have several fits of the same data set, e.g. a
signal-plus-background model with an additional background-only fit
(`fit_<data>_background_only`), whose normalization factors papers quote. The
generator takes these settings from `hs3hepdata/fit_settings.json`, by INSPIRE
id or test_id, each with its reason. `covariance_quality` is
the status of the covariance matrix when the fixture was generated; unless it
is 3 (full and accurate), the uncertainties from Hesse are not compared.

**The expected values are the ones that RooFit computed when the fixtures
were generated. For now, they are assumed to be correct: they have not been
compared with the published results of the analyses.** Every fixture says so
in its `metadata.json` and `expected_fit.json`. The test is therefore a
regression test, and the fits tell whether RooFit still finds the same
minimum, not whether that is the published one.

The test is enabled with the CMake option `-Dtest_roofit_hs3hepdata=ON`,
which requires `-Dtesting=ON`, `-Dpyroot=ON` and the Python packages
`jsonschema` and `hepdata-cli`. The latter is the client of HEPData, through
which the test accesses it (HEPData refuses requests of other scripted
clients). The runner of the suite is fetched at configuration time, like for
the conformance test suite. **The test is not enabled in the ROOT CI and
should not be**: HEPData must not be queried on every build. Run it by hand,
for example before a release or after a change to the import of older
documents:

``` {.sh}
    ctest -R roofit-hs3-hs3-hepdata --output-on-failure
```

The test needs network access when it runs and downloads a few hundred MB.
Everything it downloads is kept in the directory given by the CMake variable
`HS3HEPDATA_CACHE` (by default in the build directory) and is not downloaded
again. The fits are enabled with `-DHS3HEPDATA_FITS=ON`. The test file can
also be run directly, and is configured with environment variables:

| Variable | Meaning |
|---|---|
| `HS3HEPDATA_CACHE` | the cache directory |
| `HS3TESTSUITECODE_ROOT` | a checkout of HS3TestSuiteCode |
| `HS3HEPDATA_FITS=1` | run the fits |
| `HS3HEPDATA_EVAL_BACKEND` | the evaluation backend of RooFit for the fits, e.g. `codegen` |
| `HS3HEPDATA_OFFLINE=1` | don't contact HEPData, and skip documents that are not in the cache |
| `HS3HEPDATA_SELECT` | a regular expression for the documents to test |

The time of every step (import, round trip, and for the fits: creating the
likelihood, the minimization, Hesse, Minos and the profile scans) is written
to `timings.json` in the cache directory, for use as a benchmark.

Checks that are known to fail are listed in
`roofit/hs3/test/hs3hepdata_expected_failures.txt`, one
`<test_id>::<check_id>` per line together with a one-line reason, in the same
way as for the conformance test suite: an expected failure that starts to
pass, or an entry that matches no check, fails the test.

The fixtures are generated with
`roofit/hs3/test/hs3hepdata_generate_fixtures.py`. It adds fixtures for
documents that are new on HEPData and keeps the existing ones, unless it is
told to overwrite them. The test fails if HEPData lists HS3 documents without
fixtures, or if a document has changed, so that neither goes unnoticed.
A fit that fails, or does not finish within the time limit of the generator,
gets no `expected_fit.json`. The fits are independent of each other, so they
can be generated in parallel: the generator runs one in a process of its own
with the option `--fit-worker <test_id>`.

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

