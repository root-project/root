## Networking Libraries

A new class TS3WebFile has been introduced. The new class TS3WebFile is
an extension of TWebFile and belongs to the net module. The name
TS3WebFile reflects better the fact that this solution is intended to be
generic to several S3 servers and not limited to Amazon's, in addition
to the fact that it actually extends the capabilities of TWebFile.

Compared to the current support of S3 in ROOT (basically the class
TAS3File), the modifications include the improvements below:

-   add support for using HTTPS : you can use different schemas for
    specifying the underlying transport protocol to use "s3:",
    "s3http:", "s3https:" ["s3" uses HTTPS]. The current schema, namely
    "as3:", is supported for backwards compatibility.
-   extend support for other S3 service providers that do not offer the
    virtual hosting functionality (currently only Amazon offers this).
-   support the possibility of specifying user credentials on a per-file
    basis or for all S3 files via environment variables.
-   honor the "NOPROXY" option when specified in the constructor.
-   exploit the capability of the S3 file server to provide partial
    content responses to multi-range HTTP requests.

Here are some examples of usages from the end user perspective:

``` {.cpp}
       TFile* f = TFile::Open("s3://s3.amazonaws.com/mybucket/path/to/my/file", "AUTH=<accessKey>:<secretKey> NOPROXY")
       TFile* f = TFile::Open("s3://s3.amazonaws.com/mybucket/path/to/my/file")   // Uses environmental variables for retrieving credentials
```

Limitations:

-   we cannot efficiently detect that a S3 server is able to respond to
    multi-range HTTP GET requests. Some servers, such as Amazon's,
    respond to such kind of requests with the whole file contents. Other
    servers, such as Huawei's, respond with the exact partial content
    requested. Therefore, I added the possibility of configuring the
    behavior via the ROOT configuration file: the identity of the
    servers known to correctly support multi-range requests is
    configurable. If the server is known to support this feature, ROOT
    will send multi-range requests, otherwise it will issue multiple
    single-range GET requests, which is also the default behavior.
-   currently the virtual host syntax:
    "s3://mybucket.s3.amazonaws.com/path/to/my/file" is not supported
    but can be added if this is considered useful.

The TAS3File class will be removed and should not have been used
directly by users anyway as it was only accessed via the plugin manager
in TFile::Open().

### New HTTP Server package

A new HTTP Server package has been introduced. The idea behind such server is to provide direct access to the data from a running ROOT application. Any object can be streamed when requested and delivered to the browser.

##### Starting HTTP server

To start http server, at any time  create instance
of the **`THttpServer`** class like:

``` {.cpp}
serv = new THttpServer("http:8080");
```

This will start civetweb-based http server on port 8080.
Then, one should be able to open address "http://localhost:8080"
in any modern browser and browse objects created in application. By default, the server can access files, canvases and histograms via gROOT. All such objects can be displayed with JSRootIO graphics.

At any time one could register other objects with the command:

``` {.cpp}
    TGraph* gr = new TGraph(10);
    gr->SetName("gr1");
    serv->Register("graphs/subfolder", gr);
```

If the object content is changing in the application, like for example histograms being continuously filled, one could enable the monitoring flag in the browser, then the object view will be regularly updated.


