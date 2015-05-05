# JavaScript ROOT

The JSROOT project intends to implement ROOT graphics for web browsers.
Reading of binary ROOT files is supported.
It is the successor of the JSRootIO project.


## Installing JSROOT

The actual version of JSROOT can be found in ROOT repository, etc/http/ subfolder.
All necessary files are located there. Just copy them on any web server or use them directly from the file system.
The latest version of JSROOT can also be found online on <http://root.cern.ch/js/jsroot.html> or <http://web-docs.gsi.de/~linev/js/>.


## Reading ROOT files in JSROOT

[The main page](https://root.cern.ch/js/3.5/) of the JSROOT project provides the possibility to interactively open ROOT files and draw objects like histogram or canvas.

The following parameters can be specified in the URL string:

- file, files - name of the file(s), which will be automatically open with page loading
- item, items - name of the item(s) to be displayed
- opt, opts - drawing option(s) for the item(s)
- layout - can be 'simple', 'collapsible', 'tabs' or 'gridNxM' where N and M integer values
- nobrowser - do not display file browser
- load - name of JavaScript to load
- optimize - drawing optimization 0:off, 1:only large histograms (default), 2:always
- interactive - enable/disable interactive functions 0-disable all, 1-enable all
- noselect - hide file-selection part in the browser (only when file name is specified)
- mathjax - use MathJax for latex output

When specifying `file`, `item` or `opt` parameters, one could provide array like `file=['file1.root','file2.root']`.
One could skip quotes when specifying elements names `item=[file1.root/hpx,file2.root/hpy]` or `opt=['',colz]`.

Examples:

- <https://root.cern.ch/js/3.5/index.htm?file=../files/hsimple.root&item=hpx;1>
- <https://root.cern.ch/js/3.5/index.htm?file=../files/hsimple.root&nobrowser&item=hpxpy;1&opt=colz>
- <https://root.cern.ch/js/3.5/index.htm?file=../files/hsimple.root&noselect&layout=grid2x2&item=hprof;1>

One can very easy integrate JSROOT graphic into other HTML pages using a __iframe__ tag:

<iframe width="600" height="500" src="https://root.cern.ch/js/3.5/index.htm?nobrowser&file=../files/hsimple.root&item=hpxpy;1&opt=colz">
</iframe>

In principle, one could open any ROOT file placed in the web, providing the full URL to it like:

<https://web-docs.gsi.de/~linev/js/3.4/?file=https://root.cern.ch/js/files/hsimple.root&item=hpx>

But one should be aware of [Cross-Origin Request blocking](https://developer.mozilla.org/en/http_access_control),
when the browser blocks requests to files from domains other than current web page.
To enable CORS on Apache web server, hosting ROOT files, one should add following lines to `.htaccess` file:

    <IfModule mod_headers.c>
      <FilesMatch "\.root">
         Header set Access-Control-Allow-Origin "*"
         Header set Access-Control-Allow-Headers "range"
         Header set Access-Control-Expose-Headers "content-range,content-length,accept-ranges"
         Header set Access-Control-Allow-Methods "HEAD,GET"
      </FilesMatch>
    </IfModule>


Other solution - copy all JSROOT files to the same location than where the data files are located.
In such case one could use the server with its default settings.

A simple case is to copy only the top index.htm file on the server and specify the full path to JSRootCore.js script like:

    ...
    <script type="text/javascript" src="https://root.cern.ch/js/3.5/scripts/JSRootCore.js?gui"></script>
    ...

In such case one can also specify a custom files list:

    ...
     <div id="simpleGUI" path="files/subdir" files="userfile1.root;subdir/usefile2.root">
       loading scripts ...
     </div>
    ...


## JSROOT with THttpServer

THttpServer provides http access to objects from running ROOT application.
JSROOT is used to implement the user interface in the web browsers.

The layout of the main page coming from THttpServer is similar to the file I/O one.
One could browse existing items and display them. A snapshot of running
server can be seen on the [demo page](https://root.cern.ch/js/3.5/httpserver.C/).

One could also specify similar URL parameters to configure the displayed items and drawing options.

It is also possible to display one single item from the THttpServer server like:

<https://root.cern.ch/js/3.5/httpserver.C/Files/job1.root/hpxpy/draw.htm?opt=colz>


##  Data monitoring with JSROOT

### Monitoring with http server

The best possibility to organize the monitoring of data from a running application
is to use THttpServer. In such case the client can always access the latest
changes and request only the items currently displayed in the browser.
To enable monitoring, one should activate the appropriate checkbox or
provide __monitoring__ parameter in the URL string like:

<https://root.cern.ch/js/3.5/httpserver.C/Files/job1.root/hprof/draw.htm?monitoring=1000>

The parameter value is the update interval in milliseconds.


### JSON file-based monitoring

Solid file-based monitoring (without integration of THttpServer into application) can be
implemented in JSON format. There is the TBufferJSON class, which is capable to potentially
convert any ROOT object (beside TTree) into JSON. Any ROOT application can use such class to
create JSON files for selected objects and write such files in a directory,
which can be accessed via web server. Then one can use JSROOT to read such files and display objects in a web browser.
There is a demonstration page showing such functionality:

<https://root.cern.ch/js/3.5/demo/demo.htm>

<iframe width="500" height="300" src="https://root.cern.ch/js/3.5/demo/demo.htm">
</iframe>

This demo page reads in cycle 20 json files and displays them.

If one has a web server which already provides such JSON file, one could specify the URL to this file like:

<https://root.cern.ch/js/3.5/demo/demo.htm?addr=../httpserver.C/Canvases/c1/root.json.gz>

Here the same problem with [Cross-Origin Request](https://developer.mozilla.org/en/http_access_control) can appear.
If the web server configuration cannot be changed, just copy JSROOT to the web server itself.


### Binary file-based monitoring (not recommended)

Theoretically, one could use binary ROOT files to implement monitoring.
With such approach, a ROOT-based application creates and regularly updates content of a ROOT file, which can be accessed via normal web server. From the browser side, JSROOT could regularly read the specified objects and update their drawings. But such solution has three major caveats.

First of all, one need to store the data of all objects, which only potentially could be displayed in the browser. In case of 10 objects it does not matter, but for 1000 or 100000 objects this will be a major performance penalty. With such big amount of data one will never achieve higher update rate.

The second problem is I/O. To read the first object from the ROOT file, one need to perform several (about 5) file-reading operations via http protocol.
There is no http file locking mechanism (at least not for standard web servers),
therefore there is no guarantee that the file content is not changed/replaced between consequent read operations. Therefore, one should expect frequent I/O failures while trying to monitor data from ROOT binary files. There is a workaround for the problem - one could load the file completely and exclude many partial I/O operations by this. To achieve this with JSROOT, one should add "+" sign at the end of the file name. Of course, it only could work for small files.

The third problem is the limitations of ROOT I/O in JavaScript. Although it tries to fully repeat logic of binary I/O with the streamer infos evaluation, the JavaScript ROOT I/O will never have 100% functionality of native ROOT. Especially, the custom streamers are a problem for JavaScript - one need to implement them once again and keep them synchronous with ROOT itself. And ROOT is full of custom streamers! Therefore it is just great feature that one can read binary files from a web browser, but one should never rely on the fact that such I/O works for all cases.
Let say that major classes like TH1 or TGraph or TCanvas will be supported, but one will never see full support of TTree or RooWorkspace in JavaScript.

If somebody still wants to use monitoring of data from ROOT files, could try link like:

<https://root.cern.ch/js/3.5/index.htm?nobrowser&file=../files/hsimple.root+&item=hpx;1&monitoring=2000>

In this particular case, the histogram is not changing.


## Stand-alone usage of JSROOT

Even without any server-side application, JSROOT provides nice ROOT-like graphics,
which could be used in arbitrary HTML pages.
There is [example page](https://root.cern.ch/js/3.5/demo/example.htm),
where a 2-D histogram is artificially generated and displayed.
Details about the JSROOT API can be found in the next chapters.


## JSROOT API

JSROOT consists of several libraries (.js files). They are all provided in the ROOT
repository and are available in the 'etc/http/scripts/' subfolder.
Only the central classes and functions will be documented here.

### Scripts loading

Before JSROOT can be used, all appropriate scripts should be loaded.
Any HTML pages where JSROOT is used should include the JSRootCore.js script.
The `<head>` section of the HTML page should have the following line:

    <script type="text/javascript" src="https://root.cern.ch/js/3.5/scripts/JSRootCore.js?gui"></script>

Here, the default location of JSROOT is specified. One could have a local copy on the file system or on a private web server. When JSROOT is used with THttpServer, the address looks like:

    <script type="text/javascript" src="http://your_root_server:8080/jsrootsys/scripts/JSRootCore.js?gui"></script>

In URL string with JSRootCore.js script one should specify which JSROOT functionality will be loaded:

    + '2d' normal drawing for 1D/2D objects
    + '3d' 3D drawing for 2D/3D histograms
    + 'io' binary file I/O
    + 'mathjax' loads MathJax and uses for latex output
    + 'gui' default gui for offline/online applications
    + 'load' name of user script(s) to load
    + 'onload' name of function to call when scripts loading completed


### Use of JSON

It is strongly recommended to use JSON when communicating with ROOT application.
THttpServer  provides a JSON representation for every registered object with an url address like:

    http://your_root_server:8080/Canvases/c1/root.json

Such JSON representation generated using the [TBufferJSON](http://root.cern.ch/root/html/TBufferJSON.html) class.

To access data from a remote web server, it is recommended to use the [XMLHttpRequest](http://en.wikipedia.org/wiki/XMLHttpRequest) class.
JSROOT provides a special method to create such class and properly handle it in different browsers.
For receiving JSON from a server one could use following code:

    var req = JSROOT.NewHttpRequest("http://your_root_server:8080/Canvases/c1/root.json", 'object', userCallback);
    req.send(null);

In the callback function, one gets JavaScript object (or null in case of failure)


### Objects drawing

After an object has been created, one can directly draw it. If somewhere in a HTML page there is a `<div>` element:

    ...
    <div id="drawing"></div>
    ...

One could use the JSROOT.draw function:

    JSROOT.draw("drawing", obj, "colz");

The first argument is the id of the HTML div element, where drawing will be performed. The second argument is the object to draw and the third one is the drawing option.
One is also able to update the drawing with a new version of the object:

    // after some interval request object again
    JSROOT.redraw("drawing", obj2, "colz");

The JSROOT.redraw function will call JSROOT.draw if the drawing was not performed before.


### File API

JSROOT defines the JSROOT.TFile class, which can be used to access binary ROOT files.

    var filename = "https://root.cern.ch/js/files/hsimple.root";
    var f = new JSROOT.TFile(filename, fileReadyCallback);

One should always remember that all I/O operations are asynchronous in JSROOT.
Therefore, callback functions are used to react when the I/O operation completed.
For example, reading an object from a file and displaying it will look like:

    new JSROOT.TFile(filename, function(file) {
       file.ReadObject("hpxpy;1", function(obj) {
          JSROOT.draw("drawing", obj, "colz");
       });
    });


## Links collection

Many different examples of JSROOT usage can be found on [links collection](https://root.cern.ch/js/3.5/demo/jslinks.htm) page
