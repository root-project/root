GOAL:

  Provide http interface to arbitrary ROOT application

USAGE:

  At any place of the code create http server:

     root [0] serv = new THttpServer

  By default, civetweb web server with port number 8080 will be started.
  It gets access to files, canvases and trees, registered in gROOT.
  One additionally could register other objects to the server:

     root [1] serv->Register("abc/fold1", hpx);
     root [2] serv->Register("abc/fold2", hpxpy);
     root [3] serv->Register("extra", c1);

  Once server running, just open in any browser page: http://yourhost:8080

  Example macro: $ROOTSYS/tutorials/http/httpserver.C


FAST CGI:

   Instead of running http server, one could use fast cgi interface
   to normal web server like Apache or lighttpd or any other.
   When creating server, one could specify:

     root [0] serv = new THttpServer("fastcgi:9000");

   This opens port 9000, which should be specified in web server configuration.
   For instance, lighttpd.conf file could contain path like this:

    fastcgi.server = (
     "/remote_scripts/" =>
      (( "host" => "192.168.1.10",
         "port" => 9000,
         "check-local" => "disable",
         "docroot" => "/"
     ))
    )

    In this case one should be able to access root application via address

       http://your_lighttpd_host/remote_scripts/root.cgi/

AUTHOR:

  Sergey Linev, S.Linev@gsi.de



CHANGES:

  January 2015
   - Provide exe.json request to execute arbitrary object method and return
     result in JSON format. Server should run in non-readonly mode

  Fall 2014
   - Implement gzip for result of any submitted requests, automatically done 
     when .gz extension is provided
   - Provide access to arbitrary data member of objects, registered to the server
   - Prevent data caching in the browser by setting no-cache header

  April 2014
  - In TCivetweb class support digest authentication method. User
    can specify auth_file and auth_domain parameters to protect
    access to the server
  - Fix error in FastCgi, now correctly works with Apache
  - Avoid direct usage of TASImage

  March 2014
  - Replace mongoose by civetweb due to more liberal MIT license.
    Works out of the box while civetweb version fully corresponds to
    previously used version of mongoose.
  - Introduce TBufferJSON class to store arbitrary ROOT object
    into JSON format. It is not one-to-one storage (like XML), but
    rather JS-like structures. For instance, all TCollections converted
    into JavaScript Array. Produced JS object is similar to JSRootIO.
  - Process get.json request, which returns object in JSON form.
    It can be used directly is script without special I/O of Bertrand.
  - Use get.json on browser side to simplify logic. No need for extra
    requests for streamer infos.
  - Process get.xml request, provide full object streaming via TBufferXML.
    It is complete object data, but with many custom-streamer data.
  - Significant redesign of I/O part of JSRootIO code. Main change -
    introduce JSROOTIO.TBuffer class which has similar functionality as
    original TBufferFile class. Eliminate many places with duplicated code
  - In JSRootIO avoid objects cloning when object referenced several times
  - Treat special cases (collection, arrays) in one place.
    This is major advantage, while any new classes should be implemented once.
  - Object representation, produced by JSRootIO is similar to
    objects, produced by TBufferJSON class. By this one can exchange
    I/O engine and use same JavaSctript graphic for display.
  - More clear functions to display different elements of the file.
    In the future functions should be fully separated from I/O part
    and organized in similar way as online part.
  - Eliminate usage of gFile pointer in the I/O part.
  - Provide TBufferJSON::JsonWriteMember method. It allows to stream any
    selected data member of the class. Supported are: basic data types,
    arrays of basic data types, TString, TArray classes. Also any object
    as data member can be streamed.
  - TRootSniffer do not creates sublevels for base classes
  - When streaming data member, TBufferJSON produces array with all dimensions
    only when fCompact==0. By default, THttpServer uses compact=1 for member
  - Support both get.json and root.json requests, they have similar meaning.

  January 2014
  - Make THttpServer::CreateEngine as factory method. One could
    create http, fastcgi and dabc engines to access data from server.
    Syntax allows to provide arbitrary arguments. Examples:
        THttpServer* serv = new THttpServer();
        serv->CreateEngine("http:8080");
        serv->CreateEngine("fastcgi:9000/none?top=MyApp");
        serv->CreateEngine("dabc:1237?top=MyApp");
        serv->CreateEngine("dabc:http:8090?top=MyApp");
        serv->CreateEngine("dabc:fastcgi:9010?top=MyApp");
  - Many engines can be created at once.
  - Provide TDabcEngine (in DABC framework).
  - Support additional options for mongoose and fastcgi servers
  - Port to ROOT 6 (adjust makefiles), keep Module.mk.ver5

  December 2013
  - Start of project
  - Move ROOT-relevant functionality from DABC plugin
  - Introduce THttpServer, THttpEngine and TRootSniffer classes
  - Integrate JSRootIO code

