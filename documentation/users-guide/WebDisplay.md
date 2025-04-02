# Web-based display

In this chapter discussed how web-based display can be created in the ROOT.

## Basics

Idea of web displays is implementation of user interface, which can run remotely
or locally in the web-browsers, fully decoupled from the application code.
For the communication between application and browser websockets are used.
On the server side ROOT application runs THttpServer instance which serves one or
several clients. Client is any web browser


## Creating web-window

**`ROOT::RWebWindow`** class is represent window instance, which can be displayed in the browser

```{.cpp}

auto win = ROOT::RWebWindow::Create();

// set HTML page which is showed when window displayed
win->SetDefaultPage("file:page.html"); // set

// allow unlimited user connections to the window (default only 1)
ROOT::RWebWindowsManager::SetSingleConnMode(false);
win->SetConnLimit(0);

// configure predefined geometry
win->SetGeometry(300, 300);

```

## Communication

To receive data from the browser, data callback should be assigned.
Callback is invoked when data received from the client or when connection is established.
Normally lambda function is used:

```{.cpp}

win->SetDataCallBack([](unsigned connid, const std::string &msg) {

   printf("Msg:%s from connection:%u\n", msg.c_str(), connid);

});

```

Here **connid** is unique identifier, which assign to each connection when it is established.
The connection identifier should be used when sending message to the client:

```{.cpp}

unsigned connid = win->GetConnectionId(); // first connection
std::string msg = "Hello, world";
win->Send(connid, msg);

```

## Client code

The minimal HTML/JavaScript code, which establish connection with the server, looks like:

```{html}
<!DOCTYPE HTML>
<html>
   <head>
      <meta charset="utf-8">
      <title>RWebWindow Example</title>
      <!--jsroot_importmap-->
   </head>
   <body>
      <div id="main"></div>
   </body>
   <script type="module">
      import { connectWebWindow } from 'jsroot/webwindow';
      connectWebWindow({
         receiver: {
            onWebsocketOpened(handle) {
                console.log('Connected');
                handle.send('Init msg from client');
            },
            onWebsocketMsg(handle, msg) {
                console.log('Get message ' + msg);
                document.getElementById('main').innerHTML = msg;
            },
            onWebsocketClosed(handle) {
               // when connection closed, close panel as well
               window?.close();
            }
         }
      });
   </script>
</html>

```

Here `jsroot/webwindow` module is loaded to let establish connection with ROOT application.
It includes all necessary initialization and authentication of websocket connection with the server.
Beside this part there is no limitations which HTML/JS framework should be used to organize layout and code on the client side.


## Display window

To configure web display one uses `--web=<kind>` argument when starting ROOT.
Typical values are:

- "chrome": select Google Chrome browser for interactive web display
- "firefox": select Mozilla Firefox browser for interactive web display
- "edge": select Microsoft Edge browser for interactive web display
- "qt6": uses QWebEngine from Qt6, no real http server started (requires `qt6web` component build for ROOT)
- "cef": uses Chromium Embeded Framework, no real http server started (requires `cefweb` component build for ROOT)
- "default": system default web browser, invoked with `xdg-open` on Linux, `start` on Mac or `open` on Windows
- "off": turns off the web display and comes back to normal graphics in  interactive mode.

Alternatively one can call `gROOT->SetWebDisplay("<kind>")` to specify display kind.
Same argument can be provided directly to the `RWebWindow::Show()`.

With the method `win->GetUrl()` one obtains URL string, which can be typed in the browser address string directly.


## Use ROOT web widgets on the remote nodes

It is advised to use the `rootssh` script with built-in port forwarding and run
the user interface on the local host with the default web browser. Like:

    [localhost] rootssh username@remotenode

As with regular `ssh`, one can specify command which should be run on remote node:

    [localhost] rootssh username@remotenode "root --web -e 'new TBrowser'"

Script automatically creates tunnel and configures several shell variables in remote session. These are:

- `ROOT_WEBGUI_SOCKET` - unix socket which will be used by ROOT `THttpServer` in webgui
- `ROOT_LISTENER_SOCKET` - unix socket which gets messages from ROOT when new web widgets are started

When on remote node in ROOT session new web widget is created, default web browser will be started on the local node and display created widget.

It is highly recommended to use `rootssh` script on public nodes like `lxplus`. Unix sockets, which are created on
the remote session, are configured with `0700` file mode - means only user allowed to access them.

One can provide `--port number` parameter to configure port number on local node and `--browser <name>` to specify
web browser executable to display web widgets. Like:

    [localhost] rootssh --port 8877 --browser chromium username@remotenode

Also any kind of normal `ssh` arguments can be specified:

    [localhost] rootssh -Y -E file.log username@remotenode

On remote node root session should be started with `root --web` argument to advise ROOT use web widgets. Like:

    [remotehost] root --web hsimple.C

[`rootssh` script](https://raw.githubusercontent.com/root-project/root/master/config/rootssh) can be download and used independently from ROOT installation - it is only required that supported ROOT version installed on remote node.


