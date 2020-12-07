/// @file JSRoot.webwindow.js
/// Connections handling to RWebWindow

JSROOT.define([], () => {

   "use strict";

   /** emulate websocket via long-poll http requests */

   function LongPollSocket(addr, _raw, _args) {
      this.path = addr;
      this.connid = null;
      this.req = null;
      this.raw = _raw;
      this.args = _args;

      this.nextrequest("", "connect");
   }

   LongPollSocket.prototype.nextrequest = function(data, kind) {
      let url = this.path, reqmode = "buf", post = null;
      if (kind === "connect") {
         url += this.raw ? "?raw_connect" : "?txt_connect";
         if (this.args) url += "&" + this.args;
         console.log('longpoll connect ' + url + ' raw = ' + this.raw);
         this.connid = "connect";
      } else if (kind === "close") {
         if ((this.connid === null) || (this.connid === "close")) return;
         url += "?connection=" + this.connid + "&close";
         this.connid = "close";
         reqmode = "text;sync"; // use sync mode to close connection before browser window closed
      } else if ((this.connid === null) || (typeof this.connid !== 'number')) {
         if (!JSROOT.browser.qt5) console.error("No connection");
         return;
      } else {
         url += "?connection=" + this.connid;
         if (kind === "dummy") url += "&dummy";
      }

      if (data) {
         if (this.raw) {
            // special workaround to avoid POST request, use base64 coding
            url += "&post=" + btoa(data);
         } else {
            // send data with post request - most efficient way
            reqmode = "post";
            post = data;
         }
      }

      let req = JSROOT.NewHttpRequest(url, reqmode, function(res) {
         // this set to the request itself, res is response

         if (this.handle.req === this)
            this.handle.req = null; // get response for existing dummy request

         if (res === null)
            return this.handle.processreq(null);

         if (this.handle.raw) {
            // raw mode - all kind of reply data packed into binary buffer
            // first 4 bytes header "txt:" or "bin:"
            // after the "bin:" there is length of optional text argument like "bin:14  :optional_text"
            // and immedaitely after text binary data. Server sends binary data so, that offset should be multiple of 8

            let str = "", i = 0, u8Arr = new Uint8Array(res), offset = u8Arr.length;
            if (offset < 4) {
               if (!JSROOT.browser.qt5) console.error('longpoll got short message in raw mode ' + offset);
               return this.handle.processreq(null);
            }

            while (i < 4) str += String.fromCharCode(u8Arr[i++]);
            if (str != "txt:") {
               str = "";
               while ((i < offset) && (String.fromCharCode(u8Arr[i]) != ':')) str += String.fromCharCode(u8Arr[i++]);
               ++i;
               offset = i + parseInt(str.trim());
            }

            str = "";
            while (i < offset) str += String.fromCharCode(u8Arr[i++]);

            if (str) {
               if (str == "<<nope>>") str = "";
               this.handle.processreq(str);
            }
            if (offset < u8Arr.length)
               this.handle.processreq(res, offset);
         } else if (this.getResponseHeader("Content-Type") == "application/x-binary") {
            // binary reply with optional header
            let extra_hdr = this.getResponseHeader("LongpollHeader");
            if (extra_hdr) this.handle.processreq(extra_hdr);
            this.handle.processreq(res, 0);
         } else {
            // text reply
            if (res && typeof res !== "string") {
               let str = "", u8Arr = new Uint8Array(res);
               for (let i = 0; i < u8Arr.length; ++i)
                  str += String.fromCharCode(u8Arr[i]);
               res = str;
            }
            if (res == "<<nope>>") res = "";
            this.handle.processreq(res);
         }
      });

      req.handle = this;
      if (kind === "dummy") this.req = req; // remember last dummy request, wait for reply
      req.send(post);
   }

   LongPollSocket.prototype.processreq = function(res, _offset) {
      if (res === null) {
         if (typeof this.onerror === 'function') this.onerror("receive data with connid " + (this.connid || "---"));
         // if (typeof this.onclose === 'function') this.onclose();
         this.connid = null;
         return;
      }

      if (this.connid === "connect") {
         if (!res) {
            this.connid = null;
            if (typeof this.onerror === 'function') this.onerror("connection rejected");
            return;
         }

         this.connid = parseInt(res);
         console.log('Get new longpoll connection with id ' + this.connid);
         if (typeof this.onopen == 'function') this.onopen();
      } else if (this.connid === "close") {
         if (typeof this.onclose == 'function') this.onclose();
         return;
      } else {
         if ((typeof this.onmessage === 'function') && res)
            this.onmessage({ data: res, offset: _offset });
      }

      if (!this.req) this.nextrequest("", "dummy"); // send new poll request when necessary
   }

   LongPollSocket.prototype.send = function(str) { this.nextrequest(str); }

   LongPollSocket.prototype.close = function() { this.nextrequest("", "close"); }

   // ========================================================================================

   function FileDumpSocket(receiver) {
      this.receiver = receiver;
      this.protocol = [];
      this.cnt = 0;
      JSROOT.httpRequest("protocol.json", "text").then(res => this.get_protocol(res));
   }

   FileDumpSocket.prototype.get_protocol = function(res) {
      if (!res) return;
      this.protocol = JSON.parse(res);
      if (typeof this.onopen == 'function') this.onopen();
      this.next_operation();
   }

   FileDumpSocket.prototype.send = function(/* str */) {
      if (this.protocol[this.cnt] == "send") {
         this.cnt++;
         setTimeout(this.next_operation.bind(this), 10);
      }
   }

   FileDumpSocket.prototype.close = function() {}

   FileDumpSocket.prototype.next_operation = function() {
      // when file request running - just ignore
      if (this.wait_for_file) return;
      let fname = this.protocol[this.cnt];
      if (!fname) return;
      if (fname == "send") return; // waiting for send
      // console.log("getting file", fname, "wait", this.wait_for_file);
      this.wait_for_file = true;
      this.cnt++;
      JSROOT.httpRequest(fname, (fname.indexOf(".bin") > 0 ? "buf" : "text")).then(this.get_file.bind(this));
   }

   FileDumpSocket.prototype.get_file = function(res) {
      this.wait_for_file = false;
      if (!res) return;
      if (this.receiver.ProvideData)
         this.receiver.ProvideData(1, res, 0);
      setTimeout(this.next_operation.bind(this), 10);
   }

   // ========================================================================================


   /**
    * @summary Client communication handle for RWebWindow.
    *
    * @class
    * @memberof JSROOT
    * @desc Should be created with {@link JSROOT.connectWebWindow} function
    */

   function WebWindowHandle(socket_kind, credits) {
      this.kind = socket_kind;
      this.state = 0;
      this.credits = credits || 10;
      this.cansend = this.credits;
      this.ackn = this.credits;
   }

   /** @summary Returns arguments specified in the RWebWindow::SetUserArgs() method
     * @desc Can be any valid JSON expression. Undefined by default.
     * @param {string} [field] - if specified and user args is object, returns correspondent object member
     * @returns user arguments object */
   WebWindowHandle.prototype.GetUserArgs = function(field) {
      if (field && (typeof field == 'string')) {
         return (this.user_args && (typeof this.user_args == 'object')) ? this.user_args[field] : undefined;
      }

      return this.user_args;
   }

   /** @summary Set callbacks receiver.
     * @param {object} obj - object wiith receiver functions
     * @param {function} obj.OnWebsocketMsg - called when new data receieved from RWebWindow
     * @param {function} obj.OnWebsocketOpened - called when connection established
     * @param {function} obj.OnWebsocketClosed - called when connection closed */
   WebWindowHandle.prototype.SetReceiver = function(obj) {
      this.receiver = obj;
   }

   /** @summary Cleanup and close connection. */
   WebWindowHandle.prototype.Cleanup = function() {
      delete this.receiver;
      this.Close(true);
   }

   /** @summary Invoke method in the receiver.
    * @private */
   WebWindowHandle.prototype.InvokeReceiver = function(brdcst, method, arg, arg2) {
      if (this.receiver && (typeof this.receiver[method] == 'function'))
         this.receiver[method](this, arg, arg2);

      if (brdcst && this.channels) {
         let ks = Object.keys(this.channels);
         for (let n = 0; n < ks.length; ++n)
            this.channels[ks[n]].InvokeReceiver(false, method, arg, arg2);
      }
   }

   /** @summary Provide data for receiver. When no queue - do it directly.
    * @private */
   WebWindowHandle.prototype.ProvideData = function(chid, _msg, _len) {
      if (this.wait_first_recv) {
         console.log("FIRST MESSAGE", chid, _msg);
         delete this.wait_first_recv;
         return this.InvokeReceiver(false, "OnWebsocketOpened");
      }

      if ((chid > 1) && this.channels) {
         let channel = this.channels[chid];
         if (channel)
            return channel.ProvideData(1, _msg, _len);
      }

      let force_queue = _len && (_len < 0);

      if (!force_queue && (!this.msgqueue || !this.msgqueue.length))
         return this.InvokeReceiver(false, "OnWebsocketMsg", _msg, _len);

      if (!this.msgqueue) this.msgqueue = [];
      if (force_queue) _len = undefined;

      this.msgqueue.push({ ready: true, msg: _msg, len: _len });
   }

   /** @summary Reserve entry in queue for data, which is not yet decoded.
    * @private */
   WebWindowHandle.prototype.ReserveQueueItem = function() {
      if (!this.msgqueue) this.msgqueue = [];
      let item = { ready: false, msg: null, len: 0 };
      this.msgqueue.push(item);
      return item;
   }

   /** @summary Reserver entry in queue for data, which is not yet decoded.
    * @private */
   WebWindowHandle.prototype.DoneItem = function(item, _msg, _len) {
      item.ready = true;
      item.msg = _msg;
      item.len = _len;
      this.ProcessQueue();
   }

   /** @summary Process completed messages in the queue
     * @private */
   WebWindowHandle.prototype.ProcessQueue = function() {
      if (this._loop_msgqueue || !this.msgqueue) return;
      this._loop_msgqueue = true;
      while ((this.msgqueue.length > 0) && this.msgqueue[0].ready) {
         let front = this.msgqueue.shift();
         this.InvokeReceiver(false, "OnWebsocketMsg", front.msg, front.len);
      }
      if (this.msgqueue.length == 0)
         delete this.msgqueue;
      delete this._loop_msgqueue;
   }

   /** @summary Close connection. */
   WebWindowHandle.prototype.Close = function(force) {
      if (this.master) {
         this.master.Send("CLOSECH=" + this.channelid, 0);
         delete this.master.channels[this.channelid];
         delete this.master;
         return;
      }

      if (this.timerid) {
         clearTimeout(this.timerid);
         delete this.timerid;
      }

      if (this._websocket && (this.state > 0)) {
         this.state = force ? -1 : 0; // -1 prevent socket from reopening
         this._websocket.onclose = null; // hide normal handler
         this._websocket.close();
         delete this._websocket;
      }
   }

   /** @summary Checks number of credits for send operation
     * @param {number} [numsend = 1] - number of required send operations
     * @returns true if one allow to send specified number of text message to server */
   WebWindowHandle.prototype.CanSend = function(numsend) {
      return (this.cansend >= (numsend || 1));
   }

   /** @summary Returns number of possible send operations relative to number of credits*/
   WebWindowHandle.prototype.getRelCanSend = function() {
      return !this.credits ? 1 : this.cansend / this.credits;
   }

   /** @summary Send text message via the connection.
     * @param {string} msg - text message to send */
   WebWindowHandle.prototype.Send = function(msg, chid) {
      if (this.master)
         return this.master.Send(msg, this.channelid);

      if (!this._websocket || (this.state <= 0)) return false;

      if (isNaN(chid) || (chid === undefined)) chid = 1; // when not configured, channel 1 is used - main widget

      if (this.cansend <= 0) console.error('should be queued before sending cansend: ' + this.cansend);

      let prefix = this.ackn + ":" + this.cansend + ":" + chid + ":";
      this.ackn = 0;
      this.cansend--; // decrease number of allowed send packets

      this._websocket.send(prefix + msg);
      if (this.kind === "websocket") {
         if (this.timerid) clearTimeout(this.timerid);
         this.timerid = setTimeout(this.KeepAlive.bind(this), 10000);
      }

      return true;
   }

   /** @summary Inject message(s) into input queue, for debug purposes only
     * @private */
   WebWindowHandle.prototype.Inject = function(msg, chid, immediate) {
      // use timeout to avoid too deep call stack
      if (!immediate)
         return setTimeout(this.Inject.bind(this, msg, chid, true), 0);

      if (chid === undefined) chid = 1;

      if (Array.isArray(msg)) {
         for (let k = 0; k < msg.length; ++k)
            this.ProvideData(chid, (typeof msg[k] == "string") ? msg[k] : JSON.stringify(msg[k]), -1);
         this.ProcessQueue();
      } else if (msg) {
         this.ProvideData(chid, typeof msg == "string" ? msg : JSON.stringify(msg));
      }
   }

   /** @summary Send keepalive message.
    * @private */
   WebWindowHandle.prototype.KeepAlive = function() {
      delete this.timerid;
      this.Send("KEEPALIVE", 0);
   }

   /** @summary Method open channel, which will share same connection, but can be used independently from main
    * @private */
   WebWindowHandle.prototype.CreateChannel = function() {
      if (this.master)
         return master.CreateChannel();

      let channel = new WebWindowHandle("channel", this.credits);
      channel.wait_first_recv = true; // first received message via the channel is confirmation of established connection

      if (!this.channels) {
         this.channels = {};
         this.freechannelid = 2;
      }

      channel.master = this;
      channel.channelid = this.freechannelid++;

      // register
      this.channels[channel.channelid] = channel;

      // now server-side entity should be initialized and init message send from server side!
      return channel;
   }

   /** @summary Returns used channel ID, 1 by default */
   WebWindowHandle.prototype.getChannelId = function() { return this.channelid && this.master ? this.channelid : 1; }

   /** @summary Method opens relative path with the same kind of socket.
    * @private */
   WebWindowHandle.prototype.CreateRelative = function(relative) {
      if (!relative || !this.kind || !this.href) return null;

      let handle = new WebWindowHandle(this.kind, this.credits);
      console.log('Try to connect ', this.href + relative);
      handle.Connect(this.href + relative);
      return handle;
   }

   /** @summary Create configured socket for current object.
     * @private */
   WebWindowHandle.prototype.Connect = function(href) {

      this.Close();

      let pthis = this, ntry = 0, args = (this.key ? ("key=" + this.key) : "");

      function retry_open(first_time) {

         if (pthis.state != 0) return;

         if (!first_time) console.log("try connect window again " + new Date().toString());

         if (pthis._websocket) pthis._websocket.close();
         delete pthis._websocket;

         let conn = null;
         if (!href) {
            href = window.location.href;
            if (href && href.indexOf("#") > 0) href = href.substr(0, href.indexOf("#"));
            if (href && href.lastIndexOf("/") > 0) href = href.substr(0, href.lastIndexOf("/") + 1);
         }
         pthis.href = href;
         ntry++;

         if (first_time) console.log('Opening web socket at ' + href);

         if (ntry > 2) JSROOT.progress("Trying to connect " + href);

         let path = href;

         if (pthis.kind == "file") {
            path += "root.filedump";
            conn = new FileDumpSocket(pthis);
            console.log('configure protocol log ' + path);
         } else if ((pthis.kind === 'websocket') && first_time) {
            path = path.replace("http://", "ws://").replace("https://", "wss://") + "root.websocket";
            if (args) path += "?" + args;
            console.log('configure websocket ' + path);
            conn = new WebSocket(path);
         } else {
            path += "root.longpoll";
            console.log('configure longpoll ' + path);
            conn = new LongPollSocket(path, (pthis.kind === 'rawlongpoll'), args);
         }

         if (!conn) return;

         pthis._websocket = conn;

         conn.onopen = function() {
            if (ntry > 2) JSROOT.progress();
            pthis.state = 1;

            let key = pthis.key || "";

            pthis.Send("READY=" + key, 0); // need to confirm connection
            pthis.InvokeReceiver(false, "OnWebsocketOpened");
         }

         conn.onmessage = function(e) {
            let msg = e.data;

            if (pthis.next_binary) {

               let binchid = pthis.next_binary;
               delete pthis.next_binary;

               if (msg instanceof Blob) {
                  // this is case of websocket
                  // console.log('Get Blob object - convert to buffer array');
                  let reader = new FileReader, qitem = pthis.ReserveQueueItem();
                  reader.onload = function(event) {
                     // The file's text will be printed here
                     pthis.DoneItem(qitem, event.target.result, 0);
                  }
                  reader.readAsArrayBuffer(msg, e.offset || 0);
               } else {
                  // console.log('got array ' + (typeof msg) + ' len = ' + msg.byteLength);
                  // this is from CEF or LongPoll handler
                  pthis.ProvideData(binchid, msg, e.offset || 0);
               }

               return;
            }

            if (typeof msg != 'string') return console.log("unsupported message kind: " + (typeof msg));

            let i1 = msg.indexOf(":"),
               credit = parseInt(msg.substr(0, i1)),
               i2 = msg.indexOf(":", i1 + 1),
               // cansend = parseInt(msg.substr(i1 + 1, i2 - i1)),  // TODO: take into account when sending messages
               i3 = msg.indexOf(":", i2 + 1),
               chid = parseInt(msg.substr(i2 + 1, i3 - i2));

            pthis.ackn++;            // count number of received packets,
            pthis.cansend += credit; // how many packets client can send

            msg = msg.substr(i3 + 1);

            if (chid == 0) {
               console.log('GET chid=0 message', msg);
               if (msg == "CLOSE") {
                  pthis.Close(true); // force closing of socket
                  pthis.InvokeReceiver(true, "OnWebsocketClosed");
               }
            } else if (msg == "$$binary$$") {
               pthis.next_binary = chid;
            } else if (msg == "$$nullbinary$$") {
               pthis.ProvideData(chid, new ArrayBuffer(0), 0);
            } else {
               pthis.ProvideData(chid, msg);
            }

            if (pthis.ackn > 7)
               pthis.Send('READY', 0); // send dummy message to server
         }

         conn.onclose = function() {
            delete pthis._websocket;
            if (pthis.state > 0) {
               console.log('websocket closed');
               pthis.state = 0;
               pthis.InvokeReceiver(true, "OnWebsocketClosed");
            }
         }

         conn.onerror = function(err) {
            console.log("websocket error " + err);
            if (pthis.state > 0) {
               pthis.InvokeReceiver(true, "OnWebsocketError", err);
               pthis.state = 0;
            }
         }

         // only in interactive mode try to reconnect
         if (!JSROOT.BatchMode)
            setTimeout(retry_open, 3000); // after 3 seconds try again

      } // retry_open

      retry_open(true); // call for the first time
   }


   /** @summary Method used to initialize connection to web window.
    *
    * @param {object} arg - arguments
    * @param {string} [arg.prereq] - prerequicities, which should be loaded first
    * @param {string} [arg.openui5src] - source of openui5, either URL like "https://openui5.hana.ondemand.com" or "jsroot" which provides its own reduced openui5 package
    * @param {string} [arg.openui5libs] - list of openui5 libraries loaded, default is "sap.m, sap.ui.layout, sap.ui.unified"
    * @param {string} [arg.socket_kind] - kind of connection longpoll|websocket, detected automatically from URL
    * @param {number} [arg.credits = 10] - number of packets which can be send to server without acknowledge
    * @param {object} arg.receiver - instance of receiver for websocket events, allows to initiate connection immediately
    * @param {string} arg.first_recv - required prefix in the first message from TWebWindow, remain part of message will be returned in handle.first_msg
    * @param {string} [arg.prereq2] - second part of prerequcities, which is loaded parallel to connecting with WebWindow
    * @returns {Promise} ready-to-use WebWindowHandle instance
    */

   JSROOT.connectWebWindow = function(arg) {

      if (typeof arg == 'function') arg = { callback: arg }; else
         if (!arg || (typeof arg != 'object')) arg = {};

      if (arg.prereq) {
         if (arg.openui5src) JSROOT.openui5src = arg.openui5src;
         if (arg.openui5libs) JSROOT.openui5libs = arg.openui5libs;
         if (arg.openui5theme) JSROOT.openui5theme = arg.openui5theme;
         return JSROOT.require(arg.prereq /*, arg.prereq_logdiv */).then(() => {
            delete arg.prereq;
            if (arg.prereq_logdiv && document) {
               let elem = document.getElementById(arg.prereq_logdiv);
               if (elem) elem.innerHTML = '';
               delete arg.prereq_logdiv;
            }
            return JSROOT.connectWebWindow(arg);
         });
      }

      let d = JSROOT.decodeUrl();

      // special hold script, prevents headless browser from too early exit
      if (d.has("batch_mode") && d.get("key") && (JSROOT.browser.isChromeHeadless || JSROOT.browser.isChrome))
         JSROOT.loadScript("root_batch_holder.js?key=" + d.get("key"));

      if (!arg.platform)
         arg.platform = d.get("platform");

      if (arg.platform == "qt5")
         JSROOT.browser.qt5 = true;
      else if (arg.platform == "cef3")
         JSROOT.browser.cef3 = true;

      if (arg.batch === undefined)
         arg.batch = d.has("batch_mode");

      if (arg.batch) JSROOT.BatchMode = true;

      if (!arg.socket_kind)
         arg.socket_kind = d.get("ws");

      if (!arg.socket_kind) {
         if (JSROOT.browser.qt5)
            arg.socket_kind = "rawlongpoll";
         else if (JSROOT.browser.cef3)
            arg.socket_kind = "longpoll";
         else
            arg.socket_kind = "websocket";
      }

      // only for debug purposes
      // arg.socket_kind = "longpoll";

      let promise = new Promise((resolveFunc) => {
         let handle = new WebWindowHandle(arg.socket_kind, arg.credits);
         handle.user_args = arg.user_args;

         if (window) {
            window.onbeforeunload = handle.Close.bind(handle, true);
            if (JSROOT.browser.qt5) window.onqt5unload = window.onbeforeunload;
         }

         handle.key = d.get("key");

         if (arg.first_recv) {
            arg.receiver = {
               OnWebsocketOpened: () => { }, // dummy function when websocket connected

               OnWebsocketMsg: (handle, msg) => {
                  if (msg.indexOf(arg.first_recv) != 0)
                     return handle.Close();
                  handle.first_msg = msg.substr(arg.first_recv.length);

                  if (!arg.prereq2) resolveFunc(handle);
               },

               OnWebsocketClosed: () => { JSROOT.CloseCurrentWindow(); } // // when connection closed, close panel as well
            };
         }

         if (!arg.receiver)
            return resolveFunc(handle);

         // when receiver is exists, it handles itself callbacks
         handle.SetReceiver(arg.receiver);
         handle.Connect();

         if (arg.prereq2) {
            JSROOT.require(arg.prereq2).then(() => {
               delete arg.prereq2; // indicate that func is loaded
               if (!arg.first_recv || handle.first_msg) resolveFunc(handle);
            });
         } else if (!arg.first_recv) {
            resolveFunc(handle);
         }
      });

      return promise;
   }

   JSROOT.WebWindowHandle = WebWindowHandle;

   return JSROOT;
})
