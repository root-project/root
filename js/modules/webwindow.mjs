import { settings, httpRequest, createHttpRequest, loadScript, decodeUrl,
         browser, setBatchMode, isBatchMode, isObject, isFunc, isStr, btoa_func } from './core.mjs';
import { closeCurrentWindow, showProgress, loadOpenui5 } from './gui/utils.mjs';
import { sha256, sha256_2 } from './base/sha256.mjs';


// secret session key used for hashing connections keys
// only if set, all messages from and to server signed with HMAC hash
let sessionKey = '';

/** @summary HMAC implementation
 * @desc see https://en.wikipedia.org/wiki/HMAC for more details
 * @private */
function HMAC(key, m, o) {
   const kbis = sha256(sessionKey + key),
         block_size = 64,
         opad = 0x5c, ipad = 0x36,
         ko = [], ki = [];
   while (kbis.length < block_size)
      kbis.push(0);
   for (let i = 0; i < kbis.length; ++i) {
      const code = kbis[i];
      ko.push(code ^ opad);
      ki.push(code ^ ipad);
   }

   const hash = sha256_2(ki, (o === undefined) ? m : new Uint8Array(m, o));

   return sha256_2(ko, hash, true);
}

/**
 * @summary Class emulating web socket with long-poll http requests
 *
 * @private
 */

class LongPollSocket {

   constructor(addr, _raw, _handle, _counter) {
      this.path = addr;
      this.connid = null;
      this.req = null;
      this.raw = _raw;
      this.handle = _handle;
      this.counter = _counter;

      this.nextRequest('', 'connect');
   }

   /** @summary Submit next request */
   nextRequest(data, kind) {
      let url = this.path, reqmode = 'buf', post = null;
      if (kind === 'connect') {
         url += this.raw ? '?raw_connect' : '?txt_connect';
         if (this.handle) url += '&' + this.handle.getConnArgs(this.counter++);
         this.connid = 'connect';
      } else if (kind === 'close') {
         if ((this.connid === null) || (this.connid === 'close')) return;
         url += `?connection=${this.connid}&close`;
         if (this.handle) url += '&' + this.handle.getConnArgs(this.counter++);
         this.connid = 'close';
         reqmode = 'text;sync'; // use sync mode to close connection before browser window closed
      } else if ((this.connid === null) || (typeof this.connid !== 'number')) {
         if (!browser.qt6) console.error('No connection');
      } else {
         url += '?connection=' + this.connid;
         if (this.handle) url += '&' + this.handle.getConnArgs(this.counter++);
         if (kind === 'dummy') url += '&dummy';
      }

      if (data) {
         if (this.raw) {
            // special workaround to avoid POST request, use base64 coding
            url += '&post=' + btoa_func(data);
         } else {
            // send data with post request - most efficient way
            reqmode = 'postbuf';
            post = data;
         }
      }

      createHttpRequest(url, reqmode, function(res) {
         // this set to the request itself, res is response

         if (this.handle.req === this)
            this.handle.req = null; // get response for existing dummy request

         if (res === null)
            return this.handle.processRequest(null);

         if (this.handle.raw) {
            // raw mode - all kind of reply data packed into binary buffer
            // first 4 bytes header 'txt:' or 'bin:'
            // after the 'bin:' there is length of optional text argument like 'bin:14  :optional_text'
            // and immediately after text binary data. Server sends binary data so, that offset should be multiple of 8

            const u8Arr = new Uint8Array(res);
            let str = '', i = 0, offset = u8Arr.length;
            if (offset < 4) {
               if (!browser.qt6) console.error(`longpoll got short message in raw mode ${offset}`);
               return this.handle.processRequest(null);
            }

            while (i < 4) str += String.fromCharCode(u8Arr[i++]);
            if (str !== 'txt:') {
               str = '';
               while ((i < offset) && (String.fromCharCode(u8Arr[i]) !== ':'))
                  str += String.fromCharCode(u8Arr[i++]);
               ++i;
               offset = i + parseInt(str.trim());
            }

            str = '';
            while (i < offset) str += String.fromCharCode(u8Arr[i++]);

            if (str) {
               if (str === '<<nope>>')
                  this.handle.processRequest(-1111);
               else
                   this.handle.processRequest(str);
            }
            if (offset < u8Arr.length)
               this.handle.processRequest(res, offset);
         } else if (this.getResponseHeader('Content-Type') === 'application/x-binary') {
            // binary reply with optional header
            const extra_hdr = this.getResponseHeader('LongpollHeader');
            if (extra_hdr) this.handle.processRequest(extra_hdr);
            this.handle.processRequest(res, 0);
         } else {
            // text reply
            if (res && !isStr(res)) {
               let str = '';
               const u8Arr = new Uint8Array(res);
               for (let i = 0; i < u8Arr.length; ++i)
                  str += String.fromCharCode(u8Arr[i]);
               res = str;
            }
            if (res === '<<nope>>')
               this.handle.processRequest(-1111);
            else
               this.handle.processRequest(res);
         }
      }, function(/* err, status */) {
         this.handle.processRequest(null, 'error');
      }, true).then(req => {
         req.handle = this;
         if (!this.req)
            this.req = req; // any request can be used for response, do not submit dummy until req is there
         req.send(post);
      });
   }

   /** @summary Process request */
   processRequest(res, _offset) {
      if (res === null) {
         if (isFunc(this.onerror))
            this.onerror('receive data with connid ' + (this.connid || '---'));
         if ((_offset === 'error') && isFunc(this.onclose))
            this.onclose('force_close');
         this.connid = null;
         return;
      } else if (res === -1111)
         res = '';

      let dummy_tmout = 5;

      if (this.connid === 'connect') {
         if (!res) {
            this.connid = null;
            if (isFunc(this.onerror))
               this.onerror('connection rejected');
            return;
         }

         this.connid = parseInt(res);
         dummy_tmout = 100; // when establishing connection, wait a bit longer to submit dummy package
         if (settings.Debug)
            console.log(`Get new longpoll connection with id ${this.connid}`);
         if (isFunc(this.onopen))
            this.onopen();
      } else if (this.connid === 'close') {
         if (isFunc(this.onclose))
            this.onclose();
         return;
      } else if (isFunc(this.onmessage) && res)
         this.onmessage({ data: res, offset: _offset });

      // minimal timeout to reduce load, generate dummy only if client not submit new request immediately
      if (!this.req)
         setTimeout(() => { if (!this.req) this.nextRequest('', 'dummy'); }, dummy_tmout);
   }

   /** @summary Send data */
   send(str) { this.nextRequest(str); }

   /** @summary Close connection */
   close() { this.nextRequest('', 'close'); }

} // class LongPollSocket

// ========================================================================================

/**
 * @summary Class re-playing socket data from stored protocol
 *
 * @private
 */

class FileDumpSocket {

   #wait_for_file;
   #receiver;

   constructor(receiver) {
      this.#receiver = receiver;
      this.protocol = [];
      this.cnt = 0;
      this.sendcnt = 0;
      httpRequest('protocol.json', 'text').then(res => this.getProtocol(res));
   }

   /** @summary Get stored protocol */
   getProtocol(res) {
      if (!res) return;
      this.protocol = JSON.parse(res);
      if (isFunc(this.onopen))
         this.onopen();
      this.nextOperation();
   }

   /** @summary Emulate send - just count operation */
   send(/* str */) {
      if (this.protocol[this.cnt] === 'send') {
         this.cnt++;
         setTimeout(() => this.nextOperation(), 10);
      } else
         this.sendcnt++;
   }

   /** @summary Emulate close */
   close() {}

   /** @summary Read data for next operation */
   nextOperation() {
      // when file request running - just ignore
      if (this.#wait_for_file)
         return;
      const fname = this.protocol[this.cnt];
      if (!fname) return;

      if (fname === 'send') {
         if (this.sendcnt > 0) {
            this.sendcnt--;
            this.cnt++;
            this.nextOperation();
         }
         return;
      }

      this.#wait_for_file = true;
      this.cnt++;
      httpRequest(fname, (fname.indexOf('.bin') > 0 ? 'buf' : 'text')).then(res => {
         this.#wait_for_file = false;
         if (!res) return;
         const p = fname.indexOf('_ch'),
               chid = (p > 0) ? Number.parseInt(fname.slice(p+3, fname.indexOf('.', p))) : 1;
         if (isFunc(this.#receiver?.provideData))
            this.#receiver.provideData(chid, res, 0);
         setTimeout(() => this.nextOperation(), 10);
      });
   }

} // class FileDumpSocket


/**
 * @summary Client communication handle for RWebWindow.
 *
 * @desc Should be created with {@link connectWebWindow} function
 */

class WebWindowHandle {

   #state; // 0 - init, 1 - connected, -1 - closed
   #kind; // websocket kind - websocket, file, longpoll, rawlongpoll
   #key; // connection key
   #token; // connection token (alternative to key)
   #new_key; // new key for reconnect
   #can_modify_url; // if widget URL can be modified to store new key
   #ws; // websocket or emulation
   #href; // widget url
   #receiver; // receiver of messages
   #channels; // sub-channels
   #master; // master connection
   #channelid; // channel id if this is sub-channel in master connection
   #ask_reload; // flag set when page reload is triggered
   #credits; // configured number of credits
   #ackn; // number of acknowledged packets, regularly send to server to keep sending
   #cansend; // number of packets client can send
   #send_seq; // sequence counter of send messages
   #recv_seq; // sequence counter of received messages
   #secondary; // true when created as extra connection, not need to use keys
   #msgqueue; // messages queue
   #loop_msgqueue; // flag when looping in message queue
   #timerid; // keep alive timer
   #next_binary; // set to channel id when next binary message expected
   #next_binary_hash; // hash of next binary message
   #wait_first_recv; // first received message via the channel is confirmation of established connection
   #user_args; // user arguments for web socket
   #handling_reload; // true when Ctrl-R handler was installed

   constructor(socket_kind, credits, master, chid) {
      this.#kind = socket_kind;
      this.#state = 0;
      this.#credits = Math.max(3, credits || 10);
      this.#cansend = this.#credits;
      this.#ackn = this.#credits;
      this.#send_seq = 1;
      this.#recv_seq = 0;

      if (socket_kind === 'channel') {
         this.#wait_first_recv = true;
         this.#master = master;
         this.#channelid = chid;
      }
   }

   /** @summary Returns arguments specified in the RWebWindow::SetUserArgs() method
     * @desc Can be any valid JSON expression. Undefined by default.
     * @param {string} [field] - if specified and user args is object, returns correspondent object member
     * @return user arguments object */
   getUserArgs(field) {
      if (field && isStr(field))
         return isObject(this.#user_args) ? this.#user_args[field] : undefined;

      return this.#user_args;
   }

   /** @summary Set user args
     * @desc Normally set via RWebWindow::SetUserArgs() method */
   setUserArgs(args) { this.#user_args = args; }

   /** @summary Set key and token
     * @private */
   setKeyToken(key, token, can_modify_url) {
      this.#key = key;
      this.#token = token;
      if (can_modify_url !== undefined)
         this.#can_modify_url = can_modify_url;
   }

   /** @summary Set callbacks receiver.
     * @param {object} obj - object with receiver functions
     * @param {function} obj.onWebsocketMsg - called when new data received from RWebWindow
     * @param {function} obj.onWebsocketOpened - called when connection established
     * @param {function} obj.onWebsocketClosed - called when connection closed
     * @param {function} obj.onWebsocketError - called when get error via the connection */
   setReceiver(obj) { this.#receiver = obj; }

   /** @summary Cleanup and close connection. */
   cleanup() {
      this.#receiver = undefined;
      this.close(true);
   }

   /** @summary Invoke method in the receiver.
    * @private */
   invokeReceiver(brdcst, method, arg, arg2) {
      if (this.#receiver && isFunc(this.#receiver[method]))
         this.#receiver[method](this, arg, arg2);

      if (brdcst && this.#channels) {
         const ks = Object.keys(this.#channels);
         for (let n = 0; n < ks.length; ++n)
            this.#channels[ks[n]].invokeReceiver(false, method, arg, arg2);
      }
   }

   /** @summary Provide data for receiver. When no queue - do it directly.
    * @private */
   provideData(chid, msg, len) {
      if (this.#wait_first_recv) {
         // here dummy first recv like EMBED_DONE is handled
         this.#wait_first_recv = undefined;
         this.#state = 1;
         return this.invokeReceiver(false, 'onWebsocketOpened');
      }

      if ((chid > 1) && this.#channels) {
         const channel = this.#channels[chid];
         if (channel)
            return channel.provideData(1, msg, len);
      }

      const force_queue = len && (len < 0);
      if (!force_queue && !this.#msgqueue?.length)
         return this.invokeReceiver(false, 'onWebsocketMsg', msg, len);

      if (!this.#msgqueue)
         this.#msgqueue = [];
      if (force_queue)
         len = undefined;

      this.#msgqueue.push({ ready: true, msg, len });
   }

   /** @summary Reserve entry in queue for data, which is not yet decoded.
    * @private */
   reserveQueueItem() {
      if (!this.#msgqueue)
         this.#msgqueue = [];
      const item = { ready: false, msg: null, len: 0 };
      this.#msgqueue.push(item);
      return item;
   }

   /** @summary Provide data for item which was reserved before.
    * @private */
   markQueueItemDone(item, _msg, _len) {
      item.ready = true;
      item.msg = _msg;
      item.len = _len;
      this.processQueue();
   }

   /** @summary Process completed messages in the queue
     * @private */
   processQueue() {
      if (this.#loop_msgqueue || !this.#msgqueue)
         return;
      this.#loop_msgqueue = true;
      while (this.#msgqueue.length && this.#msgqueue[0].ready) {
         const front = this.#msgqueue.shift();
         this.invokeReceiver(false, 'onWebsocketMsg', front.msg, front.len);
      }
      if (!this.#msgqueue.length)
         this.#msgqueue = undefined;
      this.#loop_msgqueue = undefined;
   }

   /** @summary Close connection */
   close(force) {
      if (this.#master) {
         this.#master.send(`CLOSECH=${this.#channelid}`, 0);
         this.#master.removeChannel(this.#channelid);
         this.#master = undefined;
         return;
      }

      if (this.#timerid) {
         clearTimeout(this.#timerid);
         this.#timerid = undefined;
      }

      if (this.#ws && (this.#state > 0)) {
         this.#state = force ? -1 : 0; // -1 prevent socket from reopening
         this.#ws.onclose = null; // hide normal handler
         this.#ws.close();
         this.#ws = undefined;
      }
   }

   /** @summary Checks number of credits for send operation
     * @param {number} [numsend = 1] - number of required send operations
     * @return true if one allow to send specified number of text message to server */
   canSend(numsend) { return this.#cansend >= (numsend || 1); }

   /** @summary Returns number of possible send operations relative to number of credits */
   getRelCanSend() { return !this.#credits ? 1 : this.#cansend / this.#credits; }

   /** @summary Send text message via the connection.
     * @param {string} msg - text message to send
     * @param {number} [chid] - channel id, 1 by default, 0 used only for internal communication */
   send(msg, chid) {
      if (this.#master)
         return this.#master.send(msg, this.#channelid);

      if (!this.#ws || (this.#state <= 0))
         return false;

      if (!Number.isInteger(chid))
         chid = 1; // when not configured, channel 1 is used - main widget

      if (this.#cansend === 0)
         console.error('No credits for send, increase "WebGui.ConnCredits" value on server');

      const prefix = `${this.#send_seq++}:${this.#ackn}:${this.#cansend}:${chid}:`,
            hash = this.#key && sessionKey ? HMAC(this.#key, `${prefix}${msg}`) : 'none';

      this.#ackn = 0;
      this.#cansend--; // decrease number of allowed send packets

      this.#ws.send(`${hash}:${prefix}${msg}`);

      if ((this.#kind === 'websocket') || (this.#kind === 'longpoll')) {
         if (this.#timerid)
            clearTimeout(this.#timerid);
         this.#timerid = setTimeout(() => this.keepAlive(), 10000);
      }

      return true;
   }

   /** @summary Send only last message of specified kind during defined time interval.
     * @desc Idea is to prevent sending multiple messages of similar kind and overload connection
     * Instead timeout is started after which only last specified message will be send
     * @private */
   sendLast(kind, tmout, msg) {
      let d = this._delayed;
      if (!d) d = this._delayed = {};
      d[kind] = msg;
      if (!d[`${kind}_handler`])
         d[`${kind}_handler`] = setTimeout(() => { delete d[`${kind}_handler`]; this.send(d[kind]); }, tmout);
   }

   /** @summary Inject message(s) into input queue, for debug purposes only
     * @private */
   inject(msg, chid, immediate) {
      // use timeout to avoid too deep call stack
      if (!immediate)
         return setTimeout(this.inject.bind(this, msg, chid, true), 0);

      if (chid === undefined)
         chid = 1;

      if (Array.isArray(msg)) {
         for (let k = 0; k < msg.length; ++k)
            this.provideData(chid, isStr(msg[k]) ? msg[k] : JSON.stringify(msg[k]), -1);
         this.processQueue();
      } else if (msg)
         this.provideData(chid, isStr(msg) ? msg : JSON.stringify(msg));
   }

   /** @summary Send keep-alive message.
     * @desc Only for internal use, only when used with web sockets
     * @private */
   keepAlive() {
      this.#timerid = undefined;
      this.send('KEEPALIVE', 0);
   }

   /** @summary Request server to resize window
     * @desc For local displays like CEF or qt5 only server can do this */
   resizeWindow(w, h) {
      if (browser.qt6 || browser.cef3)
         this.send(`RESIZE=${w},${h}`, 0);
      else if ((typeof window !== 'undefined') && isFunc(window?.resizeTo))
         window.resizeTo(w, h);
   }

      /** @summary Remove existing channel.
    * @private */
   removeChannel(chid) {
      if (this.#channels)
         this.#channels[chid] = undefined;
   }

   /** @summary Method open channel, which will share same connection, but can be used independently from main
     * If @param url is provided - creates fully independent instance and perform connection with it
     * @private */
   createChannel(url) {
      if (url)
         return this.createNewInstance(url);

      if (this.#master)
         return this.#master.createChannel();

      if (!this.#channels)
         this.#channels = { freeid: 2 };

      const channel = new WebWindowHandle('channel', this.#credits, this, this.#channels.freeid++);

      // register
      this.#channels[channel.getChannelId()] = channel;

      // now server-side entity should be initialized and init message send from server side!
      return channel;
   }

   /** @summary Returns true if socket connected */
   isConnected() { return this.#state > 0; }

   /** @summary Return websocket kind - like 'websocket' or 'longpoll' */
   getKind() { return this.#kind; }

   /** @summary Standalone mode without server connection */
   isStandalone() { return this.#kind === 'file'; }

   /** @summary Returns used channel ID, 1 by default */
   getChannelId() { return this.#channelid && this.#master ? this.#channelid : 1; }

   /** @summary Returns true if sub-channel in master connection */
   isChannel() { return this.getChannelId() > 1; }

   /** @summary Assign href parameter
     * @param {string} [path] - absolute path, when not specified window.location.url will be used
     * @private */
   setHRef(path, secondary) {
      this.#secondary = secondary;
      if (isStr(path) && (path.indexOf('?') > 0)) {
         this.#href = path.slice(0, path.indexOf('?'));
         const d = decodeUrl(path);
         this.setKeyToken(d.get('key'), d.get('token'));
      } else {
         this.#href = path;
         this.setKeyToken();
      }
   }

   /** @summary Return href part
     * @param {string} [relative_path] - relative path to the handle
     * @private */
   getHRef(relative_path) {
      if (!relative_path || !this.#kind || !this.#href)
         return this.#href;
      let addr = this.#href;
      if (relative_path.indexOf('../') === 0) {
         const ddd = addr.lastIndexOf('/', addr.length - 2);
         addr = addr.slice(0, ddd) + relative_path.slice(2);
      } else
         addr += relative_path;

      return addr;
   }

   /** @summary provide connection args for the web socket
    * @private */
   getConnArgs(ntry) {
      let args = '';
      if (this.#key) {
         const k = HMAC(this.#key, `attempt_${ntry}`);
         args += `key=${k}&ntry=${ntry}`;
      }
      if (this.#token) {
         if (args) args += '&';
         args += `token=${this.#token}`;
      }
      return args;
   }

   /** @summary Connect to the server
     * @param [href] - optional URL to widget, use document URL instead
     * @private */
   connect(href) {
      // ignore connect if channel from master connection configured
      if (this.isChannel())
         return;

      this.close();

      if (href)
         this.setHRef(href, true);

      href = this.#href;

      let ntry = 0;

      const retry_open = first_time => {
         if (this.#state)
            return;

         if (!first_time && settings.Debug)
            console.log(`try connect window again ${new Date().toString()}`);

         if (this.#ws) {
            this.#ws.close();
            this.#ws = undefined;
         }

         if (!href) {
            href = window.location.href;
            if (href && href.indexOf('#') > 0)
               href = href.slice(0, href.indexOf('#'));
            if (href && href.lastIndexOf('/') > 0)
               href = href.slice(0, href.lastIndexOf('/') + 1);
         }
         this.#href = href;
         ntry++;

         if (first_time && settings.Debug)
            console.log(`Opening websocket at ${href}`);

         if (ntry > 2)
            showProgress(`Trying to connect ${href}`);

         let path = href;

         if (this.isStandalone()) {
            path += 'root.filedump';
            this.#ws = new FileDumpSocket(this);
            console.log(`Configure FileDumpSocket ${path}`);
         } else if ((this.#kind === 'websocket') && first_time) {
            path = path.replace('http://', 'ws://').replace('https://', 'wss://') + 'root.websocket';
            console.log(`Configure websocket ${path}`);
            path += '?' + this.getConnArgs(ntry);
            this.#ws = new WebSocket(path);
         } else {
            path += 'root.longpoll';
            console.log(`Configure longpoll ${path}`);
            this.#ws = new LongPollSocket(path, (this.#kind === 'rawlongpoll'), this, ntry);
         }

         if (!this.#ws)
            return;

         this.#ws.onopen = () => {
            if (ntry > 2) showProgress();
            this.#state = 1;

            const reply = (this.#secondary ? '' : 'generate_key;') + (this.#key || '');
            this.send(`READY=${reply}`, 0); // need to confirm connection and request new key
            this.invokeReceiver(false, 'onWebsocketOpened');
         };

         this.#ws.onmessage = e => {
            let msg = e.data;

            if (this.#next_binary) {
               const binchid = this.#next_binary,
                     server_hash = this.#next_binary_hash;
               this.#next_binary = this.#next_binary_hash = undefined;

               if (msg instanceof Blob) {
                  // convert Blob object to BufferArray
                  const reader = new FileReader(), qitem = this.reserveQueueItem();
                  // The file's text will be printed here
                  reader.onload = event => {
                     let result = event.target.result;
                     if (this.#key && sessionKey) {
                        const hash = HMAC(this.#key, result, 0);
                        if (hash !== server_hash) {
                           console.log('Discard binary buffer because of HMAC mismatch');
                           result = new ArrayBuffer(0);
                        }
                     }

                     this.markQueueItemDone(qitem, result, 0);
                  };
                  reader.readAsArrayBuffer(msg, e.offset || 0);
               } else {
                  // this is from CEF or LongPoll handler
                  let result = msg;
                  if (this.#key && sessionKey) {
                     const hash = HMAC(this.#key, result, e.offset || 0);
                     if (hash !== server_hash) {
                        console.log('Discard binary buffer because of HMAC mismatch');
                        result = new ArrayBuffer(0);
                     }
                  }
                  this.provideData(binchid, result, e.offset || 0);
               }

               return;
            }

            if (!isStr(msg))
               return console.log(`unsupported message kind: ${typeof msg}`);

            const i0 = msg.indexOf(':'),
                  server_hash = msg.slice(0, i0),
                  i1 = msg.indexOf(':', i0 + 1),
                  seq_id = Number.parseInt(msg.slice(i0 + 1, i1)),
                  i2 = msg.indexOf(':', i1 + 1),
                  credit = Number.parseInt(msg.slice(i1 + 1, i2)),
                  i3 = msg.indexOf(':', i2 + 1),
                  // cansend = parseInt(msg.slice(i2 + 1, i3)),  // TODO: take into account when sending messages
                  i4 = msg.indexOf(':', i3 + 1),
                  chid = Number.parseInt(msg.slice(i3 + 1, i4));

            // for authentication HMAC checksum and sequence id is important
            // HMAC used to authenticate server
            // sequence id is necessary to exclude submission of same packet again
            if (this.#key && sessionKey) {
               const client_hash = HMAC(this.#key, msg.slice(i0+1));
               if (server_hash !== client_hash)
                  return console.log(`Failure checking server HMAC sum ${server_hash}`);
            }

            if (seq_id <= this.#recv_seq)
               return console.log(`Failure with packet sequence ${seq_id} <= ${this.#recv_seq}`);

            this.#recv_seq = seq_id; // sequence id of received packet
            this.#ackn++;            // count number of received packets,
            this.#cansend += credit;  // how many packets client can send

            msg = msg.slice(i4 + 1);

            if (chid === 0) {
               if (msg === 'CLOSE') {
                  this.close(true); // force closing of socket
                  this.invokeReceiver(true, 'onWebsocketClosed');
               } else if (msg.indexOf('NEW_KEY=') === 0) {
                  this.#new_key = msg.slice(8);
                  this.storeKeyInUrl();
                  if (this.#ask_reload)
                     this.askReload(true);
               }
            } else if (msg.slice(0, 10) === '$$binary$$') {
               this.#next_binary = chid;
               this.#next_binary_hash = msg.slice(10);
            } else if (msg === '$$nullbinary$$')
               this.provideData(chid, new ArrayBuffer(0), 0);
            else
               this.provideData(chid, msg);

            if (this.#ackn > Math.max(2, this.#credits * 0.7))
               this.send('READY', 0); // send dummy message to server
         };

         this.#ws.onclose = arg => {
            this.#ws = undefined;
            if ((this.#state > 0) || (arg === 'force_close')) {
               if (settings.Debug)
                  console.log('websocket closed');
               this.#state = 0;
               this.invokeReceiver(true, 'onWebsocketClosed');
            }
         };

         this.#ws.onerror = err => {
            console.log(`websocket error ${err} state ${this.#state}`);
            if (this.#state > 0) {
               this.invokeReceiver(true, 'onWebsocketError', err);
               this.#state = 0;
            }
         };

         // only in interactive mode try to reconnect
         if (!isBatchMode())
            setTimeout(retry_open, 3000); // after 3 seconds try again
      }; // retry_open

      retry_open(true); // call for the first time
   }

   /** @summary Ask to reload web widget
     * @desc If new key already exists - reload immediately
     * Otherwise request server to generate new key - and then reload page
     * WARNING - call only when knowing that you are doing
     * @private */
   askReload(force) {
      if (this.#new_key || force) {
         this.close(true);
         if (typeof location !== 'undefined')
            location.reload(true);
      } else {
         this.#ask_reload = true;
         this.send('GENERATE_KEY', 0);
      }
   }

   /** @summary Instal Ctrl-R handler to reload web window
     * @desc Instead of default window reload invokes {@link askReload} method
     * WARNING - only call when you know that you are doing
     * @private */
   addReloadKeyHandler() {
      if (this.isStandalone() || this.#handling_reload)
         return;

      // this websocket will handle reload
      // embed widgets should not call this method
      this.#handling_reload = true;

      window.addEventListener('keydown', evnt => {
         if (((evnt.key === 'R') || (evnt.key === 'r')) && evnt.ctrlKey) {
            evnt.stopPropagation();
            evnt.preventDefault();
            console.log('Prevent Ctrl-R propogation - ask reload RWebWindow!');
            this.askReload();
          }
      });
   }

   /** @summary Replace widget URL with new key
     * @private */
   storeKeyInUrl() {
      // do not modify document URLs by secondary widgets
      if (this.#secondary)
         return;

      let href = (typeof document !== 'undefined') ? document.URL : null;

      if (this.#can_modify_url && isStr(href) && (typeof window !== 'undefined')) {
         let prefix = '&key=', p = href.indexOf(prefix);
         if (p < 0) {
            prefix = '?key=';
            p = href.indexOf(prefix);
         }
         if ((p > 0) && this.#new_key) {
            const p1 = href.indexOf('#', p+1), p2 = href.indexOf('&', p+1),
                  pp = (p1 < 0) ? p2 : (p2 < 0 ? p1 : Math.min(p1, p2));
            href = href.slice(0, p) + prefix + this.#new_key + (pp < 0 ? '' : href.slice(pp));
            window.history?.replaceState(window.history.state, undefined, href);
         }
      }

      if (typeof sessionStorage !== 'undefined') {
         sessionStorage.setItem('RWebWindow_SessionKey', sessionKey);
         sessionStorage.setItem('RWebWindow_Key', this.#new_key);
      }
   }

   /** @summary Create new instance of same kind
    * @private */
   createNewInstance(url) {
      const handle = new WebWindowHandle(this.#kind);
      handle.setHRef(this.getHRef(url), true);
      return handle;
   }

} // class WebWindowHandle


/** @summary Method used to initialize connection to web window.
  * @param {object} arg - arguments
  * @param {string} [arg.socket_kind] - kind of connection longpoll|websocket, detected automatically from URL
  * @param {number} [arg.credits = 10] - number of packets which can be send to server without acknowledge
  * @param {object} arg.receiver - instance of receiver for websocket events, allows to initiate connection immediately
  * @param {string} [arg.first_recv] - required prefix in the first message from RWebWindow, remain part of message will be returned in handle.first_msg
  * @param {string} [arg.href] - URL to RWebWindow, using window.location.href by default
  * @return {Promise} {@link WebWindowHandle} instance  */
async function connectWebWindow(arg) {
   // mark that jsroot used with RWebWindow
   browser.webwindow = true;

   if (isFunc(arg))
      arg = { callback: arg };
   else if (!isObject(arg))
      arg = {};

   let d_key, d_token, new_key;

   if (!arg.href) {
      let href = (typeof document !== 'undefined') ? document.URL : '', s_key;
      const p = href.indexOf('#');
      if (p > 0) {
         s_key = href.slice(p + 1);
         href = href.slice(0, p);
      }

      const d = decodeUrl(href);
      d_key = d.get('key');
      d_token = d.get('token');

      if (d_key && s_key && (s_key.length > 20)) {
         sessionKey = s_key;

         if (typeof window !== 'undefined')
            window.history?.replaceState(window.history.state, undefined, href);
      }

      if (typeof sessionStorage !== 'undefined') {
         new_key = sessionStorage.getItem('RWebWindow_Key');
         sessionStorage.removeItem('RWebWindow_Key');

         if (sessionKey)
            sessionStorage.setItem('RWebWindow_SessionKey', sessionKey);
         else
            sessionKey = sessionStorage.getItem('RWebWindow_SessionKey') || '';
      }

      // special holder script, prevents headless chrome browser from too early exit
      if (d.has('headless') && d_key && (browser.isChromeHeadless || browser.isChrome) && !arg.ignore_chrome_batch_holder)
         loadScript('root_batch_holder.js?key=' + (new_key || d_key));

      if (arg.debug || d.has('debug'))
         settings.Debug = true;

      if (!arg.platform)
         arg.platform = d.get('platform');

      if (arg.platform === 'qt6')
         browser.qt6 = true;
      else if (arg.platform === 'cef3')
         browser.cef3 = true;

      if (arg.batch === undefined)
         arg.batch = d.has('headless');

      if (arg.batch)
         setBatchMode(true);

      if (arg.dark)
         settings.DarkMode = true;

      if (!arg.socket_kind)
         arg.socket_kind = d.get('ws');

      if (!new_key && arg.winW && arg.winH && !isBatchMode() && isFunc(window?.resizeTo))
         window.resizeTo(arg.winW, arg.winH);

      if (!new_key && arg.winX && arg.winY && !isBatchMode() && isFunc(window?.moveTo))
         window.moveTo(arg.winX, arg.winY);
   }

   if (!arg.socket_kind) {
      if (browser.qt6)
         arg.socket_kind = 'rawlongpoll';
      else if (browser.cef3)
         arg.socket_kind = 'longpoll';
      else
         arg.socket_kind = 'websocket';
   }

   if (arg.settings)
      Object.assign(settings, arg.settings);

   const main = new Promise(resolveFunc => {
      const handle = new WebWindowHandle(arg.socket_kind, arg.credits);
      handle.setUserArgs(arg.user_args);
      if (arg.href)
         handle.setHRef(arg.href); // apply href now  while connect can be called from other place
      else
         handle.setKeyToken(new_key || d_key, d_token, Boolean(d_key));

      if (typeof window !== 'undefined') {
         window.onbeforeunload = () => handle.close(true);
         if (browser.qt6)
            window.onqt6unload = window.onbeforeunload;
      }

      if (arg.receiver) {
         // when receiver exists, it handles itself callbacks
         handle.setReceiver(arg.receiver);
         handle.connect();
         resolveFunc(handle);
         return;
      }

      if (!arg.first_recv) {
         resolveFunc(handle);
         return;
      }

      handle.setReceiver({
         onWebsocketOpened() {}, // dummy function when websocket connected

         onWebsocketMsg(h, msg) {
            if (msg.indexOf(arg.first_recv))
               return h.close();
            h.first_msg = msg.slice(arg.first_recv.length);
            resolveFunc(h);
         },

         onWebsocketClosed() { closeCurrentWindow(); } // when connection closed, close panel as well
      });

      handle.connect();
   });

   if (!arg.ui5)
      return main;

   return Promise.all([main, loadOpenui5(arg)]).then(arr => arr[0]);
}

export { WebWindowHandle, connectWebWindow };
