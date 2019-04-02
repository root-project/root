sap.ui.define([
    "sap/ui/model/json/JSONModel",
    "rootui5/geom/model/BrowserListBinding",
    "sap/base/Log"
], function(JSONModel, BrowserListBinding, Log) {
   "use strict";

    var hRootModel = JSONModel.extend("rootui5.eve7.model.BrowserModel", {

        constructor: function() {
            JSONModel.apply(this);
            this.setProperty("/", {
                nodes: {}, // nodes shown in the TreeTable, should be flat list
                length: 0  // total number of elements
            });

            // this is true hierarchy, created on the client side and used for creation of flat list
            this.h = {
               name: "ROOT",
               _expanded: true
            };

            this.loadDataCounter = 0; // counter of number of nodes

            this.sortOrder = "";

            this.threshold = 100; // default threshold to prefetch items
        },

        /* Method can be used when complete hierarchy is ready and can be used directly */
        setFullModel: function(topnode) {
           this.fullModel = this.h = topnode;
           if (this.oBinding)
              this.oBinding.checkUpdate(true);
        },

        clearFullModel: function() {
           this.h = {
              name: "ROOT",
             _expanded: true
           };
           delete this.fullModel;
        },


        bindTree: function(sPath, oContext, aFilters, mParameters, aSorters) {
           Log.warning("root.model.hModel#bindTree() " + sPath);

           console.log('BINDING TREE!!!!!!!!!!!!! ' + sPath);

           this.oBinding = new BrowserListBinding(this, sPath, oContext, aFilters, mParameters, aSorters);
           return this.oBinding;
        },

        getLength: function() {
           return this.getProperty("/length");
        },

        getNodeByPath: function(path) {
           var curr = this.h;
           if (!path || (typeof path !== "string") || (path == "/")) return curr;

           var names = path.split("/");

           while (names.length > 0) {
              var name = names.shift(), find = false;
              if (!name) continue;

              for (var k=0;k<curr._childs.length;++k) {
                 if (curr._childs[k].name == name) {
                    curr = curr._childs[k];
                    find = true;
                    break;
                 }
              }

              if (!find) return null;
           }
           return curr;
        },


        sendFirstRequest: function(websocket) {
           console.log('SENDING FIRST REQUEST');
           this._websocket = websocket;
           // submit top-level request already when construct model
           this.submitRequest(this.h, "/");
        },

        // submit next request to the server
        // now using simple HTTP requests, in ROOT websocket communication will be used
        submitRequest: function(elem, path, first, number) {

           if (!this._websocket || elem._requested || this.fullModel) return;
           elem._requested = true;

           this.loadDataCounter++;

           var request = {
              _typename: "ROOT::Experimental::RBrowserRequest",
              path: path,
              first: first || 0,
              number: number || 0,
              sort: this.sortOrder || ""
           };

           console.log('SEND BROWSER REQUEST ' + path);

           this._websocket.Send("BRREQ:" + JSON.stringify(request));
        },

        // process reply from server
        // In the future reply will be received via websocket channel
        processResponse: function(reply) {

           this.loadDataCounter--;

           console.log('PROCESS BR RESPONSE', reply.path, reply);

           var elem = this.getNodeByPath(reply.path);

           if (!elem) { console.error('DID NOT FOUND ' + reply.path); return; }

           if (!elem._requested) console.error('ELEMENT WAS NOT REQUESTED!!!!', reply.path);
           delete elem._requested;

           var smart_merge = false;

           if ((elem._nchilds === reply.nchilds) && elem._childs && reply.nodes) {
              if (elem._first + elem._childs.length == reply.first) {
                 elem._childs = elem._childs.concat(reply.nodes);
                 smart_merge = true;
              } else if (reply.first + reply.nodes.length == elem._first) {
                 elem._first = reply.first;
                 elem._childs = reply.nodes.concat(elem._childs);
                 smart_merge = true;
              }
           }

           if (!smart_merge) {
              elem._nchilds = reply.nchilds;
              elem._childs = reply.nodes;
              elem._first = reply.first || 0;
           }

           this.scanShifts();

           // reset existing nodes if reply does not match with expectation
           if (!smart_merge)
              this.reset_nodes = true;

           console.log("this.loadDataCounter",this.loadDataCounter );

           if (this.loadDataCounter == 0)
              if (this.oBinding)
                 this.oBinding.checkUpdate(true);
        },

        // return element of hierarchical structure by TreeTable index
        getElementByIndex: function(indx) {
           var nodes = this.getProperty("/nodes"),
               node = nodes ? nodes[indx] : null;

           return node ? node._elem : null;
        },

        // function used to calculate all ids shifts and total number of elements
        scanShifts: function() {

           var id = 0;

           function scan(lvl, elem) {

              id++;

              var before_id = id;

              if (elem._expanded) {
                 if (elem._childs === undefined) {
                    // do nothing, childs are not visible as long as we do not have any list

                    // id += 0;
                 } else {

                    // gap at the begin
                    if (elem._first)
                       id += elem._first;

                    // jump over all childs
                    for (var k=0;k<elem._childs.length;++k)
                       scan(lvl+1, elem._childs[k]);

                    // gap at the end
                    var _last = (elem._first || 0) + elem._childs.length;
                    var _remains = elem._nchilds  - _last;
                    if (_remains > 0) id += _remains;
                 }
              }

              // this shift can be later applied to jump over all elements
              elem._shift = id - before_id;
           }

           scan(0, this.h);

           this.setProperty("/length", id);

           return id;
        },

        // main  method to create flat list of nodes - only whose which are specified in selection
        // following arguments:
        //    args.begin     - first visisble element from flat list
        //    args.end       - first not-visisble element
        //    args.threshold - extra elements (before/after) which probably should be prefetched
        // returns total number of nodes
        buildFlatNodes: function(args) {

           var pthis = this,
               id = 0,            // current id, at the same time number of items
               threshold = args.threshold || this.threshold || 100,
               threshold2 = Math.round(threshold/2),
               nodes = this.reset_nodes ? {} : this.getProperty("/nodes");

           // main method to scan through all existing sub-folders
           function scan(lvl, elem, path) {
              // create elements with safety margin
              if ((nodes !== null) && !nodes[id] && (id >= args.begin - threshold2) && (id < args.end + threshold2) )
                 nodes[id] = {
                    name: elem.name,
                    level: lvl,
                    index: id,
                    _elem: elem,

                    // these are optional, should be eliminated in the future
                    type: elem.nchilds || (id == 0) ? "folder" : "file",
                    isLeaf: !elem.nchilds,
                    expanded: !!elem._expanded
                 };

              id++;

              if (!elem._expanded) return;

              if (elem._childs === undefined) {
                 // add new request - can we check if only special part of childs is required?

                 // TODO: probably one could guess more precise request
                 pthis.submitRequest(elem, path);

                 return;
              }

              // check if scan is required
              if (((id + elem._shift) < args.begin - threshold2) || (id >= args.end + threshold2)) {
                 id += elem._shift;
                 return;
              }

              // when not all childs from very beginning is loaded, but may be required
              if (elem._first) {

                 // check if requests are needed to load part in the begin of the list
                 if (args.begin - id - threshold2 < elem._first) {

                    var first = Math.max(args.begin - id - threshold2, 0),
                        number = Math.min(elem._first - first, threshold);

                    pthis.submitRequest(elem, path, first, number);
                 }

                 id += elem._first;
              }

              for (var k=0;k<elem._childs.length;++k)
                 scan(lvl+1, elem._childs[k], path + elem._childs[k].name + "/");

              // check if more elements are required

              var _last = (elem._first || 0) + elem._childs.length;
              var _remains = elem._nchilds  - _last;

              if (_remains > 0) {
                 if (args.end + threshold2 > id) {

                    var first = _last, number = args.end + threshold2 - id;
                    if (number < threshold) number = threshold; // always request much
                    if (number > _remains) number = _remains; // but not too much
                    if (number > threshold) {
                       first += (number - threshold);
                       number = threshold;
                    }

                    pthis.submitRequest(elem, path, first, number);
                 }

                 id += _remains;
              }
           }

           scan(0, this.h, "/");

           if (this.getProperty("/length") != id) {
              // console.error('LENGTH MISMATCH', this.getProperty("/length"), id);
              this.setProperty("/length", id); // update length property
           }

           if (this.reset_nodes) {
              this.setProperty("/nodes", nodes);
              delete this.reset_nodes;
           }

           return id;
        },

        // toggle expand state of specified node
        toggleNode: function(index) {

           var elem = this.getElementByIndex(index);
           if (!elem) return;

           console.log('Toggle element', elem.name)

           if (elem._expanded) {
              delete elem._expanded;
              delete elem._childs; // TODO: for the future keep childs but make request if expand once again

              // close folder - reassign shifts
              this.reset_nodes = true;
              this.scanShifts();

              return true;

           } else if (elem.nchilds || (elem.index==0)) {

              elem._expanded = true;
              // structure is changing but not immediately

              return true;

           } else {
              // nothing to do
              return;
           }

           // for now - reset all existing nodes and rebuild from the beginning
           // all nodes should be created from scratch

           // no need to update - this should be invoked from openui anyway
           //   if (this.oBinding) this.oBinding.checkUpdate(true);
        },


        // change sorting method, for now server supports default, "direct" and "reverse"
        changeSortOrder: function(newValue) {
           if (newValue === undefined)
               newValue = this.getProperty("/sortOrder") || "";

           if ((newValue !== "") && (newValue !=="direct") && (newValue !== "reverse")) {
              console.error('WRONG sorting order ', newValue, 'use default');
              newValue = "";
           }

           // ignore same value
           if (newValue === this.sortOrder)
              return;


           this.sortOrder = newValue;

           // now we should request values once again

           this.submitRequest(this.h, "/");

        }

    });

    return hRootModel;

});
