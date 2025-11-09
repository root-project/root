sap.ui.define([
   'sap/ui/model/json/JSONModel',
   'rootui5/browser/model/BrowserListBinding'
], function (JSONModel, BrowserListBinding) {
   'use strict';

   let hRootModel = JSONModel.extend('rootui5.browser.model.BrowserModel', {

      constructor: function () {
         JSONModel.apply(this);
         this.setProperty('/', {
            nodes: {}, // nodes shown in the TreeTable, should be flat list
            length: 0  // total number of elements
         });

         // this is true hierarchy, created on the client side and used for creation of flat list
         this.h = {
            name: '__Holder__',
            expanded: true
         };

         this.useIndexSuffix = true; // use index suffix in path for handling name duplication

         this.loadDataCounter = 0; // counter of number of nodes

         this.sortMethod = 'name'; // 'name', 'size'
         this.reverseOrder = false;
         this.itemsFilter = '';
         this.showHidden = false;
         this.appendToCanvas = false;
         this.handleDoubleClick = true;
         this.onlyLastCycle = false;

         this.threshold = 100; // default threshold to prefetch items
      },

      /** @summary Assign tree table control */
      assignTreeTable(t) {
         this.treeTable = t;
      },

      /** @summary Set sort method */
      setSortMethod(arg) { this.sortMethod = arg; },

      /** @summary Get sort method */
      getSortMethod() { return this.sortMethod; },

      /** @summary Set show hidden flag */
      setShowHidden(flag) { this.showHidden = flag; },

      /** @summary Is show hidden flag set */
      isShowHidden() { return this.showHidden; },

      /** @summary Set show hidden flag */
      setAppendToCanvas(flag) { this.appendToCanvas = flag; },

      /** @summary Is append to canvas when double click set */
      isAppendToCanvas() { return this.appendToCanvas; },

      /** @summary Set double click handler */
      setHandleDoubleClick(flag) { this.handleDoubleClick = flag; },

      /** @summary Is double-click handler activated */
      isHandleDoubleClick() { return this.handleDoubleClick; },

      getOnlyLastCycle() { return this.onlyLastCycle; },

      setOnlyLastCycle(v) { this.onlyLastCycle = v; },

      /** @summary Set reverse order */
      setReverseOrder(on) { this.reverseOrder = on; },

      /** @summary Is reverse order */
      isReverseOrder() { return this.reverseOrder; },

      /** @summary Set full model
        * @desc Method can be used when complete hierarchy is ready and can be used directly */
      setFullModel(topnode) {
         this.fullModel = true;
         if (topnode.length) {
            this.h.nchilds = topnode.length;
            this.h.childs = topnode;
         } else {
            this.h.nchilds = 1;
            this.h.childs = [topnode];
         }
         delete this.h._requested; // reply on top element can be full description

         if (!this.mainModel) {
            this.mainModel = this.h.childs;
            this.mainFullModel = true;
         }

         topnode.expanded = true;
         this.reset_nodes = true;
         delete this.noData;
         this.scanShifts();
         if (this.oBinding)
            this.oBinding.checkUpdate(true);
      },

      /** @summary Clear full model */
      clearFullModel() {
         if (!this.fullModel) return;

         delete this.h.childs;
         delete this.h.nchilds;
         delete this.fullModel;
         delete this.noData;
         this.reset_nodes = true;
         if (this.oBinding)
            this.oBinding.checkUpdate(true);
      },

      /** @summary Set nodata flag */
      setNoData(on) {
         this.noData = on;
         if (this.oBinding)
            this.oBinding.checkUpdate(true);
      },

      /** @summary Create list binding */
      bindTree(sPath, oContext, aFilters, mParameters, aSorters) {
         this.oBinding = new BrowserListBinding(this, sPath, oContext, aFilters, mParameters, aSorters);
         return this.oBinding;
      },

      /** @summary Return length property */
      getLength() {
         return this.noData ? 0 : this.getProperty('/length');
      },

      /** @summary Code index as string */
      codeIndex(indx) {
         return this.useIndexSuffix ? '###' + indx + '$$$' : '';
      },

      /** @summary Extract index from string */
      extractIndex(name) {
         if (!name || typeof name != 'string') return '';
         let p1 = name.lastIndexOf('###'), p2 = name.lastIndexOf('$$$');
         if ((p1 < 0) || (p2 < 0) || (p1 >= p2) || (p2 != name.length - 3)) return name;

         let indx = parseInt(name.slice(p1 + 3, p2));
         if (isNaN(indx) || (indx < 0)) return name;
         return { n: name.slice(0, p1), i: indx };
      },

      /** @summary Get node by path which is array of strings, optionally includes indices */
      getNodeByPath(path) {
         let curr = this.h;
         if (!path || (path.length == 0)) return curr;

         let names = path.slice(); // make copy to avoid changes in argument

         while (names.length > 0) {
            let name = this.extractIndex(names.shift()), find = false, indx = -1;

            if (typeof name != 'string') {
               indx = name.i;
               name = name.n;
            }

            if (indx >= 0) {
               indx -= (curr.first || 0);
               if (curr.childs[indx] && (curr.childs[indx].name == name)) {
                  curr = curr.childs[indx];
                  find = true;
               }
            }

            for (let k = 0; !find && (k < curr.childs.length); ++k) {
               if (curr.childs[k].name == name) {
                  curr = curr.childs[k];
                  find = true;
               }
            }

            if (!find) return null;
         }
         return curr;
      },

      /** @summary expand node by given path
        * @desc When path not exists - try to send request */
      expandNodeByPath(path) {
         if (!path || (typeof path !== 'string') || (path == '/')) return -1;

         let names = path.split('/'), curr = this.h, currpath = [];

         while (names.length > 0) {
            let name = names.shift(), find = false;
            if (!name) continue; // ignore start or stop slash

            if (!curr.childs) {
               // request childs for current element
               // TODO: we do not know child index, but simply can supply search child as argument
               if (!this.fullModel && curr.nchilds && (curr.nchilds > 0)) {
                  curr.expanded = true;
                  this.reset_nodes = true;
                  this._expanding_path = path;
                  this.submitRequest(false, curr, currpath, 'expanding');
               }
               return -1;
            }

            for (let k = 0; k < curr.childs.length; ++k) {
               if (curr.childs[k].name == name) {
                  this.reset_nodes = true;
                  curr.expanded = true;
                  this.processExpandedList(currpath, 'add');
                  curr = curr.childs[k];
                  find = true;
                  break;
               }
            }

            if (!find) return -1;

            currpath.push(curr.name);
         }

         return this.scanShifts(curr);
      },

      /** @summary Assign web socket and submit first request */
      sendFirstRequest(websocket) {
         this._websocket = websocket;
         // submit top-level request already when construct model
         this.submitRequest(false, this.h);
      },

      /** @summary Reload main model
        * @desc One can force to submit new request while settings were changed
        * One also can force to reload items on the server side if they can be potentially changed */
      reloadMainModel(force_request, force_reload, path) {
         if (this.mainModel && !force_request && !force_reload) {
            this.h.nchilds = this.mainModel.length;
            this.h.childs = this.mainModel;
            this.h.expanded = true;
            this.reset_nodes = true;
            this.fullModel = this.mainFullModel;
            delete this.noData;
            this.scanShifts();
            if (this.oBinding)
               this.oBinding.checkUpdate(true);
         } else if (!this.fullModel) {
            if (force_request)
               this.processExpandedList([], 'cleanup');

            // send request, content will be reassigned
            this.submitRequest(force_reload, this.h, path);
         }
      },

      /** @summary submit next request to the server
        * @param {Array} path - path as array of strings
        * @desc directly use web socket, later can be dedicated channel */
      submitRequest(force_reload, elem, path, first, number) {
         if (!this._websocket || elem._requested || this.fullModel) return;

         if (first === 'expanding')
            first = 0;
         else
            delete this._expanding_path;

         elem._requested = true;

         this.loadDataCounter++;

         let regex = this.itemsFilter || '', specials = /[\*^\(\)\?]/;

         if (regex && !specials.test(regex))
            regex = `^(${regex}.*)$`;

         let request = {
            path: path || [],
            first: first || 0,
            number: number || this.threshold || 100,
            sort: this.sortMethod || '',
            reverse: this.reverseOrder || false,
            hidden: this.showHidden ? true : false,
            reload: force_reload ? true : false,  // re-scan items by server even when path was not changed
            regex
         };
         this._websocket.send('BRREQ:' + JSON.stringify(request));
      },

      /** @summary process reply from server
        * @desc In the future reply will be received via websocket channel */
      processResponse(reply) {

         this.loadDataCounter--;

         let elem = this.getNodeByPath(reply.path);

         if (!elem)
            return console.error(`did not found ${reply.path}`);

         if (!elem._requested) console.error(`element ${reply.path} was not requested!!!!`);
         delete elem._requested;

         let smart_merge = false;

         if ((elem.nchilds === reply.nchilds) && elem.childs && reply.nodes) {
            if (elem.first + elem.childs.length == reply.first) {
               elem.childs = elem.childs.concat(reply.nodes);
               smart_merge = true;
            } else if (reply.first + reply.nodes.length == elem.first) {
               elem.first = reply.first;
               elem.childs = reply.nodes.concat(elem.childs);
               smart_merge = true;
            }
         }

         if (!smart_merge) {
            elem.nchilds = reply.nchilds;
            elem.childs = reply.nodes;
            elem.first = reply.first || 0;
         }

         // remember main model
         if ((reply.path.length === 0) && !this.mainModel) {
            this.mainModel = elem.childs;
            this.mainFullModel = false;
         }

         this.scanShifts();

         // reset existing nodes if reply does not match with expectation
         if (!smart_merge)
            this.reset_nodes = true;

         if (this.loadDataCounter == 0)
            if (this.oBinding)
               this.oBinding.checkUpdate(true);

         if (this._expanding_path) {
            let d = this._expanding_path;
            delete this._expanding_path;
            let index = this.expandNodeByPath(d);
            if ((index > 0) && this.treeTable) {
               this.treeTable.setFirstVisibleRow(Math.max(0, index - Math.round(this.treeTable.getVisibleRowCount() / 2)));
               this.refresh(true);
            }
         }

      },

      /** @summary Return node by index */
      getNodeByIndex(indx) {
         let nodes = this.getProperty('/nodes');
         return nodes ? nodes[indx] : null;
      },

      /** @summary return element of hierarchical structure by TreeTable index */
      getElementByIndex(indx) {
         let node = this.getNodeByIndex(indx);
         return node ? node._elem : null;
      },

      /** @summary function used to calculate all ids shifts and total number of elements
        * @desc if element specified - returns index of that element */
      scanShifts(for_elem) {

         let id = 0, full = this.fullModel, res = -1;

         function scan(lvl, elem) {

            if (elem === for_elem) res = id;

            if (lvl >= 0) id++;

            let before_id = id;

            if (elem.expanded) {
               if (elem.childs === undefined) {
                  // do nothing, childs are not visible as long as we do not have any list

                  // id += 0;
               } else {

                  // gap at the begin
                  if (!full && elem.first)
                     id += elem.first;

                  // jump over all childs
                  for (let k = 0; k < elem.childs.length; ++k)
                     scan(lvl + 1, elem.childs[k]);

                  // gap at the end
                  if (!full) {
                     let _last = (elem.first || 0) + elem.childs.length,
                        _remains = elem.nchilds - _last;
                     if (_remains > 0) id += _remains;
                  }
               }
            }

            // this shift can be later applied to jump over all elements
            elem._shift = id - before_id;
         }

         scan(-1, this.h);

         this.setProperty('/length', id);

         return for_elem ? res : id;
      },

      /** @summary main  method to create flat list of nodes - only whose which are specified in selection
        * @desc following arguments:
        *    args.begin     - first visible element from flat list
        *    args.end       - first none-visible element
        *    args.threshold - extra elements (before/after) which probably should be prefetched
        * @return holder object with all existing nodes */
      buildFlatNodes(args) {

         if (this.noData) return null;

         let id = 0,            // current id, at the same time number of items
            threshold = args.threshold || this.threshold || 100,
            threshold2 = Math.round(threshold / 2),
            nodes = this.reset_nodes ? {} : this.getProperty('/nodes');

         // main method to scan through all existing sub-folders
         let scan = (lvl, elem, path) => {

            // create elements with safety margin
            if ((lvl >= 0) && (nodes !== null) && (id >= args.begin - threshold2) && (id < args.end + threshold2)) {
               if (!nodes[id])
                  nodes[id] = {
                     name: elem.name,
                     path: path.slice(), // make array copy
                     index: id,
                     _elem: elem,
                     isLeaf: !elem.nchilds,
                     // these are required by list binding, should be eliminated in the future
                     type: elem.nchilds ? 'folder' : 'file',
                     level: lvl,
                     context: this.getContext('/nodes/' + id),
                     nodeState: {
                        expanded: !!elem.expanded,
                        selected: !!elem.selected,
                        sum: false // ????
                     }
                  };
               else {
                  nodes[id].nodeState.expanded = !!elem.expanded;
                  nodes[id].nodeState.selected = !!elem.selected;
               }

               // always provide nodes attributes
               if (typeof this.addNodeAttributes == 'function')
                  this.addNodeAttributes(nodes[id], elem);
            }

            if (lvl >= 0) id++;

            if (!elem.expanded && !this.processExpandedList(path, 'test')) return;

            elem.expanded = true;

            if (elem.childs === undefined) {
               // add new request - can we check if only special part of childs is required?

               // TODO: probably one could guess more precise request
               if ((elem.nchilds === undefined) || (elem.nchilds !== 0))
                  this.submitRequest(false, elem, path);

               return;
            }

            // check if scan is required
            if (((id + elem._shift) < args.begin - threshold2) || (id >= args.end + threshold2)) {
               id += elem._shift;
               return;
            }

            // when not all childs from very beginning is loaded, but may be required
            if (elem.first && !this.fullModel) {

               // check if requests are needed to load part in the begin of the list
               if (args.begin - id - threshold2 < elem.first) {

                  let first = Math.max(args.begin - id - threshold2, 0),
                     number = Math.min(elem.first - first, threshold);

                  this.submitRequest(false, elem, path, first, number);
               }

               id += elem.first;
            }


            let subpath = path.slice(), // make copy to avoid changes in argument
               subindx = subpath.push('') - 1;

            for (let k = 0; k < elem.childs.length; ++k) {
               subpath[subindx] = elem.childs[k].name + this.codeIndex((elem.first || 0) + k);
               scan(lvl + 1, elem.childs[k], subpath);
            }

            // check if more elements are required

            if (!this.fullModel) {
               let _last = (elem.first || 0) + elem.childs.length,
                  _remains = elem.nchilds - _last;

               if (_remains > 0) {
                  if (args.end + threshold2 > id) {

                     let first = _last, number = args.end + threshold2 - id;
                     if (number < threshold) number = threshold; // always request much
                     if (number > _remains) number = _remains; // but not too much
                     if (number > threshold) {
                        first += (number - threshold);
                        number = threshold;
                     }

                     this.submitRequest(false, elem, path, first, number);
                  }

                  id += _remains;
               }
            }
         }

         // start scan from very top
         scan(-1, this.h, []);

         if (this.getProperty('/length') != id)
            this.setProperty('/length', id); // update length property

         if (this.reset_nodes) {
            this.setProperty('/nodes', nodes);
            delete this.reset_nodes;
         }

         return nodes;
      },

      processExpandedList(path, action) {
         if (action == 'cleanup') {
            delete this._last_expands;
            return true;
         }

         const exact = action != 'remove', len = path?.length;
         if (!len) return false;

         if (!this._last_expands)
            this._last_expands = [];
         for (let n = 0; n < this._last_expands.length; ++n) {
            let test = this._last_expands[n], match = true;
            if ((len > test.length) || (exact && (len != test.length)))
               continue;
            for (let k = 0; k < len; ++k)
               if (test[k] != path[k]) {
                  match = false;
                  break;
               }

            if (match) {
               if (action == 'remove')
                  this._last_expands.splice(n--, 1);
               else
                  return true;
            }
         }
         if (action == 'add')
            this._last_expands.push(path.slice());
         return false;
      },

      /** @summary toggle expand state of specified node */
      toggleNode(index, do_expand) {
         let node = this.getNodeByIndex(index),
            elem = this.getElementByIndex(index);

         if (!node || !elem) return;

         if (elem.expanded && (do_expand === false)) {
            this.processExpandedList(node.path, 'remove');
            delete elem.expanded;
            if (!this.fullModel)
               delete elem.childs; // TODO: for the future keep childs but make request if expand once again

            // close folder - reassign shifts
            this.reset_nodes = true;
            this.scanShifts();

            return true;

         } else if ((elem.nchilds || !elem.index) && (do_expand === true)) {
            this.processExpandedList(node.path, 'add');

            elem.expanded = true;
            // structure is changing but not immediately

            if (this.fullModel) {
               this.reset_nodes = true;
               this.scanShifts();
            }

            return true;
         }
      },

      /** @summary Change items filter */
      setItemsFilter(newValue) {
         if (newValue === undefined)
            newValue = this.getProperty('/itemsFilter') || '';

         // ignore same value
         if (newValue === this.itemsFilter)
            return;

         this.itemsFilter = newValue;

         // now we should request values once again
         this.submitRequest(false, this.h);
      }

   });

   return hRootModel;

});
