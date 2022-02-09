/// @file JSRoot.hierarchy.js
/// Hierarchy display functionality

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   // ===========================================================================================

   /** @summary draw list content
     * @desc used to draw all items from TList or TObjArray inserted into the TCanvas list of primitives
     * @memberof JSROOT.Painter
     * @private */
   function drawList(dom, lst, opt) {
      if (!lst || !lst.arr) return Promise.resolve(null);

      let obj = {
        divid: dom,
        lst: lst,
        opt: opt,
        indx: -1,
        painter: null,
        draw_next: function() {
           while (++this.indx < this.lst.arr.length) {
              let item = this.lst.arr[this.indx],
                  opt = (this.lst.opt && this.lst.opt[this.indx]) ? this.lst.opt[this.indx] : this.opt;
              if (!item) continue;
              return JSROOT.draw(this.getDom(), item, opt).then(p => {
                 if (p && !this.painter) this.painter = p;
                 return this.draw_next(); // reenter loop
              });
           }

           return Promise.resolve(this.painter);
        }
      }

      return obj.draw_next();
   }

   // ===================== hierarchy scanning functions ==================================

   /** @summary Create hierarchy elements for TFolder object
     * @memberof JSROOT.Painter
     * @private */
   function folderHierarchy(item, obj) {

      if (!obj || !('fFolders' in obj) || (obj.fFolders===null)) return false;

      if (obj.fFolders.arr.length===0) { item._more = false; return true; }

      item._childs = [];

      for ( let i = 0; i < obj.fFolders.arr.length; ++i) {
         let chld = obj.fFolders.arr[i];
         item._childs.push( {
            _name : chld.fName,
            _kind : "ROOT." + chld._typename,
            _obj : chld
         });
      }
      return true;
   }

   /** @summary Create hierarchy elements for TTask object
     * @memberof JSROOT.Painter
     * @private */
   function taskHierarchy(item, obj) {
      // function can be used for different derived classes
      // we show not only child tasks, but all complex data members

      if (!obj || !('fTasks' in obj) || (obj.fTasks === null)) return false;

      objectHierarchy(item, obj, { exclude: ['fTasks', 'fName'] } );

      if ((obj.fTasks.arr.length===0) && (item._childs.length==0)) { item._more = false; return true; }

      // item._childs = [];

      for ( let i = 0; i < obj.fTasks.arr.length; ++i) {
         let chld = obj.fTasks.arr[i];
         item._childs.push({
            _name : chld.fName,
            _kind : "ROOT." + chld._typename,
            _obj : chld
         });
      }
      return true;
   }

   /** @summary Create hierarchy elements for TList object
     * @memberof JSROOT.Painter
     * @private */
   function listHierarchy(folder, lst) {
      if (!JSROOT.isRootCollection(lst)) return false;

      if ((lst.arr === undefined) || (lst.arr.length === 0)) {
         folder._more = false;
         return true;
      }

      let do_context = false, prnt = folder;
      while (prnt) {
         if (prnt._do_context) do_context = true;
         prnt = prnt._parent;
      }

      // if list has objects with similar names, create cycle number for them
      let ismap = (lst._typename == 'TMap'), names = [], cnt = [], cycle = [];

      for (let i = 0; i < lst.arr.length; ++i) {
         let obj = ismap ? lst.arr[i].first : lst.arr[i];
         if (!obj) continue; // for such objects index will be used as name
         let objname = obj.fName || obj.name;
         if (!objname) continue;
         let indx = names.indexOf(objname);
         if (indx>=0) {
            cnt[indx]++;
         } else {
            cnt[names.length] = cycle[names.length] = 1;
            names.push(objname);
         }
      }

      folder._childs = [];
      for (let i = 0; i < lst.arr.length; ++i) {
         let obj = ismap ? lst.arr[i].first : lst.arr[i],
             item = !obj || !obj._typename ?
              {
               _name: i.toString(),
               _kind: "ROOT.NULL",
               _title: "NULL",
               _value: "null",
               _obj: null
             } : {
               _name: obj.fName || obj.name,
               _kind: "ROOT." + obj._typename,
               _title: (obj.fTitle || "") + " type:"  +  obj._typename,
               _obj: obj
           };

           switch(obj._typename) {
              case 'TColor': item._value = jsrp.getRGBfromTColor(obj); break;
              case 'TText':
              case 'TLatex': item._value = obj.fTitle; break;
              case 'TObjString': item._value = obj.fString; break;
              default: if (lst.opt && lst.opt[i] && lst.opt[i].length) item._value = lst.opt[i];
           }

           if (do_context && jsrp.canDraw(obj._typename)) item._direct_context = true;

           // if name is integer value, it should match array index
           if (!item._name || (Number.isInteger(parseInt(item._name)) && (parseInt(item._name) !== i))
               || (lst.arr.indexOf(obj) < i)) {
              item._name = i.toString();
           } else {
              // if there are several such names, add cycle number to the item name
              let indx = names.indexOf(obj.fName);
              if ((indx >= 0) && (cnt[indx] > 1)) {
                 item._cycle = cycle[indx]++;
                 item._keyname = item._name;
                 item._name = item._keyname + ";" + item._cycle;
              }
           }

         folder._childs.push(item);
      }
      return true;
   }

   /** @summary Create hierarchy of TKey lists in file or sub-directory
     * @memberof JSROOT.Painter
     * @private */
   function keysHierarchy(folder, keys, file, dirname) {

      if (keys === undefined) return false;

      folder._childs = [];

      for (let i = 0; i < keys.length; ++i) {
         let key = keys[i],
             item = {
            _name: key.fName + ";" + key.fCycle,
            _cycle: key.fCycle,
            _kind: "ROOT." + key.fClassName,
            _title: key.fTitle,
            _keyname: key.fName,
            _readobj: null,
            _parent: folder
         };

         if (key.fObjlen > 1e5) item._title += ' (size: ' + (key.fObjlen/1e6).toFixed(1) + 'MB)';

         if ('fRealName' in key)
            item._realname = key.fRealName + ";" + key.fCycle;

         if (key.fClassName == 'TDirectory' || key.fClassName == 'TDirectoryFile') {
            let dir = null;
            if (dirname && file) dir = file.getDir(dirname + key.fName);
            if (!dir) {
               item._more = true;
               item._expand = function(node, obj) {
                  // one can get expand call from child objects - ignore them
                  return keysHierarchy(node, obj.fKeys);
               };
            } else {
               // remove cycle number - we have already directory
               item._name = key.fName;
               keysHierarchy(item, dir.fKeys, file, dirname + key.fName + "/");
            }
         } else if ((key.fClassName == 'TList') && (key.fName == 'StreamerInfo')) {
            if (JSROOT.settings.SkipStreamerInfos) continue;
            item._name = 'StreamerInfo';
            item._kind = "ROOT.TStreamerInfoList";
            item._title = "List of streamer infos for binary I/O";
            item._readobj = file.fStreamerInfos;
         }

         folder._childs.push(item);
      }

      return true;
   }

   /** @summary Create hierarchy for arbitrary object
     * @memberof JSROOT.Painter
     * @private */
   function objectHierarchy(top, obj, args) {
      if (!top || (obj===null)) return false;

      top._childs = [];

      let proto = Object.prototype.toString.apply(obj);

      if (proto === '[object DataView]') {

         let item = {
             _parent: top,
             _name: 'size',
             _value: obj.byteLength.toString(),
             _vclass: 'h_value_num'
         };

         top._childs.push(item);
         let namelen = (obj.byteLength < 10) ? 1 : Math.log10(obj.byteLength);

         for (let k = 0; k < obj.byteLength; ++k) {
            if (k % 16 === 0) {
               item = {
                 _parent: top,
                 _name: k.toString(),
                 _value: "",
                 _vclass: 'h_value_num'
               };
               while (item._name.length < namelen) item._name = "0" + item._name;
               top._childs.push(item);
            }

            let val = obj.getUint8(k).toString(16);
            while (val.length<2) val = "0"+val;
            if (item._value.length>0)
               item._value += (k%4===0) ? " | " : " ";

            item._value += val;
         }
         return true;
      }

      // check nosimple property in all parents
      let nosimple = true, do_context = false, prnt = top;
      while (prnt) {
         if (prnt._do_context) do_context = true;
         if ('_nosimple' in prnt) { nosimple = prnt._nosimple; break; }
         prnt = prnt._parent;
      }

      let isarray = (JSROOT._.is_array_proto(proto) > 0) && obj.length,
          compress = isarray && (obj.length > JSROOT.settings.HierarchyLimit),  arrcompress = false;

      if (isarray && (top._name==="Object") && !top._parent) top._name = "Array";

      if (compress) {
         arrcompress = true;
         for (let k=0;k<obj.length;++k) {
            let typ = typeof obj[k];
            if ((typ === 'number') || (typ === 'boolean') || (typ=='string' && (obj[k].length<16))) continue;
            arrcompress = false; break;
         }
      }

      if (!('_obj' in top))
         top._obj = obj;
      else if (top._obj !== obj)
         alert('object missmatch');

      if (!top._title) {
         if (obj._typename)
            top._title = "ROOT." + obj._typename;
         else if (isarray)
            top._title = "Array len: " + obj.length;
      }

      if (arrcompress) {
         for (let k = 0; k < obj.length;) {

            let nextk = Math.min(k+10,obj.length), allsame = true, prevk = k;

            while (allsame) {
               allsame = true;
               for (let d=prevk;d<nextk;++d)
                  if (obj[k]!==obj[d]) allsame = false;

               if (allsame) {
                  if (nextk===obj.length) break;
                  prevk = nextk;
                  nextk = Math.min(nextk+10,obj.length);
               } else if (prevk !== k) {
                  // last block with similar
                  nextk = prevk;
                  allsame = true;
                  break;
               }
            }

            let item = { _parent: top, _name: k+".."+(nextk-1), _vclass: 'h_value_num' };

            if (allsame) {
               item._value = obj[k].toString();
            } else {
               item._value = "";
               for (let d=k;d<nextk;++d)
                  item._value += ((d===k) ? "[ " : ", ") + obj[d].toString();
               item._value += " ]";
            }

            top._childs.push(item);

            k = nextk;
         }
         return true;
      }

      let lastitem, lastkey, lastfield, cnt;

      for (let key in obj) {
         if ((key == '_typename') || (key[0]=='$')) continue;
         let fld = obj[key];
         if (typeof fld == 'function') continue;
         if (args && args.exclude && (args.exclude.indexOf(key)>=0)) continue;

         if (compress && lastitem) {
            if (lastfield===fld) { ++cnt; lastkey = key; continue; }
            if (cnt > 0) lastitem._name += ".." + lastkey;
         }

         let item = { _parent: top, _name: key };

         if (compress) { lastitem = item;  lastkey = key; lastfield = fld; cnt = 0; }

         if (fld === null) {
            item._value = item._title = "null";
            if (!nosimple) top._childs.push(item);
            continue;
         }

         let simple = false;

         if (typeof fld == 'object') {

            proto = Object.prototype.toString.apply(fld);

            if (JSROOT._.is_array_proto(proto) > 0) {
               item._title = "array len=" + fld.length;
               simple = (proto != '[object Array]');
               if (fld.length === 0) {
                  item._value = "[ ]";
                  item._more = false; // hpainter will not try to expand again
               } else {
                  item._value = "[...]";
                  item._more = true;
                  item._expand = objectHierarchy;
                  item._obj = fld;
               }
            } else if (proto === "[object DataView]") {
               item._title = 'DataView len=' + fld.byteLength;
               item._value = "[...]";
               item._more = true;
               item._expand = objectHierarchy;
               item._obj = fld;
            } else if (proto === "[object Date]") {
               item._more = false;
               item._title = 'Date';
               item._value = fld.toString();
               item._vclass = 'h_value_num';
            } else {

               if (fld.$kind || fld._typename)
                  item._kind = item._title = "ROOT." + (fld.$kind || fld._typename);

               if (fld._typename) {
                  item._title = fld._typename;
                  if (do_context && jsrp.canDraw(fld._typename)) item._direct_context = true;
               }

               // check if object already shown in hierarchy (circular dependency)
               let curr = top, inparent = false;
               while (curr && !inparent) {
                  inparent = (curr._obj === fld);
                  curr = curr._parent;
               }

               if (inparent) {
                  item._value = "{ prnt }";
                  item._vclass = 'h_value_num';
                  item._more = false;
                  simple = true;
               } else {
                  item._obj = fld;
                  item._more = false;

                  switch(fld._typename) {
                     case 'TColor': item._value = jsrp.getRGBfromTColor(fld); break;
                     case 'TText':
                     case 'TLatex': item._value = fld.fTitle; break;
                     case 'TObjString': item._value = fld.fString; break;
                     default:
                        if (JSROOT.isRootCollection(fld) && (typeof fld.arr === "object")) {
                           item._value = fld.arr.length ? "[...]" : "[]";
                           item._title += ", size:"  + fld.arr.length;
                           if (fld.arr.length>0) item._more = true;
                        } else {
                           item._more = true;
                           item._value = "{ }";
                        }
                  }
               }
            }
         } else if ((typeof fld === 'number') || (typeof fld === 'boolean')) {
            simple = true;
            if (key == 'fBits')
               item._value = "0x" + fld.toString(16);
            else
               item._value = fld.toString();
            item._vclass = 'h_value_num';
         } else if (typeof fld === 'string') {
            simple = true;
            item._value = '&quot;' + fld.replace(/\&/g, '&amp;').replace(/\"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '&quot;';
            item._vclass = 'h_value_str';
         } else if (typeof fld === 'undefined') {
            simple = true;
            item._value = "undefined";
            item._vclass = 'h_value_num';
         } else {
            simple = true;
            alert('miss ' + key + '  ' + typeof fld);
         }

         if (!simple || !nosimple)
            top._childs.push(item);
      }

      if (compress && lastitem && (cnt>0)) lastitem._name += ".." + lastkey;

      return true;
   }

   /** @summary Create hierarchy for streamer info object
     * @memberof JSROOT.Painter
     * @private */
   function createStreamerInfoContent(lst) {
      let h = { _name : "StreamerInfo", _childs : [] };

      for (let i = 0; i < lst.arr.length; ++i) {
         let entry = lst.arr[i];

         if (entry._typename == "TList") continue;

         if (typeof entry.fName == 'undefined') {
            console.warn(`strange element in StreamerInfo with type ${entry._typename}`);
            continue;
         }

         let item = {
            _name : entry.fName + ";" + entry.fClassVersion,
            _kind : "class " + entry.fName,
            _title : "class:" + entry.fName + ' version:' + entry.fClassVersion + ' checksum:' + entry.fCheckSum,
            _icon: "img_class",
            _childs : []
         };

         if (entry.fTitle != '') item._title += '  ' + entry.fTitle;

         h._childs.push(item);

         if (typeof entry.fElements == 'undefined') continue;
         for ( let l = 0; l < entry.fElements.arr.length; ++l) {
            let elem = entry.fElements.arr[l];
            if (!elem || !elem.fName) continue;
            let info = elem.fTypeName + " " + elem.fName,
                title = elem.fTypeName + " type:" + elem.fType;
            if (elem.fArrayDim===1)
               info += "[" + elem.fArrayLength + "]";
            else
               for (let dim = 0; dim < elem.fArrayDim; ++dim)
                  info += "[" + elem.fMaxIndex[dim] + "]";
            if (elem.fBaseVersion === 4294967295) info += ":-1"; else
            if (elem.fBaseVersion !== undefined) info += ":" + elem.fBaseVersion;
            info += ";";
            if (elem.fTitle) info += " // " + elem.fTitle;

            item._childs.push({ _name : info, _title: title, _kind: elem.fTypeName, _icon: (elem.fTypeName == 'BASE') ? "img_class" : "img_member" });
         }
         if (!item._childs.length) delete item._childs;
      }

      return h;
   }

   /** @summary Create hierarchy for object inspector
     * @memberof JSROOT.Painter
     * @private */
   function createInspectorContent(obj) {
      let h = { _name: "Object", _title: "", _click_action: "expand", _nosimple: false, _do_context: true };

      if ((typeof obj.fName === 'string') && (obj.fName.length > 0))
         h._name = obj.fName;

      if ((typeof obj.fTitle === 'string') && (obj.fTitle.length > 0))
         h._title = obj.fTitle;

      if (obj._typename)
         h._title += "  type:" + obj._typename;

      if (JSROOT.isRootCollection(obj)) {
         h._name = obj.name || obj._typename;
         listHierarchy(h, obj);
      } else {
         objectHierarchy(h, obj);
      }

      return h;
   }


   /** @summary Parse string value as array.
    * @desc It could be just simple string:  "value" or
    * array with or without string quotes:  [element], ['elem1',elem2]
    * @private */
   let parseAsArray = val => {

      let res = [];

      if (typeof val != 'string') return res;

      val = val.trim();
      if (val=="") return res;

      // return as array with single element
      if ((val.length < 2) || (val[0] != '[') || (val[val.length-1] != ']')) {
         res.push(val); return res;
      }

      // try to split ourself, checking quotes and brackets
      let nbr = 0, nquotes = 0, ndouble = 0, last = 1;

      for (let indx = 1; indx < val.length; ++indx) {
         if (nquotes > 0) {
            if (val[indx]==="'") nquotes--;
            continue;
         }
         if (ndouble > 0) {
            if (val[indx]==='"') ndouble--;
            continue;
         }
         switch (val[indx]) {
            case "'": nquotes++; break;
            case '"': ndouble++; break;
            case "[": nbr++; break;
            case "]": if (indx < val.length - 1) { nbr--; break; }
            case ",":
               if (nbr === 0) {
                  let sub = val.substring(last, indx).trim();
                  if ((sub.length > 1) && (sub[0] == sub[sub.length-1]) && ((sub[0] == '"') || (sub[0] == "'")))
                     sub = sub.substr(1, sub.length-2);
                  res.push(sub);
                  last = indx+1;
               }
               break;
         }
      }

      if (res.length === 0)
         res.push(val.substr(1, val.length-2).trim());

      return res;
   }

   // =================================================================================================

   /**
     * @summary Special browser layout
     *
     * @memberof JSROOT
     * @desc Contains three different areas for browser (left), status line (bottom) and central drawing
     * Main application is normal browser in JSROOT, but also used in other applications like ROOT6 canvas
     * @private
     */

   class BrowserLayout {

      /** @summary Constructor */
      constructor(id, hpainter, objpainter) {
         this.gui_div = id;
         this.hpainter = hpainter; // painter for brwoser area (if any)
         this.objpainter = objpainter; // painter for object area (if any)
         this.browser_kind = null; // should be 'float' or 'fix'
      }

      /** @summary Selects main element */
      main() {
         return d3.select("#" + this.gui_div);
      }

      /** @summary Returns drawing divid */
      drawing_divid() {
         return this.gui_div + "_drawing";
      }

      /** @summary Check resize action */
      checkResize() {
         if (this.hpainter && (typeof this.hpainter.checkResize == 'function'))
            this.hpainter.checkResize();
         else if (this.objpainter && (typeof this.objpainter.checkResize == 'function')) {
            this.objpainter.checkResize(true);
         }
      }

      /** @summary method used to create basic elements
        * @desc should be called only once */
      create(with_browser) {
         let main = this.main();

         main.append("div").attr("id", this.drawing_divid())
                           .classed("jsroot_draw_area", true)
                           .style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('right',0);

         if (with_browser) main.append("div").classed("jsroot_browser", true);
      }

      /** @summary Create buttons in the layout */
      createBrowserBtns() {
         let br = this.main().select(".jsroot_browser");
         if (br.empty()) return;
         let btns = br.append("div").classed("jsroot_browser_btns", true).classed("jsroot", true);
         btns.style('position',"absolute").style("left","7px").style("top","7px");
         if (JSROOT.browser.touches) btns.style('opacity','0.2'); // on touch devices should be always visible
         return btns;
      }

      /** @summary Remove browser buttons */
      removeBrowserBtns() {
         this.main().select(".jsroot_browser").select(".jsroot_browser_btns").remove();
      }

      /** @summary Set browser content */
      setBrowserContent(guiCode) {
         let main = d3.select("#" + this.gui_div + " .jsroot_browser");
         if (main.empty()) return;

         main.insert('div', ".jsroot_browser_btns").classed('jsroot_browser_area', true)
             .style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('width','250px')
             .style('overflow', 'hidden')
             .style('padding-left','5px')
             .style('display','flex').style('flex-direction', 'column')   /* use the flex model */
             .html("<p class='jsroot_browser_title'>title</p><div class='jsroot_browser_resize' style='display:none'>&#9727</div>" + guiCode);
      }

      /** @summary Check if there is browser content */
      hasContent() {
         let main = d3.select("#" + this.gui_div + " .jsroot_browser");
         if (main.empty()) return false;
         return !main.select(".jsroot_browser_area").empty();
      }

      /** @summary Delete content */
      deleteContent() {
         let main = d3.select("#" + this.gui_div + " .jsroot_browser");
         if (main.empty()) return;

         this.createStatusLine(0, "delete");

         this.toggleBrowserVisisbility(true);

         main.selectAll("*").remove();
         delete this.browser_visible;
         delete this.browser_kind;

         this.checkResize();
      }

      /** @summary Returns true when status line exists */
      hasStatus() {
         let main = d3.select("#"+this.gui_div+" .jsroot_browser");
         if (main.empty()) return false;

         let id = this.gui_div + "_status",
             line = d3.select("#"+id);

         return !line.empty();
      }

      /** @summary Set browser title text
        * @desc Title also used for dragging of the float browser */
      setBrowserTitle(title) {
         let main = d3.select("#" + this.gui_div + " .jsroot_browser");
         if (!main.empty())
            main.select(".jsroot_browser_title").text(title).style('cursor',this.browser_kind == 'flex' ? "move" : null);
      }

      /** @summary Toggle browser kind
        * @desc used together with browser buttons */
      toggleKind(browser_kind) {
         if (this.browser_visible!=='changing') {
            if (browser_kind === this.browser_kind) this.toggleBrowserVisisbility();
                                               else this.toggleBrowserKind(browser_kind);
         }
      }

      /** @summary Creates status line */
      createStatusLine(height, mode) {

         let main = d3.select("#"+this.gui_div+" .jsroot_browser");
         if (main.empty())
            return Promise.resolve('');

         let id = this.gui_div + "_status",
             line = d3.select("#"+id),
             is_visible = !line.empty();

         if (mode==="toggle") { mode = !is_visible; } else
         if (mode==="delete") { mode = false; height = 0; delete this.status_layout; } else
         if (mode===undefined) { mode = true; this.status_layout = "app"; }

         if (is_visible) {
            if (mode === true)
               return Promise.resolve(id);

            let hsepar = main.select(".jsroot_h_separator");

            hsepar.remove();
            line.remove();

            if (this.status_layout !== "app")
               delete this.status_layout;

            if (this.status_handler && (jsrp.showStatus === this.status_handler)) {
               delete jsrp.showStatus;
               delete this.status_handler;
            }

            this.adjustSeparators(null, 0, true);
            return Promise.resolve("");
         }

         if (mode === false)
            return Promise.resolve("");

         let left_pos = d3.select("#" + this.gui_div + "_drawing").style('left');

         main.insert("div",".jsroot_browser_area")
             .attr("id",id)
             .classed("jsroot_status_area", true)
             .style('position',"absolute").style('left',left_pos).style('height',"20px").style('bottom',0).style('right',0)
             .style('margin',0).style('border',0);

         let hsepar = main.insert("div",".jsroot_browser_area")
                          .classed("jsroot_separator", true).classed("jsroot_h_separator", true)
                          .style('position','absolute').style('left',left_pos).style('right',0).style('bottom','20px').style('height','5px');

         let drag_move = d3.drag().on("start", () => {
             this._hsepar_move = this._hsepar_position;
             hsepar.style('background-color', 'grey');
         }).on("drag", evnt => {
             this._hsepar_move -= evnt.dy; // hsepar is position from bottom
             this.adjustSeparators(null, Math.max(5, Math.round(this._hsepar_move)));
         }).on("end", () => {
             delete this._hsepar_move;
             hsepar.style('background-color', null);
             this.checkResize();
         });

         hsepar.call(drag_move);

         // need to get touches events handling in drag
         if (JSROOT.browser.touches && !main.on("touchmove"))
            main.on("touchmove", function() { });

         if (!height || (typeof height === 'string')) height = this.last_hsepar_height || 20;

         this.adjustSeparators(null, height, true);

         if (this.status_layout == "app")
            return Promise.resolve(id);

         this.status_layout = new JSROOT.GridDisplay(id, 'horizx4_1213');

         let frame_titles = ['object name','object title','mouse coordinates','object info'];
         for (let k = 0; k < 4; ++k)
            d3.select(this.status_layout.getGridFrame(k))
              .attr('title', frame_titles[k]).style('overflow','hidden')
              .append("label").attr("class","jsroot_status_label");

         this.status_handler = this.showStatus.bind(this);

         jsrp.showStatus = this.status_handler;

         return Promise.resolve(id);
      }

      /** @summary Adjust separator positions */
      adjustSeparators(vsepar, hsepar, redraw, first_time) {

         if (!this.gui_div) return;

         let main = d3.select("#" + this.gui_div + " .jsroot_browser"), w = 5;

         if ((hsepar===null) && first_time && !main.select(".jsroot_h_separator").empty()) {
            // if separator set for the first time, check if status line present
            hsepar = main.select(".jsroot_h_separator").style('bottom');
            if ((typeof hsepar=='string') && (hsepar.length > 2) && (hsepar.indexOf('px') == hsepar.length-2))
               hsepar = hsepar.substr(0,hsepar.length-2);
            else
               hsepar = null;
         }

         if (hsepar!==null) {
            hsepar = parseInt(hsepar);
            let elem = main.select(".jsroot_h_separator"), hlimit = 0;

            if (!elem.empty()) {
               if (hsepar < 5) hsepar = 5;

               let maxh = main.node().clientHeight - w;
               if (maxh > 0) {
                  if (hsepar < 0) hsepar += maxh;
                  if (hsepar > maxh) hsepar = maxh;
               }

               this.last_hsepar_height = hsepar;
               elem.style('bottom', hsepar+'px').style('height', w+'px');
               d3.select("#" + this.gui_div + "_status").style('height', hsepar+'px');
               hlimit = (hsepar+w) + 'px';
            }

            this._hsepar_position = hsepar;

            d3.select("#" + this.gui_div + "_drawing").style('bottom',hlimit);
         }

         if (vsepar!==null) {
            vsepar = parseInt(vsepar);
            if (vsepar < 50) vsepar = 50;
            this._vsepar_position = vsepar;
            main.select(".jsroot_browser_area").style('width',(vsepar-5)+'px');
            d3.select("#" + this.gui_div + "_drawing").style('left',(vsepar+w)+'px');
            main.select(".jsroot_h_separator").style('left', (vsepar+w)+'px');
            d3.select("#" + this.gui_div + "_status").style('left',(vsepar+w)+'px');
            main.select(".jsroot_v_separator").style('left',vsepar+'px').style('width',w+"px");
         }

         if (redraw) this.checkResize();
      }

      /** @summary Show status information inside special fields of browser layout */
      showStatus(/*name, title, info, coordinates*/) {
         if (!this.status_layout) return;

         let maxh = 0;
         for (let n = 0; n < 4; ++n) {
            let lbl = this.status_layout.getGridFrame(n).querySelector('label');
            maxh = Math.max(maxh, lbl.clientHeight);
            lbl.innerHTML = arguments[n] || "";
         }

         if (!this.status_layout.first_check) {
            this.status_layout.first_check = true;
            if ((maxh > 5) && ((maxh > this.last_hsepar_height) || (maxh < this.last_hsepar_height+5)))
               this.adjustSeparators(null, maxh, true);
         }
      }

      /** @summary Toggle browser visibility */
      toggleBrowserVisisbility(fast_close) {
         if (!this.gui_div || (typeof this.browser_visible==='string')) return;

         let main = d3.select("#" + this.gui_div + " .jsroot_browser"),
             area = main.select('.jsroot_browser_area');

         if (area.empty()) return;

         let vsepar = main.select(".jsroot_v_separator"),
             drawing = d3.select("#" + this.gui_div + "_drawing"),
             tgt = area.property('last_left'),
             tgt_separ = area.property('last_vsepar'),
             tgt_drawing = area.property('last_drawing');

         if (!this.browser_visible) {
            if (fast_close) return;
            area.property('last_left', null).property('last_vsepar',null).property('last_drawing', null);
         } else {
            area.property('last_left', area.style('left'));
            if (!vsepar.empty()) {
               area.property('last_vsepar', vsepar.style('left'));
               area.property('last_drawing', drawing.style('left'));
            }

            tgt = (-area.node().clientWidth - 10) + "px";
            let mainw = main.node().clientWidth;

            if (vsepar.empty() && (area.node().offsetLeft > mainw/2))
               tgt = (mainw+10) + "px";

            tgt_separ = "-10px";
            tgt_drawing = "0px";
         }

         let visible_at_the_end  = !this.browser_visible, _duration = fast_close ? 0 : 700;

         this.browser_visible = 'changing';

         area.transition().style('left', tgt).duration(_duration).on("end", () => {
            if (fast_close) return;
            this.browser_visible = visible_at_the_end;
            if (visible_at_the_end) this.setButtonsPosition();
         });

         if (!visible_at_the_end)
            main.select(".jsroot_browser_btns").transition().style('left', '7px').style('top', '7px').duration(_duration);

         if (!vsepar.empty()) {
            vsepar.transition().style('left', tgt_separ).duration(_duration);
            drawing.transition().style('left', tgt_drawing).duration(_duration).on("end", this.checkResize.bind(this));
         }

         if (this.status_layout && (this.browser_kind == 'fix')) {
            main.select(".jsroot_h_separator").transition().style('left', tgt_drawing).duration(_duration);
            main.select(".jsroot_status_area").transition().style('left', tgt_drawing).duration(_duration);
         }
      }

      /** @summary Adjust browser size */
      adjustBrowserSize(onlycheckmax) {
         if (!this.gui_div || (this.browser_kind !== "float")) return;

         let main = d3.select("#" + this.gui_div + " .jsroot_browser");
         if (main.empty()) return;

         let area = main.select(".jsroot_browser_area"),
             cont = main.select(".jsroot_browser_hierarchy"),
             chld = d3.select(cont.node().firstChild);

         if (onlycheckmax) {
            if (area.node().parentNode.clientHeight - 10 < area.node().clientHeight)
               area.style('bottom', '0px').style('top','0px');
            return;
         }

         if (chld.empty()) return;
         let h1 = cont.node().clientHeight,
             h2 = chld.node().clientHeight;

         if ((h2!==undefined) && (h2 < h1*0.7)) area.style('bottom', '');
      }

      /** @summary Set buttons position */
      setButtonsPosition() {
         if (!this.gui_div) return;

         let main = d3.select("#"+this.gui_div+" .jsroot_browser"),
             btns = main.select(".jsroot_browser_btns"),
             top = 7, left = 7;

         if (btns.empty()) return;

         if (this.browser_visible) {
            let area = main.select(".jsroot_browser_area");

            top = area.node().offsetTop + 7;

            left = area.node().offsetLeft - main.node().offsetLeft + area.node().clientWidth - 27;
         }

         btns.style('left', left+'px').style('top', top+'px');
      }

      /** @summary Toggle browser kind */
      toggleBrowserKind(kind) {

         if (!this.gui_div)
            return Promise.resolve(null);

         if (!kind) {
            if (!this.browser_kind)
               return Promise.resolve(null);
            kind = (this.browser_kind === "float") ? "fix" : "float";
         }

         let main = d3.select("#"+this.gui_div+" .jsroot_browser"),
             area = main.select(".jsroot_browser_area");

         if (this.browser_kind === "float") {
             area.style('bottom', '0px')
                 .style('top', '0px')
                 .style('width','').style('height','')
                 .classed('jsroot_float_browser', false);

              //jarea.resizable("destroy")
              //     .draggable("destroy");
         } else if (this.browser_kind === "fix") {
            main.select(".jsroot_v_separator").remove();
            area.style('left', '0px');
            d3.select("#"+this.gui_div+"_drawing").style('left','0px'); // reset size
            main.select(".jsroot_h_separator").style('left','0px');
            d3.select("#"+this.gui_div+"_status").style('left','0px'); // reset left
            this.checkResize();
         }

         this.browser_kind = kind;
         this.browser_visible = true;

         main.select(".jsroot_browser_resize").style("display", (kind === "float") ? null : "none");
         main.select(".jsroot_browser_title").style("cursor", (kind === "float") ? "move" : null);

         if (kind === "float") {
            area.style('bottom', '40px').classed('jsroot_float_browser', true);
           let drag_move = d3.drag().on("start", () => {
              let sl = area.style('left'), st = area.style('top');
              this._float_left = parseInt(sl.substr(0,sl.length-2));
              this._float_top = parseInt(st.substr(0,st.length-2));
              this._max_left = main.node().clientWidth - area.node().offsetWidth - 1;
              this._max_top = main.node().clientHeight - area.node().offsetHeight - 1;

           }).filter(evnt => {
               return main.select(".jsroot_browser_title").node() === evnt.target;
           }).on("drag", evnt => {
              this._float_left += evnt.dx;
              this._float_top += evnt.dy;

              area.style('left', Math.min(Math.max(0, this._float_left), this._max_left) + "px")
                  .style('top', Math.min(Math.max(0, this._float_top), this._max_top) + "px");

              this.setButtonsPosition();
           });

           let drag_resize = d3.drag().on("start", () => {
              let sw = area.style('width');
              this._float_width = parseInt(sw.substr(0,sw.length-2));
              this._float_height = area.node().clientHeight;
              this._max_width = main.node().clientWidth - area.node().offsetLeft - 1;
              this._max_height = main.node().clientHeight - area.node().offsetTop - 1;

           }).on("drag", evnt => {
              this._float_width += evnt.dx;
              this._float_height += evnt.dy;

              area.style('width', Math.min(Math.max(100, this._float_width), this._max_width) + "px")
                  .style('height', Math.min(Math.max(100, this._float_height), this._max_height) + "px");

              this.setButtonsPosition();
           });

           main.call(drag_move);
           main.select(".jsroot_browser_resize").call(drag_resize);

           this.adjustBrowserSize();

        } else {

           area.style('left', 0).style('top', 0).style('bottom', 0).style('height', null);

           let vsepar =
              main.append('div')
                  .classed("jsroot_separator", true).classed('jsroot_v_separator', true)
                  .style('position', 'absolute').style('top',0).style('bottom',0);

           let drag_move = d3.drag().on("start", () => {
               this._vsepar_move = this._vsepar_position;
               vsepar.style('background-color', 'grey');
           }).on("drag", evnt => {
               this._vsepar_move += evnt.dx;
               this.setButtonsPosition();
               this.adjustSeparators(Math.round(this._vsepar_move), null);
           }).on("end", () => {
               delete this._vsepar_move;
               vsepar.style('background-color', null);
               this.checkResize();
           });

           vsepar.call(drag_move);

           // need to get touches events handling in drag
           if (JSROOT.browser.touches && !main.on("touchmove"))
              main.on("touchmove", function() { });

           this.adjustSeparators(250, null, true, true);
        }

         this.setButtonsPosition();

         return Promise.resolve(this);
      }
   } // class BrowserLayout

   // ==============================================================================


   /** @summary central function for expand of all online items
     * @private */
   function onlineHierarchy(node, obj) {
      if (obj && node && ('_childs' in obj)) {

         for (let n = 0; n < obj._childs.length; ++n)
            if (obj._childs[n]._more || obj._childs[n]._childs)
               obj._childs[n]._expand = onlineHierarchy;

         node._childs = obj._childs;
         obj._childs = null;
         return true;
      }

      return false;
   }

   // ==============================================================

   /** @summary Current hierarchy painter
     * @desc Instance of {@link JSROOT.HierarchyPainter} object
     * @private */
   JSROOT.hpainter = null;

   /**
     * @summary Painter of hierarchical structures
     *
     * @memberof JSROOT
     * @example
     * // create hierarchy painter in "myTreeDiv"
     * let h = new JSROOT.HierarchyPainter("example", "myTreeDiv");
     * // configure 'simple' layout in "myMainDiv"
     * // one also can specify "grid2x2" or "flex" or "tabs"
     * h.setDisplay("simple", "myMainDiv");
     * // open file and display element
     * h.openRootFile("https://root.cern/js/files/hsimple.root").then(() => h.display("hpxpy;1","colz")); */

   class HierarchyPainter extends JSROOT.BasePainter {

      /** @summary Create painter
        * @param {string} name - symbolic name
        * @param {string} frameid - element id where hierarchy is drawn
        * @param {string} [backgr] - background color */
      constructor(name, frameid, backgr) {
         super(frameid);
         this.name = name;
         this.h = null; // hierarchy
         this.with_icons = true;
         this.background = backgr;
         this.files_monitoring = !frameid; // by default files monitored when nobrowser option specified
         this.nobrowser = (frameid === null);

         // remember only very first instance
         if (!JSROOT.hpainter)
            JSROOT.hpainter = this;
      }

      /** @summary Cleanup hierarchy painter
        * @desc clear drawing and browser */
      cleanup() {
         this.clearHierarchy(true);

         super.cleanup();

         if (JSROOT.hpainter === this)
            JSROOT.hpainter = null;
      }

      /** @summary Create file hierarchy
        * @private */
      fileHierarchy(file) {
         let painter = this;

         let folder = {
            _name : file.fFileName,
            _title : (file.fTitle ? (file.fTitle + ", path ") : "")  + file.fFullURL,
            _kind : "ROOT.TFile",
            _file : file,
            _fullurl : file.fFullURL,
            _localfile : file.fLocalFile,
            _had_direct_read : false,
            // this is central get method, item or itemname can be used, returns promise
            _get : function(item, itemname) {

               let fff = this; // file item

               if (item && item._readobj)
                  return Promise.resolve(item._readobj);

               if (item) itemname = painter.itemFullName(item, fff);

               function ReadFileObject(file) {
                  if (!fff._file) fff._file = file;

                  if (!file) return Promise.resolve(null);

                  return file.readObject(itemname).then(obj => {

                     // if object was read even when item did not exist try to reconstruct new hierarchy
                     if (!item && obj) {
                        // first try to found last read directory
                        let d = painter.findItem({name:itemname, top:fff, last_exists:true, check_keys:true });
                        if ((d!=null) && ('last' in d) && (d.last!=fff)) {
                           // reconstruct only subdir hierarchy
                           let dir = file.getDir(painter.itemFullName(d.last, fff));
                           if (dir) {
                              d.last._name = d.last._keyname;
                              let dirname = painter.itemFullName(d.last, fff);
                              keysHierarchy(d.last, dir.fKeys, file, dirname + "/");
                           }
                        } else {
                           // reconstruct full file hierarchy
                           keysHierarchy(fff, file.fKeys, file, "");
                        }
                        item = painter.findItem({name:itemname, top: fff});
                     }

                     if (item) {
                        item._readobj = obj;
                        // remove cycle number for objects supporting expand
                        if ('_expand' in item) item._name = item._keyname;
                     }

                     return obj;
                  });
               }

               if (fff._file) return ReadFileObject(fff._file);
               if (fff._localfile) return JSROOT.openFile(fff._localfile).then(f => ReadFileObject(f));
               if (fff._fullurl) return JSROOT.openFile(fff._fullurl).then(f => ReadFileObject(f));
               return Promise.resolve(null);
            }
         };

         keysHierarchy(folder, file.fKeys, file, "");

         return folder;
      }

      /** @summary Iterate over all items in hierarchy
        * @param {function} func - function called for every item
        * @param {object} [top] - top item to start from
        * @private */
      forEachItem(func, top) {
         function each_item(item, prnt) {
            if (!item) return;
            if (prnt) item._parent = prnt;
            func(item);
            if ('_childs' in item)
               for (let n = 0; n < item._childs.length; ++n)
                  each_item(item._childs[n], item);
         }

         if (typeof func == 'function')
            each_item(top || this.h);
      }

      /** @summary Search item in the hierarchy
        * @param {object|string} arg - item name or object with arguments
        * @param {string} arg.name -  item to search
        * @param {boolean} [arg.force] - specified elements will be created when not exists
        * @param {boolean} [arg.last_exists] -  when specified last parent element will be returned
        * @param {boolean} [arg.check_keys] - check TFile keys with cycle suffix
        * @param {boolean} [arg.allow_index] - let use sub-item indexes instead of name
        * @param {object} [arg.top] - element to start search from
        * @private */
      findItem(arg) {

         function find_in_hierarchy(top, fullname) {

            if (!fullname || (fullname.length == 0) || !top) return top;

            let pos = fullname.length;

            if (!top._parent && (top._kind !== 'TopFolder') && (fullname.indexOf(top._name)===0)) {
               // it is allowed to provide item name, which includes top-parent like file.root/folder/item
               // but one could skip top-item name, if there are no other items
               if (fullname === top._name) return top;

               let len = top._name.length;
               if (fullname[len] == "/") {
                  fullname = fullname.substr(len+1);
                  pos = fullname.length;
               }
            }

            function process_child(child, ignore_prnt) {
               // set parent pointer when searching child
               if (!ignore_prnt) child._parent = top;

               if ((pos >= fullname.length-1) || (pos < 0)) return child;

               return find_in_hierarchy(child, fullname.substr(pos + 1));
            }

            while (pos > 0) {
               // we try to find element with slashes inside - start from full name
               let localname = (pos >= fullname.length) ? fullname : fullname.substr(0, pos);

               if (top._childs) {
                  // first try to find direct matched item
                  for (let i = 0; i < top._childs.length; ++i)
                     if (top._childs[i]._name == localname)
                        return process_child(top._childs[i]);

                  // if first child online, check its elements
                  if ((top._kind === 'TopFolder') && (top._childs[0]._online!==undefined))
                     for (let i = 0; i < top._childs[0]._childs.length; ++i)
                        if (top._childs[0]._childs[i]._name == localname)
                           return process_child(top._childs[0]._childs[i], true);

                  // if allowed, try to found item with key
                  if (arg.check_keys) {
                     let newest = null;
                     for (let i = 0; i < top._childs.length; ++i) {
                       if (top._childs[i]._keyname === localname) {
                          if (!newest || (newest._cycle < top._childs[i]._cycle)) newest = top._childs[i];
                       }
                     }
                     if (newest) return process_child(newest);
                  }

                  let allow_index = arg.allow_index;
                  if ((localname[0] === '[') && (localname[localname.length-1] === ']') &&
                       /^\d+$/.test(localname.substr(1,localname.length-2))) {
                     allow_index = true;
                     localname = localname.substr(1,localname.length-2);
                  }

                  // when search for the elements it could be allowed to check index
                  if (allow_index && /^\d+$/.test(localname)) {
                     let indx = parseInt(localname);
                     if (Number.isInteger(indx) && (indx >= 0) && (indx < top._childs.length))
                        return process_child(top._childs[indx]);
                  }
               }

               pos = fullname.lastIndexOf("/", pos - 1);
            }

            if (arg.force) {
                // if did not found element with given name we just generate it
                if (top._childs === undefined) top._childs = [];
                pos = fullname.indexOf("/");
                let child = { _name: ((pos < 0) ? fullname : fullname.substr(0, pos)) };
                top._childs.push(child);
                return process_child(child);
            }

            return (arg.last_exists && top) ? { last: top, rest: fullname } : null;
         }

         let top = this.h, itemname = "";

         if (arg === null) return null; else
         if (typeof arg == 'string') { itemname = arg; arg = {}; } else
         if (typeof arg == 'object') { itemname = arg.name; if ('top' in arg) top = arg.top; } else
            return null;

         if (itemname === "__top_folder__") return top;

         if ((typeof itemname == 'string') && (itemname.indexOf("img:")==0)) return null;

         return find_in_hierarchy(top, itemname);
      }

      /** @summary Produce full string name for item
        * @param {Object} node - item element
        * @param {Object} [uptoparent] - up to which parent to continue
        * @param {boolean} [compact] - if specified, top parent is not included
        * @returns {string} produced name
        * @private */
      itemFullName(node, uptoparent, compact) {

         if (node && node._kind ==='TopFolder') return "__top_folder__";

         let res = "";

         while (node) {
            // online items never includes top-level folder
            if ((node._online!==undefined) && !uptoparent) return res;

            if ((node === uptoparent) || (node._kind==='TopFolder')) break;
            if (compact && !node._parent) break; // in compact form top-parent is not included
            if (res.length > 0) res = "/" + res;
            res = node._name + res;
            node = node._parent;
         }

         return res;
      }

       /** @summary Executes item marked as 'Command'
         * @desc If command requires additional arguments, they could be specified as extra arguments arg1, arg2, ...
         * @param {String} itemname - name of command item
         * @param {Object} [elem] - HTML element for command execution
         * @param [arg1] - first optional argument
         * @param [arg2] - second optional argument and so on
         * @returns {Promise} with command result */
      executeCommand(itemname, elem) {

         let hitem = this.findItem(itemname),
             url = this.getOnlineItemUrl(hitem) + "/cmd.json",
             d3node = d3.select(elem),
             cmdargs = [];

         if ('_numargs' in hitem)
            for (let n = 0; n < hitem._numargs; ++n)
               cmdargs.push((n+2 < arguments.length) ? arguments[n+2] : "");

         let promise = (cmdargs.length == 0) || !elem ? Promise.resolve(cmdargs) :
                        jsrp.createMenu().then(menu => menu.showCommandArgsDialog(hitem._name, cmdargs));

         return promise.then(args => {
            if (args === null) return false;

            let urlargs = "";
            for (let k = 0; k < args.length; ++k)
               urlargs += `${k > 0 ?  "&" : "?"}arg${k+1}=${args[k]}`;

           if (!d3node.empty()) {
               d3node.style('background','yellow');
               if (hitem && hitem._title) d3node.attr('title', "Executing " + hitem._title);
            }

            return JSROOT.httpRequest(url + urlargs, 'text').then(res => {
               if (!d3node.empty()) {
                  let col = ((res != null) && (res != 'false')) ? 'green' : 'red';
                  if (hitem && hitem._title) d3node.attr('title', hitem._title + " lastres=" + res);
                  d3node.style('background', col);
                  setTimeout(() => d3node.style('background', ''), 2000);
                  if ((col == 'green') && ('_hreload' in hitem))
                     this.reload();
                  if ((col == 'green') && ('_update_item' in hitem))
                     this.updateItems(hitem._update_item.split(";"));
               }
               return res;
            });
         });
      }

      /** @summary Get object item with specified name
        * @desc depending from provided option, same item can generate different object types
        * @param {Object} arg - item name or config object
        * @param {string} arg.name - item name
        * @param {Object} arg.item - or item itself
        * @param {string} options - supposed draw options
        * @returns {Promise} with object like { item, obj, itemname }
        * @private */
      getObject(arg, options) {

         let itemname, item, result = { item: null, obj: null };

         if (arg===null) return Promise.resolve(result);

         if (typeof arg === 'string') {
            itemname = arg;
         } else if (typeof arg === 'object') {
            if ((arg._parent!==undefined) && (arg._name!==undefined) && (arg._kind!==undefined)) item = arg; else
            if (arg.name!==undefined) itemname = arg.name; else
            if (arg.arg!==undefined) itemname = arg.arg; else
            if (arg.item!==undefined) item = arg.item;
         }

         if ((typeof itemname == 'string') && (itemname.indexOf("img:")==0)) {
            // artificial class, can be created by users
            result.obj = {_typename: "TJSImage", fName: itemname.substr(4)};
            return Promise.resolve(result);
         }

         if (item) itemname = this.itemFullName(item);
              else item = this.findItem( { name: itemname, allow_index: true, check_keys: true } );

         // if item not found, try to find nearest parent which could allow us to get inside

         let d = item ? null : this.findItem({ name: itemname, last_exists: true, check_keys: true, allow_index: true });

         // if item not found, try to expand hierarchy central function
         // implements not process get in central method of hierarchy item (if exists)
         // if last_parent found, try to expand it
         if ((d !== null) && ('last' in d) && (d.last !== null)) {
            let parentname = this.itemFullName(d.last);

            // this is indication that expand does not give us better path to searched item
            if ((typeof arg == 'object') && ('rest' in arg))
               if ((arg.rest == d.rest) || (arg.rest.length <= d.rest.length))
                  return Promise.resolve(result);

            return this.expandItem(parentname, undefined, options != "hierarchy_expand_verbose").then(res => {
               if (!res) return result;
               let newparentname = this.itemFullName(d.last);
               if (newparentname.length > 0) newparentname += "/";
               return this.getObject( { name: newparentname + d.rest, rest: d.rest }, options);
            });
         }

         result.item = item;

         if ((item !== null) && (typeof item._obj == 'object')) {
            result.obj = item._obj;
            return Promise.resolve(result);
         }

         // normally search _get method in the parent items
         let curr = item;
         while (curr) {
            if (typeof curr._get == 'function')
               return curr._get(item, null, options).then(obj => { result.obj = obj; return result; });
            curr = ('_parent' in curr) ? curr._parent : null;
         }

         return Promise.resolve(result);
      }

         /** @summary returns true if item is last in parent childs list
        * @private */
      isLastSibling(hitem) {
         if (!hitem || !hitem._parent || !hitem._parent._childs) return false;
         let chlds = hitem._parent._childs, indx = chlds.indexOf(hitem);
         if (indx < 0) return false;
         while (++indx < chlds.length)
            if (!('_hidden' in chlds[indx])) return false;
         return true;
      }

      /** @summary Create item html code
        * @private */
      addItemHtml(hitem, d3prnt, arg) {
         if (!hitem || ('_hidden' in hitem)) return true;

         let isroot = (hitem === this.h),
             has_childs = ('_childs' in hitem),
             handle = jsrp.getDrawHandle(hitem._kind),
             img1 = "", img2 = "", can_click = false, break_list = false,
             d3cont, itemname = this.itemFullName(hitem);

         if (handle !== null) {
            if ('icon' in handle) img1 = handle.icon;
            if ('icon2' in handle) img2 = handle.icon2;
            if ((img1.length==0) && (typeof handle.icon_get == 'function'))
               img1 = handle.icon_get(hitem, this);
            if (('func' in handle) || ('execute' in handle) || ('aslink' in handle) ||
                (('expand' in handle) && (hitem._more !== false))) can_click = true;
         }

         if ('_icon' in hitem) img1 = hitem._icon;
         if ('_icon2' in hitem) img2 = hitem._icon2;
         if ((img1.length == 0) && ('_online' in hitem))
            hitem._icon = img1 = "img_globe";
         if ((img1.length == 0) && isroot)
            hitem._icon = img1 = "img_base";

         if (hitem._more || hitem._expand || hitem._player || hitem._can_draw)
            can_click = true;

         let can_menu = can_click;
         if (!can_menu && (typeof hitem._kind == 'string') && (hitem._kind.indexOf("ROOT.")==0))
            can_menu = can_click = true;

         if (img2.length == 0) img2 = img1;
         if (img1.length == 0) img1 = (has_childs || hitem._more) ? "img_folder" : "img_page";
         if (img2.length == 0) img2 = (has_childs || hitem._more) ? "img_folderopen" : "img_page";

         if (arg === "update") {
            d3prnt.selectAll("*").remove();
            d3cont = d3prnt;
         } else {
            d3cont = d3prnt.append("div");
            if (arg && (arg >= (hitem._parent._show_limit || JSROOT.settings.HierarchyLimit))) break_list = true;
         }

         hitem._d3cont = d3cont.node(); // set for direct referencing
         d3cont.attr("item", itemname);

         // line with all html elements for this item (excluding childs)
         let d3line = d3cont.append("div").attr('class','h_line');

         // build indent
         let prnt = isroot ? null : hitem._parent;
         while (prnt && (prnt !== this.h)) {
            d3line.insert("div",":first-child")
                  .attr("class", this.isLastSibling(prnt) ? "img_empty" : "img_line");
            prnt = prnt._parent;
         }

         let icon_class = "", plusminus = false;

         if (isroot) {
            // for root node no extra code
         } else if (has_childs && !break_list) {
            icon_class = hitem._isopen ? "img_minus" : "img_plus";
            plusminus = true;
         } else /*if (hitem._more) {
            icon_class = "img_plus"; // should be special plus ???
            plusminus = true;
         } else */ {
            icon_class = "img_join";
         }

         let h = this;

         if (icon_class.length > 0) {
            if (break_list || this.isLastSibling(hitem)) icon_class += "bottom";
            let d3icon = d3line.append("div").attr('class', icon_class);
            if (plusminus) d3icon.style('cursor','pointer')
                                 .on("click", function(evnt) { h.tree_click(evnt, this, "plusminus"); });
         }

         // make node icons

         if (this.with_icons && !break_list) {
            let icon_name = hitem._isopen ? img2 : img1, d3img;

            if (icon_name.indexOf("img_")==0)
               d3img = d3line.append("div")
                             .attr("class", icon_name)
                             .attr("title", hitem._kind);
            else
               d3img = d3line.append("img")
                             .attr("src", icon_name)
                             .attr("alt","")
                             .attr("title", hitem._kind)
                             .style('vertical-align','top').style('width','18px').style('height','18px');

            if (('_icon_click' in hitem) || (handle && ('icon_click' in handle)))
               d3img.on("click", function(evnt) { h.tree_click(evnt, this, "icon"); });
         }

         let d3a = d3line.append("a");
         if (can_click || has_childs || break_list)
            d3a.attr("class","h_item")
               .on("click", function(evnt) { h.tree_click(evnt, this); });

         if (break_list) {
            hitem._break_point = true; // indicate that list was broken here
            d3a.attr('title', 'there are ' + (hitem._parent._childs.length-arg) + ' more items')
               .text("...more...");
            return false;
         }

         if ('disp_kind' in h) {
            if (JSROOT.settings.DragAndDrop && can_click)
              this.enableDrag(d3a, itemname);

            if (JSROOT.settings.ContextMenu && can_menu)
               d3a.on('contextmenu', function(evnt) { h.tree_contextmenu(evnt, this); });

            d3a.on("mouseover", function() { h.tree_mouseover(true, this); })
               .on("mouseleave", function() { h.tree_mouseover(false, this); });
         } else if (hitem._direct_context && JSROOT.settings.ContextMenu)
            d3a.on('contextmenu', function(evnt) { h.direct_contextmenu(evnt, this); });

         let element_name = hitem._name, element_title = "";

         if ('_realname' in hitem)
            element_name = hitem._realname;

         if ('_title' in hitem)
            element_title = hitem._title;

         if ('_fullname' in hitem)
            element_title += "  fullname: " + hitem._fullname;

         if (!element_title)
            element_title = element_name;

         d3a.attr('title', element_title)
            .text(element_name + ('_value' in hitem ? ":" : ""))
            .style('background', hitem._background ? hitem._background : null);

         if ('_value' in hitem) {
            let d3p = d3line.append("p");
            if ('_vclass' in hitem) d3p.attr('class', hitem._vclass);
            if (!hitem._isopen) d3p.html(hitem._value);
         }

         if (has_childs && (isroot || hitem._isopen)) {
            let d3chlds = d3cont.append("div").attr("class", "h_childs");
            if (this.show_overflow) d3chlds.style("overflow", "initial");
            for (let i = 0; i < hitem._childs.length; ++i) {
               let chld = hitem._childs[i];
               chld._parent = hitem;
               if (!this.addItemHtml(chld, d3chlds, i)) break; // if too many items, skip rest
            }
         }

         return true;
      }

      /** @summary Toggle open state of the item
        * @desc Used with "open all" / "close all" buttons in normal GUI
        * @param {boolean} isopen - if items should be expand or closed
        * @returns {boolean} true when any item was changed */
      toggleOpenState(isopen, h) {
         let hitem = h || this.h;

         if (hitem._childs === undefined) {
            if (!isopen) return false;

            if (this.with_icons) {
               // in normal hierarchy check precisely if item can be expand
               if (!hitem._more && !hitem._expand && !this.canExpandItem(hitem)) return false;
            }

            this.expandItem(this.itemFullName(hitem));
            if (hitem._childs !== undefined) hitem._isopen = true;
            return hitem._isopen;
         }

         if ((hitem !== this.h) && isopen && !hitem._isopen) {
            // when there are childs and they are not see, simply show them
            hitem._isopen = true;
            return true;
         }

         let change_child = false;
         for (let i = 0; i < hitem._childs.length; ++i)
            if (this.toggleOpenState(isopen, hitem._childs[i]))
               change_child = true;

         if ((hitem !== this.h) && !isopen && hitem._isopen && !change_child) {
            // if none of the childs can be closed, than just close that item
            delete hitem._isopen;
            return true;
          }

         if (!h) this.refreshHtml();
         return false;
      }

      /** @summary Refresh HTML code of hierarchy painter
        * @returns {Promise} when done */
      refreshHtml() {

         let d3elem = this.selectDom();
         if (d3elem.empty())
            return Promise.resolve(this);

         d3elem.html("")   // clear html - most simple way
               .style('overflow',this.show_overflow ? 'auto' : 'hidden')
               .style('display','flex')
               .style('flex-direction','column');

         let h = this, factcmds = [], status_item = null;
         this.forEachItem(item => {
            delete item._d3cont; // remove html container
            if (('_fastcmd' in item) && (item._kind == 'Command')) factcmds.push(item);
            if (('_status' in item) && !status_item) status_item = item;
         });

         if (!this.h || d3elem.empty())
            return Promise.resolve(this);

         if (factcmds.length) {
            let fastbtns = d3elem.append("div").attr("style", "display: inline; vertical-align: middle; white-space: nowrap;");
            for (let n = 0; n < factcmds.length; ++n) {
               let btn = fastbtns.append("button")
                          .text("")
                          .attr("class",'jsroot_fastcmd_btn')
                          .attr("item", this.itemFullName(factcmds[n]))
                          .attr("title", factcmds[n]._title)
                          .on("click", function() { h.executeCommand(d3.select(this).attr("item"), this); } );

               if (factcmds[n]._icon)
                  btn.style("background-image", `url("${factcmds[n]._icon}")`);
            }
         }

         let d3btns = d3elem.append("p").attr("class", "jsroot").style("margin-bottom","3px").style("margin-top",0);
         d3btns.append("a").attr("class", "h_button").text("open all")
               .attr("title","open all items in the browser").on("click", () => this.toggleOpenState(true));
         d3btns.append("text").text(" | ");
         d3btns.append("a").attr("class", "h_button").text("close all")
               .attr("title","close all items in the browser").on("click", () => this.toggleOpenState(false));

         if (typeof this.removeInspector == 'function') {
            d3btns.append("text").text(" | ");
            d3btns.append("a").attr("class", "h_button").text("remove")
                  .attr("title","remove inspector").on("click", () => this.removeInspector());
         }

         if ('_online' in this.h) {
            d3btns.append("text").text(" | ");
            d3btns.append("a").attr("class", "h_button").text("reload")
                  .attr("title","reload object list from the server").on("click", () => this.reload());
         }

         if ('disp_kind' in this) {
            d3btns.append("text").text(" | ");
            d3btns.append("a").attr("class", "h_button").text("clear")
                  .attr("title","clear all drawn objects").on("click", () => this.clearHierarchy(false));
         }

         let maindiv =
            d3elem.append("div")
                  .attr("class", "jsroot")
                  .style('font-size', this.with_icons ? "12px" : "15px")
                  .style("flex","1");

         if (!this.show_overflow)
            maindiv.style("overflow","auto");

         if (this.background) // case of object inspector and streamer infos display
            maindiv.style("background-color", this.background)
                   .style('margin', '2px').style('padding', '2px');

         this.addItemHtml(this.h, maindiv.append("div").attr("class","h_tree"));

         this.setTopPainter(); //assign hpainter as top painter

         if (status_item && !this.status_disabled && !JSROOT.decodeUrl().has('nostatus')) {
            let func = JSROOT.findFunction(status_item._status);
            if (typeof func == 'function')
               return this.createStatusLine().then(sdiv => {
                  if (sdiv) func(sdiv, this.itemFullName(status_item));
               });
         }

         return Promise.resolve(this);
      }

      /** @summary Update item node
        * @private */
      updateTreeNode(hitem, d3cont) {
         if ((d3cont === undefined) || d3cont.empty())  {
            d3cont = d3.select(hitem._d3cont ? hitem._d3cont : null);
            let name = this.itemFullName(hitem);
            if (d3cont.empty())
               d3cont = this.selectDom().select("[item='" + name + "']");
            if (d3cont.empty() && ('_cycle' in hitem))
               d3cont = this.selectDom().select("[item='" + name + ";" + hitem._cycle + "']");
            if (d3cont.empty()) return;
         }

         this.addItemHtml(hitem, d3cont, "update");

         if (this.brlayout) this.brlayout.adjustBrowserSize(true);
      }

      /** @summary Update item background
        * @private */
      updateBackground(hitem, scroll_into_view) {
         if (!hitem || !hitem._d3cont) return;

         let d3cont = d3.select(hitem._d3cont);

         if (d3cont.empty()) return;

         let d3a = d3cont.select(".h_item");

         d3a.style('background', hitem._background ? hitem._background : null);

         if (scroll_into_view && hitem._background)
            d3a.node().scrollIntoView(false);
      }

      /** @summary Focus on hierarchy item
        * @param {Object|string} hitem - item to open or its name
        * @desc all parents to the otem will be opened first
        * @returns {Promise} when done
        * @private */
      focusOnItem(hitem) {
         if (typeof hitem == "string")
            hitem = this.findItem(hitem);

         let name = hitem ? this.itemFullName(hitem) : "";
         if (!name) return Promise.resolve(false)

         let itm = hitem, need_refresh = false;

         while (itm) {
            if ((itm._childs !== undefined) && !itm._isopen) {
               itm._isopen = true;
               need_refresh = true;
            }
            itm = itm._parent;
         }

         let promise = need_refresh ? this.refreshHtml() : Promise.resolve(true);

         return promise.then(() => {
            let d3cont = this.selectDom().select("[item='" + name + "']");
            if (d3cont.empty()) return false;
            d3cont.node().scrollIntoView();
            return true;
         });
      }

      /** @summary Handler for click event of item in the hierarchy
        * @private */
      tree_click(evnt, node, place) {
         if (!node) return;

         let d3cont = d3.select(node.parentNode.parentNode),
             itemname = d3cont.attr('item'),
             hitem = itemname ? this.findItem(itemname) : null;
         if (!hitem) return;

         if (hitem._break_point) {
            // special case of more item

            delete hitem._break_point;

            // update item itself
            this.addItemHtml(hitem, d3cont, "update");

            let prnt = hitem._parent, indx = prnt._childs.indexOf(hitem),
                d3chlds = d3.select(d3cont.node().parentNode);

            if (indx < 0) return console.error('internal error');

            prnt._show_limit = (prnt._show_limit || JSROOT.settings.HierarchyLimit) * 2;

            for (let n = indx+1; n < prnt._childs.length; ++n) {
               let chld = prnt._childs[n];
               chld._parent = prnt;
               if (!this.addItemHtml(chld, d3chlds, n)) break; // if too many items, skip rest
            }

            return;
         }

         let prnt = hitem, dflt = undefined;
         while (prnt) {
            if ((dflt = prnt._click_action) !== undefined) break;
            prnt = prnt._parent;
         }

         if (!place || (place=="")) place = "item";
         let selector = (hitem._kind == "ROOT.TKey" && hitem._more) ? "noinspect" : "",
             sett = jsrp.getDrawSettings(hitem._kind, selector), handle = sett.handle;

         if (place == "icon") {
            let func = null;
            if (typeof hitem._icon_click == 'function')
               func = hitem._icon_click;
            else if (handle && typeof handle.icon_click == 'function')
               func = handle.icon_click;
            if (func && func(hitem,this))
               this.updateTreeNode(hitem, d3cont);
            return;
         }

         // special feature - all items with '_expand' function are not drawn by click
         if ((place=="item") && ('_expand' in hitem) && !evnt.ctrlKey && !evnt.shiftKey) place = "plusminus";

         // special case - one should expand item
         if (((place == "plusminus") && !('_childs' in hitem) && hitem._more) ||
             ((place == "item") && (dflt === "expand"))) {
            return this.expandItem(itemname, d3cont);
         }

         if (place == "item") {

            if ('_player' in hitem)
               return this.player(itemname);

            if (handle && handle.aslink)
               return window.open(itemname + "/");

            if (handle && handle.execute)
               return this.executeCommand(itemname, node.parentNode);

            if (handle && handle.ignore_online && this.isOnlineItem(hitem)) return;

            let can_draw = hitem._can_draw,
                can_expand = hitem._more,
                dflt_expand = (this.default_by_click === "expand"),
                drawopt = "";

            if (evnt.shiftKey) {
               drawopt = (handle && handle.shift) ? handle.shift : "inspect";
               if ((drawopt==="inspect") && handle && handle.noinspect) drawopt = "";
            }
            if (handle && handle.ctrl && evnt.ctrlKey)
               drawopt = handle.ctrl;

            if (!drawopt) {
               for (let pitem = hitem._parent; !!pitem; pitem = pitem._parent) {
                  if (pitem._painter) { can_draw = false; if (can_expand===undefined) can_expand = false; break; }
               }
            }

            if (hitem._childs) can_expand = false;

            if (can_draw === undefined) can_draw = sett.draw;
            if (can_expand === undefined) can_expand = sett.expand;

            if (can_draw && can_expand && !drawopt) {
               // if default action specified as expand, disable drawing
               // if already displayed, try to expand
               if (dflt_expand || (handle && (handle.dflt === 'expand')) || this.isItemDisplayed(itemname)) can_draw = false;
            }

            if (can_draw && !drawopt && handle && handle.dflt && (handle.dflt !== 'expand'))
               drawopt = handle.dflt;

            if (can_draw)
               return this.display(itemname, drawopt);

            if (can_expand || dflt_expand)
               return this.expandItem(itemname, d3cont);

            // cannot draw, but can inspect ROOT objects
            if ((typeof hitem._kind === "string") && (hitem._kind.indexOf("ROOT.")===0) && sett.inspect && (can_draw !== false))
               return this.display(itemname, "inspect");

            if (!hitem._childs || (hitem === this.h)) return;
         }

         if (hitem._isopen)
            delete hitem._isopen;
         else
            hitem._isopen = true;

         this.updateTreeNode(hitem, d3cont);
      }

      /** @summary Handler for mouse-over event
        * @private */
      tree_mouseover(on, elem) {
         let itemname = d3.select(elem.parentNode.parentNode).attr('item'),
              hitem = this.findItem(itemname);

         if (!hitem) return;

         let painter, prnt = hitem;
         while (prnt && !painter) {
            painter = prnt._painter;
            prnt = prnt._parent;
         }

         if (painter && typeof painter.mouseOverHierarchy === 'function')
            painter.mouseOverHierarchy(on, itemname, hitem);
      }

      /** @summary alternative context menu, used in the object inspector
        * @private */
      direct_contextmenu(evnt, elem) {
         evnt.preventDefault();
         let itemname = d3.select(elem.parentNode.parentNode).attr('item'),
              hitem = this.findItem(itemname);
         if (!hitem) return;

         if (typeof this.fill_context == 'function')
            jsrp.createMenu(evnt, this).then(menu => {
               this.fill_context(menu, hitem);
               if (menu.size() > 0) {
                  menu.tree_node = elem.parentNode;
                  menu.show();
               }
            });
      }

      /** @summary Handle context menu in the hieararchy
        * @private */
      tree_contextmenu(evnt, elem) {
         evnt.preventDefault();
         let itemname = d3.select(elem.parentNode.parentNode).attr('item'),
              hitem = this.findItem(itemname);
         if (!hitem) return;

         let onlineprop = this.getOnlineProp(itemname),
             fileprop = this.getFileProp(itemname);

         function qualifyURL(url) {
            const escapeHTML = s => s.split('&').join('&amp;').split('<').join('&lt;').split('"').join('&quot;'),
                  el = document.createElement('div');
            el.innerHTML = '<a href="' + escapeHTML(url) + '">x</a>';
            return el.firstChild.href;
         }

         jsrp.createMenu(evnt, this).then(menu => {

            if ((!itemname || !hitem._parent) && !('_jsonfile' in hitem)) {
               let files = [], addr = "", cnt = 0,
                   separ = () => (cnt++ > 0) ? "&" : "?";

               this.forEachRootFile(item => files.push(item._file.fFullURL));

               if (!this.getTopOnlineItem())
                  addr = JSROOT.source_dir + "index.htm";

               if (this.isMonitoring())
                  addr += separ() + "monitoring=" + this.getMonitoringInterval();

               if (files.length == 1)
                  addr += separ() + "file=" + files[0];
               else if (files.length > 1)
                  addr += separ() + "files=" + JSON.stringify(files);

               if (this.disp_kind)
                  addr += separ() + "layout=" + this.disp_kind.replace(/ /g, "");

               let items = [], opts = [];

               if (this.disp)
                  this.disp.forEachFrame(frame => {
                     let dummy = new JSROOT.ObjectPainter(frame),
                         top = dummy.getTopPainter(),
                         item = top ? top.getItemName() : null, opt;

                     if (item) {
                        opt  = top.getDrawOpt() || top.getItemDrawOpt();
                     } else {
                        top = null;
                        dummy.forEachPainter(p => {
                           let _item = p.getItemName();
                           if (!_item) return;
                           let _opt = p.getDrawOpt() || p.getItemDrawOpt() || "";
                           if (!top) {
                              top = p;
                              item = _item;
                              opt = _opt;
                           } else if (top.getPadPainter() === p.getPadPainter()) {
                              if (_opt.indexOf("same ")==0) _opt = _opt.substr(5);
                              item += "+" + _item;
                              opt += "+" + _opt;
                           }
                        });
                     }

                     if (item) {
                        items.push(item);
                        opts.push(opt || "");
                     }
                  });

               if (items.length == 1) {
                  addr += separ() + "item=" + items[0] + separ() + "opt=" + opts[0];
               } else if (items.length > 1) {
                  addr += separ() + "items=" + JSON.stringify(items) + separ() + "opts=" + JSON.stringify(opts);
               }

               menu.add("Direct link", () => window.open(addr));
               menu.add("Only items", () => window.open(addr + "&nobrowser"));
            } else if (onlineprop) {
               this.fillOnlineMenu(menu, onlineprop, itemname);
            } else {
               let sett = jsrp.getDrawSettings(hitem._kind, 'nosame');

               // allow to draw item even if draw function is not defined
               if (hitem._can_draw) {
                  if (!sett.opts) sett.opts = [""];
                  if (sett.opts.indexOf("") < 0) sett.opts.unshift("");
               }

               if (sett.opts)
                  menu.addDrawMenu("Draw", sett.opts, arg => this.display(itemname, arg));

               if (fileprop && sett.opts && !fileprop.localfile) {
                  let filepath = qualifyURL(fileprop.fileurl);
                  if (filepath.indexOf(JSROOT.source_dir) == 0)
                     filepath = filepath.slice(JSROOT.source_dir.length);
                  filepath = fileprop.kind + "=" + filepath;
                  if (fileprop.itemname.length > 0) {
                     let name = fileprop.itemname;
                     if (name.search(/\+| |\,/)>=0) name = "\'" + name + "\'";
                     filepath += "&item=" + name;
                  }

                  menu.addDrawMenu("Draw in new tab", sett.opts,
                                   arg => window.open(JSROOT.source_dir + "index.htm?nobrowser&"+filepath +"&opt="+arg));
               }

               if (sett.expand && !('_childs' in hitem) && (hitem._more || !('_more' in hitem)))
                  menu.add("Expand", () => this.expandItem(itemname));

               if (hitem._kind === "ROOT.TStyle")
                  menu.add("Apply", () => this.applyStyle(itemname));
            }

            if (typeof hitem._menu == 'function')
               hitem._menu(menu, hitem, this);

            if (menu.size() > 0) {
               menu.tree_node = elem.parentNode;
               if (menu.separ) menu.add("separator"); // add separator at the end
               menu.add("Close");
               menu.show();
            }

         }); // end menu creation

         return false;
      }

      /** @summary Starts player for specified item
        * @desc Same as "Player" context menu
        * @param {string} itemname - item name for which player should be started
        * @param {string} [option] - extra options for the player
        * @returns {Promise} when ready*/
      player(itemname, option) {
         let item = this.findItem(itemname);

         if (!item || !item._player) return Promise.resolve(null);

         return JSROOT.require(item._prereq || '').then(() => {

            let player_func = JSROOT.findFunction(item._player);
            if (!player_func) return null;

            return this.createDisplay().then(mdi => {
               let res = mdi ? player_func(this, itemname, option) : null;
               return res;
            });
         });
      }

      /** @summary Checks if item can be displayed with given draw option
        * @private */
      canDisplay(item, drawopt) {
         if (!item) return false;
         if (item._player) return true;
         if (item._can_draw !== undefined) return item._can_draw;
         if (drawopt == 'inspect') return true;
         const handle = jsrp.getDrawHandle(item._kind, drawopt);
         return handle && (handle.func || handle.class || handle.draw_field);
      }

      /** @summary Returns true if given item displayed
        * @param {string} itemname - item name */
      isItemDisplayed(itemname) {
         let mdi = this.getDisplay();
         return mdi ? mdi.findFrame(itemname) !== null : false;
      }

      /** @summary Display specified item
        * @param {string} itemname - item name
        * @param {string} [drawopt] - draw option for the item
        * @returns {Promise} with created painter object */
      display(itemname, drawopt) {
         let h = this,
             painter = null,
             updating = false,
             item = null,
             display_itemname = itemname,
             frame_name = itemname,
             marker = "::_display_on_frame_::",
             p = drawopt ? drawopt.indexOf(marker) : -1;

         if (p >= 0) {
            frame_name = drawopt.substr(p + marker.length);
            drawopt = drawopt.substr(0, p);
         }

         function complete(respainter, err) {
            if (err) console.log('When display ', itemname, err);

            if (updating && item) delete item._doing_update;
            if (!updating) jsrp.showProgress();
            if (respainter && (typeof respainter === 'object') && (typeof respainter.setItemName === 'function')) {
               respainter.setItemName(display_itemname, updating ? null : drawopt, h); // mark painter as created from hierarchy
               if (item && !item._painter) item._painter = respainter;
            }

            return respainter || painter;
         }

         return h.createDisplay().then(mdi => {

            if (!mdi) return complete();

            item = h.findItem(display_itemname);

            if (item && ('_player' in item))
               return h.player(display_itemname, drawopt).then(res => complete(res));

            updating = (typeof drawopt == 'string') && (drawopt.indexOf("update:")==0);

            if (updating) {
               drawopt = drawopt.substr(7);
               if (!item || item._doing_update) return complete();
               item._doing_update = true;
            }

            if (item && !h.canDisplay(item, drawopt)) return complete();

            let divid = "", use_dflt_opt = false;
            if ((typeof drawopt == 'string') && (drawopt.indexOf("divid:") >= 0)) {
               let pos = drawopt.indexOf("divid:");
               divid = drawopt.slice(pos+6);
               drawopt = drawopt.slice(0, pos);
            }

            if (drawopt == "__default_draw_option__") {
               use_dflt_opt = true;
               drawopt = "";
            }

            if (!updating) jsrp.showProgress("Loading " + display_itemname);

            return h.getObject(display_itemname, drawopt).then(result => {
               if (!updating) jsrp.showProgress();

               if (!item) item = result.item;
               let obj = result.obj;

               if (!obj) return complete();

               if (!updating) jsrp.showProgress("Drawing " + display_itemname);

               let handle = obj._typename ? jsrp.getDrawHandle("ROOT." + obj._typename) : null;

               if (handle && handle.draw_field && obj[handle.draw_field]) {
                  obj = obj[handle.draw_field];
                  if (!drawopt) drawopt = handle.draw_field_opt || "";
                  handle = obj._typename ? jsrp.getDrawHandle("ROOT." + obj._typename) : null;
               }

               if (use_dflt_opt && handle && handle.dflt && !drawopt && (handle.dflt != 'expand'))
                  drawopt = handle.dflt;

               if (divid.length > 0) {
                  let func = updating ? JSROOT.redraw : JSROOT.draw;
                  return func(divid, obj, drawopt).then(p => complete(p)).catch(err => complete(null, err));
               }

               mdi.forEachPainter((p, frame) => {
                  if (p.getItemName() != display_itemname) return;
                  // verify that object was drawn with same option as specified now (if any)
                  if (!updating && drawopt && (p.getItemDrawOpt() != drawopt)) return;

                  // do not actiavte frame when doing update
                  // mdi.activateFrame(frame);

                  if ((typeof p.redrawObject == 'function') && p.redrawObject(obj, drawopt)) painter = p;
               });

               if (painter) return complete();

               if (updating) {
                  console.warn(`something went wrong - did not found painter when doing update of ${display_itemname}`);
                  return complete();
               }

               let frame = mdi.findFrame(frame_name, true);
               d3.select(frame).html("");
               mdi.activateFrame(frame);

               return JSROOT.draw(frame, obj, drawopt)
                            .then(p => complete(p))
                            .catch(err => complete(null, err));

            });
         });
      }

      /** @summary Enable drag of the element
        * @private  */
      enableDrag(d3elem /*, itemname*/) {
         d3elem.attr("draggable", "true").on("dragstart", function(ev) {
            let itemname = this.parentNode.parentNode.getAttribute('item');
            ev.dataTransfer.setData("item", itemname);
         });
      }

      /** @summary Enable drop on the frame
        * @private  */
      enableDrop(frame) {
         let h = this;
         d3.select(frame).on("dragover", function(ev) {
            let itemname = ev.dataTransfer.getData("item"),
                 ditem = h.findItem(itemname);
            if (ditem && (typeof ditem._kind == 'string') && (ditem._kind.indexOf("ROOT.")==0))
               ev.preventDefault(); // let accept drop, otherwise it will be refuced
         }).on("dragenter", function() {
            d3.select(this).classed('jsroot_drag_area', true);
         }).on("dragleave", function() {
            d3.select(this).classed('jsroot_drag_area', false);
         }).on("drop", function(ev) {
            d3.select(this).classed('jsroot_drag_area', false);
            let itemname = ev.dataTransfer.getData("item");
            if (itemname) h.dropItem(itemname, this.getAttribute("id"));
         });
      }

      /** @summary Remove all drop handlers on the frame
        * @private  */
      clearDrop(frame) {
         d3.select(frame).on("dragover", null).on("dragenter", null).on("dragleave", null).on("drop", null);
      }

     /** @summary Drop item on specified element for drawing
       * @returns {Promise} when completed
       * @private */
      dropItem(itemname, divid, opt) {

         if (opt && typeof opt === 'function') { call_back = opt; opt = ""; }
         if (opt===undefined) opt = "";

         let drop_complete = (drop_painter, is_main_painter) => {
            if (drop_painter && !is_main_painter && (typeof drop_painter === 'object') && (typeof drop_painter.setItemName == 'function'))
               drop_painter.setItemName(itemname, null, this);
            return drop_painter;
         }

         if (itemname == "$legend")
            return JSROOT.require("hist")
                         .then(() => jsrp.produceLegend(divid, opt))
                         .then(legend_painter => drop_complete(legend_painter));

         return this.getObject(itemname).then(res => {

            if (!res.obj) return null;

            let main_painter = jsrp.getElementMainPainter(divid);

            if (main_painter && (typeof main_painter.performDrop === 'function'))
               return main_painter.performDrop(res.obj, itemname, res.item, opt).then(p => drop_complete(p, main_painter === p));

            if (main_painter && main_painter.accept_drops)
               return JSROOT.draw(divid, res.obj, "same " + opt).then(p => drop_complete(p, main_painter === p));

            this.cleanupFrame(divid);
            return JSROOT.draw(divid, res.obj, opt).then(p => drop_complete(p));
         });
      }

      /** @summary Update specified items
        * @desc Method can be used to fetch new objects and update all existing drawings
        * @param {string|array|boolean} arg - either item name or array of items names to update or true if only automatic items will be updated
        * @returns {Promise} when ready */
      updateItems(arg) {

         if (!this.disp)
            return Promise.resolve(false);

         let allitems = [], options = [], only_auto_items = false, want_update_all = false;

         if (typeof arg == "string")
            arg = [ arg ];
         else if (typeof arg != 'object') {
            if (arg === undefined) arg = !this.isMonitoring();
            want_update_all = true;
            only_auto_items = !!arg;
         }

         // first collect items
         this.disp.forEachPainter(p => {
            let itemname = p.getItemName();

            if ((typeof itemname != 'string') || (allitems.indexOf(itemname) >= 0)) return;

            if (want_update_all) {
               let item = this.findItem(itemname);
               if (!item || ('_not_monitor' in item) || ('_player' in item)) return;
               if (!('_always_monitor' in item)) {
                  let forced = false, handle = jsrp.getDrawHandle(item._kind);
                  if (handle && ('monitor' in handle)) {
                     if ((handle.monitor === false) || (handle.monitor == 'never')) return;
                     if (handle.monitor == 'always') forced = true;
                  }
                  if (!forced && only_auto_items) return;
               }
            } else {
               if (arg.indexOf(itemname) < 0) return;
            }

            allitems.push(itemname);
            options.push("update:" + p.getItemDrawOpt());
         }, true); // only visible panels are considered

         // force all files to read again (normally in non-browser mode)
         if (this.files_monitoring && !only_auto_items && want_update_all)
            this.forEachRootFile(item => {
               this.forEachItem(fitem => { delete fitem._readobj; }, item);
               delete item._file;
            });

         return this.displayItems(allitems, options);
      }

      /** @summary Display all provided elements
        * @returns {Promise} when drawing finished
        * @private */
      displayItems(items, options) {

         if (!items || (items.length == 0))
            return Promise.resolve(true);

         let h = this;

         if (!options) options = [];
         while (options.length < items.length)
            options.push("__default_draw_option__");

         if ((options.length == 1) && (options[0] == "iotest")) {
            this.clearHierarchy();
            d3.select("#" + this.disp_frameid).html("<h2>Start I/O test</h2>");

            let tm0 = new Date();
            return this.getObject(items[0]).then(() => {
               let tm1 = new Date();
               d3.select("#" + this.disp_frameid).append("h2").html("Item " + items[0] + " reading time = " + (tm1.getTime() - tm0.getTime()) + "ms");
               return Promise.resolve(true);
            });
         }

         let dropitems = new Array(items.length),
             dropopts = new Array(items.length),
             images = new Array(items.length);

         // First of all check that items are exists, look for cycle extension and plus sign
         for (let i = 0; i < items.length; ++i) {
            dropitems[i] = dropopts[i] = null;

            let item = items[i], can_split = true;

            if (item && item.indexOf("img:")==0) { images[i] = true; continue; }

            if (item && (item.length>1) && (item[0]=='\'') && (item[item.length-1]=='\'')) {
               items[i] = item.substr(1, item.length-2);
               can_split = false;
            }

            let elem = h.findItem({ name: items[i], check_keys: true });
            if (elem) { items[i] = h.itemFullName(elem); continue; }

            if (can_split && (items[i][0]=='[') && (items[i][items[i].length-1]==']')) {
               dropitems[i] = parseAsArray(items[i]);
               items[i] = dropitems[i].shift();
            } else if (can_split && (items[i].indexOf("+") > 0)) {
               dropitems[i] = items[i].split("+");
               items[i] = dropitems[i].shift();
            }

            if (dropitems[i] && dropitems[i].length > 0) {
               // allow to specify _same_ item in different file
               for (let j = 0; j < dropitems[i].length; ++j) {
                  let pos = dropitems[i][j].indexOf("_same_");
                  if ((pos > 0) && (h.findItem(dropitems[i][j]) === null))
                     dropitems[i][j] = dropitems[i][j].substr(0,pos) + items[i].substr(pos);

                  elem = h.findItem({ name: dropitems[i][j], check_keys: true });
                  if (elem) dropitems[i][j] = h.itemFullName(elem);
               }

               if ((options[i][0] == "[") && (options[i][options[i].length-1] == "]")) {
                  dropopts[i] = parseAsArray(options[i]);
                  options[i] = dropopts[i].shift();
               } else if (options[i].indexOf("+") > 0) {
                  dropopts[i] = options[i].split("+");
                  options[i] = dropopts[i].shift();
               } else {
                  dropopts[i] = [];
               }

               while (dropopts[i].length < dropitems[i].length) dropopts[i].push("");
            }

            // also check if subsequent items has _same_, than use name from first item
            let pos = items[i].indexOf("_same_");
            if ((pos > 0) && !h.findItem(items[i]) && (i > 0))
               items[i] = items[i].substr(0,pos) + items[0].substr(pos);

            elem = h.findItem({ name: items[i], check_keys: true });
            if (elem) items[i] = h.itemFullName(elem);
         }

         // now check that items can be displayed
         for (let n = items.length - 1; n >= 0; --n) {
            if (images[n]) continue;
            let hitem = h.findItem(items[n]);
            if (!hitem || h.canDisplay(hitem, options[n])) continue;
            // try to expand specified item
            h.expandItem(items[n], null, true);
            items.splice(n, 1);
            options.splice(n, 1);
            dropitems.splice(n, 1);
         }

         if (items.length == 0)
            return Promise.resolve(true);

         let frame_names = new Array(items.length), items_wait = new Array(items.length);
         for (let n = 0; n < items.length; ++n) {
            items_wait[n] = 0;
            let fname = items[n], k = 0;
            if (items.indexOf(fname) < n) items_wait[n] = true; // if same item specified, one should wait first drawing before start next
            let p = options[n].indexOf("frameid:");
            if (p >= 0) {
               fname = options[n].substr(p+8);
               options[n] = options[n].substr(0,p);
            } else {
               while (frame_names.indexOf(fname)>=0)
                  fname = items[n] + "_" + k++;
            }
            frame_names[n] = fname;
         }

         // now check if several same items present - select only one for the drawing
         // if draw option includes 'main', such item will be drawn first
         for (let n = 0; n < items.length; ++n) {
            if (items_wait[n] !== 0) continue;
            let found_main = n;
            for (let k = 0; k < items.length; ++k)
               if ((items[n]===items[k]) && (options[k].indexOf('main')>=0)) found_main = k;
            for (let k = 0; k < items.length; ++k)
               if (items[n]===items[k]) items_wait[k] = (found_main != k);
         }

         return this.createDisplay().then(mdi => {
            if (!mdi) return false;

            // Than create empty frames for each item
            for (let i = 0; i < items.length; ++i)
               if (options[i].indexOf('update:')!==0) {
                  mdi.createFrame(frame_names[i]);
                  options[i] += "::_display_on_frame_::"+frame_names[i];
               }

            function DropNextItem(indx, painter) {
               if (painter && dropitems[indx] && (dropitems[indx].length > 0))
                  return h.dropItem(dropitems[indx].shift(), painter.getDom(), dropopts[indx].shift()).then(() => DropNextItem(indx, painter));

               dropitems[indx] = null; // mark that all drop items are processed
               items[indx] = null; // mark item as ready

               for (let cnt = 0; cnt < items.length; ++cnt) {
                  if (items[cnt]===null) continue; // ignore completed item
                  if (items_wait[cnt] && items.indexOf(items[cnt])===cnt) {
                     items_wait[cnt] = false;
                     return h.display(items[cnt], options[cnt]).then(painter => DropNextItem(cnt, painter));
                  }
               }
            }

            let promises = [];

            // We start display of all items parallel, but only if they are not the same
            for (let i = 0; i < items.length; ++i)
               if (!items_wait[i])
                  promises.push(h.display(items[i], options[i]).then(painter => DropNextItem(i, painter)));

            return Promise.all(promises);
         });
      }

      /** @summary Reload hierarchy and refresh html code
        * @returns {Promise} when completed */
      reload() {
         if ('_online' in this.h)
            return this.openOnline(this.h._online).then(() => this.refreshHtml());
         return Promise.resolve(false);
      }

      /** @summary activate (select) specified item
        * @param {Array} items - array of items names
        * @param {boolean} [force] - if specified, all required sub-levels will be opened
        * @private */
      activateItems(items, force) {

         if (typeof items == 'string') items = [ items ];

         let active = [], // array of elements to activate
             update = []; // array of elements to update
         this.forEachItem(item => { if (item._background) { active.push(item); delete item._background; } });

         let mark_active = () => {

            for (let n = update.length-1; n >= 0; --n)
               this.updateTreeNode(update[n]);

            for (let n = 0; n < active.length; ++n)
               this.updateBackground(active[n], force);
         }

         let find_next = (itemname, prev_found) => {
            if (itemname === undefined) {
               // extract next element
               if (items.length == 0) return mark_active();
               itemname = items.shift();
            }

            let hitem = this.findItem(itemname);

            if (!hitem) {
               let d = this.findItem({ name: itemname, last_exists: true, check_keys: true, allow_index: true });
               if (!d || !d.last) return find_next();
               d.now_found = this.itemFullName(d.last);

               if (force) {

                  // if after last expand no better solution found - skip it
                  if ((prev_found!==undefined) && (d.now_found === prev_found)) return find_next();

                  return this.expandItem(d.now_found).then(res => {
                     if (!res) return find_next();
                     let newname = this.itemFullName(d.last);
                     if (newname.length > 0) newname+="/";
                     find_next(newname + d.rest, d.now_found);
                  });
               }
               hitem = d.last;
            }

            if (hitem) {
               // check that item is visible (opened), otherwise should enable parent

               let prnt = hitem._parent;
               while (prnt) {
                  if (!prnt._isopen) {
                     if (force) {
                        prnt._isopen = true;
                        if (update.indexOf(prnt)<0) update.push(prnt);
                     } else {
                        hitem = prnt; break;
                     }
                  }
                  prnt = prnt._parent;
               }

               hitem._background = 'LightSteelBlue';
               if (active.indexOf(hitem)<0) active.push(hitem);
            }

            find_next();
         }

         if (force && this.brlayout) {
            if (!this.brlayout.browser_kind)
              return this.createBrowser('float', true).then(() => find_next());
            if (!this.brlayout.browser_visible)
               this.brlayout.toggleBrowserVisisbility();
         }

         // use recursion
         find_next();
      }

      /** @summary Check if item can be (potentially) expand
        * @private */
      canExpandItem(item) {
         if (!item) return false;
         if (item._expand) return true;
         let handle = jsrp.getDrawHandle(item._kind, "::expand");
         return handle && (handle.expand_item || handle.expand);
      }

      /** @summary expand specified item
        * @param {String} itemname - item name
        * @returns {Promise} when ready */
      expandItem(itemname, d3cont, silent) {
         let hitem = this.findItem(itemname);

         if (!hitem && d3cont)
            return Promise.resolve();

         let DoExpandItem = (_item, _obj) => {

            if (typeof _item._expand == 'string')
               _item._expand = JSROOT.findFunction(item._expand);

            if (typeof _item._expand !== 'function') {
               let handle = jsrp.getDrawHandle(_item._kind, "::expand");

               if (handle && handle.expand_item) {
                  _obj = _obj[handle.expand_item];
                 if (_obj && _obj._typename)
                    handle = jsrp.getDrawHandle("ROOT."+_obj._typename, "::expand");
               }

               if (handle && handle.expand) {
                  if (typeof handle.expand == 'string')
                     return JSROOT.require(handle.prereq).then(() => {
                        _item._expand = handle.expand = JSROOT.findFunction(handle.expand);
                        return _item._expand ? DoExpandItem(_item, _obj) : true;
                     });
                  _item._expand = handle.expand;
               }
            }

            // try to use expand function
            if (_obj && _item && (typeof _item._expand === 'function')) {
               if (_item._expand(_item, _obj)) {
                  _item._isopen = true;
                  if (_item._parent && !_item._parent._isopen) {
                     _item._parent._isopen = true; // also show parent
                     if (!silent) this.updateTreeNode(_item._parent);
                  } else {
                     if (!silent) this.updateTreeNode(_item, d3cont);
                  }
                  return Promise.resolve(_item);
               }
            }

            if (_obj && objectHierarchy(_item, _obj)) {
               _item._isopen = true;
               if (_item._parent && !_item._parent._isopen) {
                  _item._parent._isopen = true; // also show parent
                  if (!silent) this.updateTreeNode(_item._parent);
               } else {
                  if (!silent) this.updateTreeNode(_item, d3cont);
               }
               return Promise.resolve(_item);
            }

            return Promise.resolve(-1);
         };

         let promise = Promise.resolve(-1);

         if (hitem) {
            // item marked as it cannot be expanded, also top item cannot be changed
            if ((hitem._more === false) || (!hitem._parent && hitem._childs))
               return Promise.resolve();

            if (hitem._childs && hitem._isopen) {
               hitem._isopen = false;
               if (!silent) this.updateTreeNode(hitem, d3cont);
               return Promise.resolve();
            }

            if (hitem._obj) promise = DoExpandItem(hitem, hitem._obj);
         }

         return promise.then(res => {
            if (res !== -1) return res; // done

            jsrp.showProgress("Loading " + itemname);

            return this.getObject(itemname, silent ? "hierarchy_expand" : "hierarchy_expand_verbose").then(res => {

               jsrp.showProgress();

               if (res.obj) return DoExpandItem(res.item, res.obj).then(res => { return (res !== -1) ? res : undefined; });
            });
         });

      }

      /** @summary Return main online item
        * @private */
      getTopOnlineItem(item) {
         if (item) {
            while (item && (!('_online' in item))) item = item._parent;
            return item;
         }

         if (!this.h) return null;
         if ('_online' in this.h) return this.h;
         if (this.h._childs && ('_online' in this.h._childs[0])) return this.h._childs[0];
         return null;
      }

      /** @summary Call function for each item which corresponds to JSON file
        * @private */
      forEachJsonFile(func) {
         if (!this.h) return;
         if ('_jsonfile' in this.h)
            return func(this.h);

         if (this.h._childs)
            for (let n = 0; n < this.h._childs.length; ++n) {
               let item = this.h._childs[n];
               if ('_jsonfile' in item) func(item);
            }
      }

      /** @summary Open JSON file
        * @param {string} filepath - URL to JSON file
        * @returns {Promise} when object ready */
      openJsonFile(filepath) {
         let isfileopened = false;
         this.forEachJsonFile(item => { if (item._jsonfile==filepath) isfileopened = true; });
         if (isfileopened) return Promise.resolve();

         return JSROOT.httpRequest(filepath, 'object').then(res => {
            let h1 = { _jsonfile: filepath, _kind: "ROOT." + res._typename, _jsontmp: res, _name: filepath.split("/").pop() };
            if (res.fTitle) h1._title = res.fTitle;
            h1._get = function(item /* ,itemname */) {
               if (item._jsontmp)
                  return Promise.resolve(item._jsontmp);
               return JSROOT.httpRequest(item._jsonfile, 'object')
                            .then(res => {
                                item._jsontmp = res;
                                return res;
                             });
            }
            if (!this.h) this.h = h1; else
            if (this.h._kind == 'TopFolder') this.h._childs.push(h1); else {
               let h0 = this.h, topname = ('_jsonfile' in h0) ? "Files" : "Items";
               this.h = { _name: topname, _kind: 'TopFolder', _childs : [h0, h1] };
            }

            return this.refreshHtml();
         });
      }

      /** @summary Call function for each item which corresponds to ROOT file
        * @private */
      forEachRootFile(func) {
         if (!this.h) return;
         if ((this.h._kind == "ROOT.TFile") && this.h._file)
            return func(this.h);

         if (this.h._childs)
            for (let n = 0; n < this.h._childs.length; ++n) {
               let item = this.h._childs[n];
               if ((item._kind == 'ROOT.TFile') && ('_fullurl' in item))
                  func(item);
            }
      }

      /** @summary Open ROOT file
        * @param {string} filepath - URL to ROOT file
        * @returns {Promise} when file is opened */
      openRootFile(filepath) {

         let isfileopened = false;
         this.forEachRootFile(item => { if (item._fullurl===filepath) isfileopened = true; });
         if (isfileopened) return Promise.resolve();

         jsrp.showProgress("Opening " + filepath + " ...");

         return JSROOT.openFile(filepath).then(file => {

            let h1 = this.fileHierarchy(file);
            h1._isopen = true;
            if (!this.h) {
               this.h = h1;
               if (this._topname) h1._name = this._topname;
            } else if (this.h._kind == 'TopFolder') {
               this.h._childs.push(h1);
            }  else {
               let h0 = this.h, topname = (h0._kind == "ROOT.TFile") ? "Files" : "Items";
               this.h = { _name: topname, _kind: 'TopFolder', _childs : [h0, h1], _isopen: true };
            }

            return this.refreshHtml();
         }).catch(() => {
            // make CORS warning
            if (JSROOT.batch_mode)
               console.error(`Fail to open ${filepath} - check CORS headers`);
            else if (!d3.select("#gui_fileCORS").style("background","red").empty())
               setTimeout(() => d3.select("#gui_fileCORS").style("background",''), 5000);
            return false;
         }).finally(() => jsrp.showProgress());
      }

      /** @summary Apply loaded TStyle object
        * @desc One also can specify item name of JSON file name where style is loaded
        * @param {object|string} style - either TStyle object of item name where object can be load */
      applyStyle(style) {
         if (!style)
            return Promise.resolve();

         if (typeof style === 'object') {
            if (style._typename === "TStyle")
               JSROOT.extend(JSROOT.gStyle, style);
            return Promise.resolve();
         }

         if (typeof style === 'string') {

            let item = this.findItem({ name: style, allow_index: true, check_keys: true });

            if (item!==null)
               return this.getObject(item).then(res => this.applyStyle(res.obj));

            if (style.indexOf('.json') > 0)
               return JSROOT.httpRequest(style, 'object')
                            .then(res => this.applyStyle(res));
         }

         return Promise.resolve();
      }

      /** @summary Provides information abouf file item
        * @private */
      getFileProp(itemname) {
         let item = this.findItem(itemname);
         if (!item) return null;

         let subname = item._name;
         while (item._parent) {
            item = item._parent;
            if ('_file' in item)
               return { kind: "file", fileurl: item._file.fURL, itemname: subname, localfile: !!item._file.fLocalFile };

            if ('_jsonfile' in item)
               return { kind: "json", fileurl: item._jsonfile, itemname: subname };

            subname = item._name + "/" + subname;
         }

         return null;
      }

      /** @summary Provides URL for online item
        * @desc Such URL can be used  to request data from the server
        * @returns string or null if item is not online
        * @private */
      getOnlineItemUrl(item) {
         if (typeof item == "string") item = this.findItem(item);
         let prnt = item;
         while (prnt && (prnt._online===undefined)) prnt = prnt._parent;
         return prnt ? (prnt._online + this.itemFullName(item, prnt)) : null;
      }

      /** @summary Returns true if item is online
        * @private */
      isOnlineItem(item) {
         return this.getOnlineItemUrl(item) !== null;
      }

      /** @summary method used to request object from the http server
        * @returns {Promise} with requested object
        * @private */
      getOnlineItem(item, itemname, option) {

         let url = itemname, h_get = false, req = "", req_kind = "object", draw_handle = null;

         if ((typeof option == "string") && (option.indexOf('hierarchy_expand')==0)) {
            h_get = true;
            option = undefined;
         }

         if (item) {
            url = this.getOnlineItemUrl(item);
            let func = null;
            if ('_kind' in item) draw_handle = jsrp.getDrawHandle(item._kind);

            if (h_get) {
               req = 'h.json?compact=3';
               item._expand = onlineHierarchy; // use proper expand function
            } else if (item._make_request) {
               func = JSROOT.findFunction(item._make_request);
            } else if (draw_handle && draw_handle.make_request) {
               func = draw_handle.make_request;
            }

            if (typeof func == 'function') {
               // ask to make request
               let dreq = func(this, item, url, option);
               // result can be simple string or object with req and kind fields
               if (dreq!=null)
                  if (typeof dreq == 'string') req = dreq; else {
                     if ('req' in dreq) req = dreq.req;
                     if ('kind' in dreq) req_kind = dreq.kind;
                  }
            }

            if ((req.length == 0) && (item._kind.indexOf("ROOT.") != 0))
              req = 'item.json.gz?compact=3';
         }

         if (!itemname && item && ('_cached_draw_object' in this) && (req.length == 0)) {
            // special handling for online draw when cashed
            let obj = this._cached_draw_object;
            delete this._cached_draw_object;
            return Promise.resolve(obj);
         }

         if (req.length == 0) req = 'root.json.gz?compact=23';

         if (url.length > 0) url += "/";
         url += req;

         return new Promise(resolveFunc => {

            let itemreq = JSROOT.NewHttpRequest(url, req_kind, obj => {

               let func = null;

               if (!h_get && item && ('_after_request' in item)) {
                  func = JSROOT.findFunction(item._after_request);
               } else if (draw_handle && ('after_request' in draw_handle))
                  func = draw_handle.after_request;

               if (typeof func == 'function') {
                  let res = func(this, item, obj, option, itemreq);
                  if (res && (typeof res == "object")) obj = res;
               }

               resolveFunc(obj);
            });

            itemreq.send(null);
         });
      }

      /** @summary Access THttpServer with provided address
        * @param {string} server_address - URL to server like "http://localhost:8090/"
        * @returns {Promise} when ready */
      openOnline(server_address) {
         let AdoptHierarchy = result => {
            this.h = result;
            if (!result) return Promise.resolve(null);

            if (('_title' in this.h) && (this.h._title!='')) document.title = this.h._title;

            result._isopen = true;

            // mark top hierarchy as online data and
            this.h._online = server_address;

            this.h._get = (item, itemname, option) => {
               return this.getOnlineItem(item, itemname, option);
            };

            this.h._expand = onlineHierarchy;

            let styles = [], scripts = [], modules = [];
            this.forEachItem(item => {
               if ('_childs' in item) item._expand = onlineHierarchy;

               if ('_autoload' in item) {
                  let arr = item._autoload.split(";");
                  arr.forEach(name => {
                     if ((name.length > 3) && (name.lastIndexOf(".js") == name.length-3)) {
                        if (!scripts.find(elem => elem == name)) scripts.push(name);
                     } else if ((name.length > 4) && (name.lastIndexOf(".css") == name.length-4)) {
                        if (!styles.find(elem => elem == name)) styles.push(name);
                     } else if (name && !modules.find(elem => elem == name)) {
                        modules.push(name);
                     }
                  });
               }
            });

            return JSROOT.require(modules)
                  .then(() => JSROOT.require(scripts))
                  .then(() => JSROOT.loadScript(styles))
                  .then(() => {
                     this.forEachItem(item => {
                        if (!('_drawfunc' in item) || !('_kind' in item)) return;
                        let typename = "kind:" + item._kind;
                        if (item._kind.indexOf('ROOT.')==0) typename = item._kind.slice(5);
                        let drawopt = item._drawopt;
                        if (!jsrp.canDraw(typename) || drawopt)
                           jsrp.addDrawFunc({ name: typename, func: item._drawfunc, script: item._drawscript, opt: drawopt });
                     });

                     return this;
                  });
         }

         if (!server_address) server_address = "";

         if (typeof server_address == 'object') {
            let h = server_address;
            server_address = "";
            return AdoptHierarchy(h);
         }

         return JSROOT.httpRequest(server_address + "h.json?compact=3", 'object')
                      .then(hh => AdoptHierarchy(hh));
      }

      /** @summary Get properties for online item  - server name and relative name
        * @private */
      getOnlineProp(itemname) {
         let item = this.findItem(itemname);
         if (!item) return null;

         let subname = item._name;
         while (item._parent) {
            item = item._parent;

            if ('_online' in item) {
               return {
                  server: item._online,
                  itemname: subname
               };
            }
            subname = item._name + "/" + subname;
         }

         return null;
      }

      /** @summary Fill context menu for online item
        * @private */
      fillOnlineMenu(menu, onlineprop, itemname) {

         let node = this.findItem(itemname),
             sett = jsrp.getDrawSettings(node._kind, 'nosame;noinspect'),
             handle = jsrp.getDrawHandle(node._kind),
             root_type = (typeof node._kind == 'string') ? node._kind.indexOf("ROOT.") == 0 : false;

         if (sett.opts && (node._can_draw !== false)) {
            sett.opts.push('inspect');
            menu.addDrawMenu("Draw", sett.opts, arg => this.display(itemname, arg));
         }

         if (!node._childs && (node._more !== false) && (node._more || root_type || sett.expand))
            menu.add("Expand", () => this.expandItem(itemname));

         if (handle && ('execute' in handle))
            menu.add("Execute", () => this.executeCommand(itemname, menu.tree_node));

         if (sett.opts && (node._can_draw !== false))
            menu.addDrawMenu("Draw in new window", sett.opts,
                              arg => window.open(onlineprop.server + "?nobrowser&item=" + onlineprop.itemname +
                                                 (this.isMonitoring() ? "&monitoring=" + this.getMonitoringInterval() : "") +
                                                 (arg ? "&opt=" + arg : "")));

         if (sett.opts && (sett.opts.length > 0) && root_type && (node._can_draw !== false))
            menu.addDrawMenu("Draw as png", sett.opts,
                              arg => window.open(onlineprop.server + onlineprop.itemname + "/root.png?w=600&h=400" + (arg ? "&opt=" + arg : "")));

         if ('_player' in node)
            menu.add("Player", () => this.player(itemname));
      }

      /** @summary Assign existing hierarchy to the painter and refresh HTML code
        * @private */
      setHierarchy(h) {
         this.h = h;
         this.refreshHtml();
      }

      /** @summary Configures monitoring interval
        * @param {number} interval - repetition interval in ms
        * @param {boolean} flag - initial monitoring state */
      setMonitoring(interval, monitor_on) {

         this._runMonitoring("cleanup");

         if (interval) {
            interval = parseInt(interval);
            if (Number.isInteger(interval) && (interval > 0)) {
               this._monitoring_interval = Math.max(100,interval);
               monitor_on = true;
            } else {
               this._monitoring_interval = 3000;
            }
         }

         this._monitoring_on = monitor_on;

         if (this.isMonitoring())
            this._runMonitoring();
      }

      /** @summary Runs monitoring event loop
        * @private */
      _runMonitoring(arg) {
         if ((arg == "cleanup") || !this.isMonitoring()) {
            if (this._monitoring_handle) {
               clearTimeout(this._monitoring_handle);
               delete this._monitoring_handle;
            }

            if (this._monitoring_frame) {
               cancelAnimationFrame(this._monitoring_frame);
               delete this._monitoring_frame;
            }
            return;
         }

         if (arg == "frame") {
            // process of timeout, request animation frame
            delete this._monitoring_handle;
            this._monitoring_frame = requestAnimationFrame(this._runMonitoring.bind(this,"draw"));
            return;
         }

         if (arg == "draw") {
            delete this._monitoring_frame;
            this.updateItems();
         }

         this._monitoring_handle = setTimeout(this._runMonitoring.bind(this,"frame"), this.getMonitoringInterval());
      }

      /** @summary Returns configured monitoring interval in ms */
      getMonitoringInterval() {
         return this._monitoring_interval || 3000;
      }

      /** @summary Returns true when monitoring is enabled */
      isMonitoring() {
         return this._monitoring_on;
      }

      /** @summary Assign default layout and place where drawing will be performed
        * @param {string} layout - layout like "simple" or "grid2x2"
        * @param {string} frameid - DOM element id where object drawing will be performed */
      setDisplay(layout, frameid) {
         if (!frameid && (typeof layout == 'object')) {
            this.disp = layout;
            this.disp_kind = 'custom';
            this.disp_frameid = null;
         } else {
            this.disp_kind = layout;
            this.disp_frameid = frameid;
         }

         if (!this.register_resize) {
            this.register_resize = true;
            jsrp.registerForResize(this);
         }
      }

      /** @summary Returns configured layout */
      getLayout() {
         return this.disp_kind;
      }

      /** @summary Remove painter reference from hierarhcy
        * @private */
      removePainter(obj_painter) {
         this.forEachItem(item => {
            if (item._painter === obj_painter) {
               // delete painter reference
               delete item._painter;
               // also clear data which could be associated with item
               if (typeof item.clear == 'function') item.clear();
            }
         });
      }

      /** @summary Cleanup all items in hierarchy
        * @private */
      clearHierarchy(withbrowser) {
         if (this.disp) {
            this.disp.cleanup();
            delete this.disp;
         }

         let plainarr = [];

         this.forEachItem(item => {
            delete item._painter; // remove reference on the painter
            // when only display cleared, try to clear all browser items
            if (!withbrowser && (typeof item.clear == 'function')) item.clear();
            if (withbrowser) plainarr.push(item);
         });

         if (withbrowser) {
            // cleanup all monitoring loops
            this.enableMonitoring(false);
            // simplify work for javascript and delete all (ok, most of) cross-references
            this.selectDom().html("");
            plainarr.forEach(d => { delete d._parent; delete d._childs; delete d._obj; delete d._d3cont; });
            delete this.h;
         }
      }

      /** @summary Returns actual MDI display object
        * @desc It should an instance of {@link JSROOT.MDIDsiplay} class */
      getDisplay() {
         return this.disp;
      }

      /** @summary method called when MDI element is cleaned up
        * @desc hook to perform extra actions when frame is cleaned
        * @private */
      cleanupFrame(frame) {

         d3.select(frame).attr("frame_title", null);

         this.clearDrop(frame);

         let lst = JSROOT.cleanup(frame);

         // we remove all painters references from items
         if (lst && (lst.length > 0))
            this.forEachItem(item => {
               if (item._painter && lst.indexOf(item._painter) >= 0)
                  delete item._painter;
            });
      }

      /** @summary Creates configured JSROOT.MDIDisplay object
        * @returns {Promise} when ready
        * @private */
      createDisplay() {

         if ('disp' in this) {
            if ((this.disp.numDraw() > 0) || (this.disp_kind == "custom"))
               return Promise.resolve(this.disp);
            this.disp.cleanup();
            delete this.disp;
         }

         // check that we can found frame where drawing should be done
         if (!document.getElementById(this.disp_frameid))
            return Promise.resolve(null);

         if ((this.disp_kind.indexOf("flex") == 0) || (this.disp_kind == "tabs") || (this.disp_kind.indexOf("coll") == 0))
            this.disp = new JSROOT.FlexibleDisplay(this.disp_frameid);
         else
            this.disp = new JSROOT.GridDisplay(this.disp_frameid, this.disp_kind);

         this.disp.cleanupFrame = this.cleanupFrame.bind(this);
         if (JSROOT.settings.DragAndDrop)
             this.disp.setInitFrame(this.enableDrop.bind(this));

         return Promise.resolve(this.disp);
      }

      /** @summary If possible, creates custom JSROOT.MDIDisplay for given item
        * @param itemname - name of item, for which drawing is created
        * @param custom_kind - display kind
        * @returns {Promise} with mdi object created
        * @private */
      createCustomDisplay(itemname, custom_kind) {

         if (this.disp_kind != "simple")
            return this.createDisplay();

         this.disp_kind = custom_kind;

         // check if display can be erased
         if (this.disp) {
            let num = this.disp.numDraw();
            if ((num > 1) || ((num == 1) && !this.disp.findFrame(itemname)))
               return this.createDisplay();
            this.disp.cleanup();
            delete this.disp;
         }

         return this.createDisplay();
      }

      /** @summary function updates object drawings for other painters
        * @private */
      updateOnOtherFrames(painter, obj) {
         let mdi = this.disp, handle = null, isany = false;
         if (!mdi) return false;

         if (obj._typename) handle = jsrp.getDrawHandle("ROOT." + obj._typename);
         if (handle && handle.draw_field && obj[handle.draw_field])
            obj = obj[handle.draw_field];

         mdi.forEachPainter((p, frame) => {
            if ((p === painter) || (p.getItemName() != painter.getItemName())) return;

            // do not actiavte frame when doing update
            // mdi.activateFrame(frame);
            if ((typeof p.redrawObject == 'function') && p.redrawObject(obj)) isany = true;
         });
         return isany;
      }

      /** @summary Process resize event
        * @private */
      checkResize(size) {
         if (this.disp) this.disp.checkMDIResize(null, size);
      }

      /** @summary Start GUI
        * @returns {Promise} when ready
        * @private */
      startGUI(gui_div, url) {

         let d = JSROOT.decodeUrl(url);

         let GetOption = opt => {
            let res = d.get(opt, null);
            if ((res===null) && gui_div && !gui_div.empty() && gui_div.node().hasAttribute(opt))
               res = gui_div.attr(opt);
            return res;
         };

         let GetUrlOptionAsArray = opt => {

            let res = [];

            while (opt.length > 0) {
               let separ = opt.indexOf(";");
               let part = (separ>0) ? opt.substr(0, separ) : opt;

               if (separ > 0) opt = opt.substr(separ+1); else opt = "";

               let canarray = true;
               if (part[0]=='#') { part = part.substr(1); canarray = false; }

               let val = d.get(part,null);

               if (canarray)
                  res = res.concat(parseAsArray(val));
               else if (val!==null)
                  res.push(val);
            }
            return res;
         };

         let GetOptionAsArray = opt => {
            let res = GetUrlOptionAsArray(opt);
            if (res.length>0 || !gui_div || gui_div.empty()) return res;
            while (opt.length>0) {
               let separ = opt.indexOf(";");
               let part = separ>0 ? opt.substr(0, separ) : opt;
               if (separ>0) opt = opt.substr(separ+1); else opt = "";

               let canarray = true;
               if (part[0]=='#') { part = part.substr(1); canarray = false; }
               if (part==='files') continue; // special case for normal UI

               if (!gui_div.node().hasAttribute(part)) continue;

               let val = gui_div.attr(part);

               if (canarray) res = res.concat(parseAsArray(val));
               else if (val!==null) res.push(val);
            }
            return res;
         };

         let prereq = GetOption('prereq') || "",
             filesdir = d.get("path") || "", // path used in normal gui
             filesarr = GetOptionAsArray("#file;files"),
             localfile = GetOption("localfile"),
             jsonarr = GetOptionAsArray("#json;jsons"),
             expanditems = GetOptionAsArray("expand"),
             focusitem = GetOption("focus"),
             itemsarr = GetOptionAsArray("#item;items"),
             optionsarr = GetOptionAsArray("#opt;opts"),
             monitor = GetOption("monitoring"),
             layout = GetOption("layout"),
             style = GetOptionAsArray("#style"),
             statush = 0, status = GetOption("status"),
             browser_kind = GetOption("browser"),
             browser_configured = !!browser_kind,
             title = GetOption("title");

         if (monitor === null)
            monitor = 0;
         else if (monitor === "")
            monitor = 3000;
         else
            monitor = parseInt(monitor);

         if (GetOption("float")!==null) { browser_kind = 'float'; browser_configured = true; } else
         if (GetOption("fix")!==null) { browser_kind = 'fix'; browser_configured = true; }

         this.no_select = GetOption("noselect");

         if (GetOption('files_monitoring')!==null) this.files_monitoring = true;

         if (title) document.title = title;

         let load = GetOption("load");

         if (expanditems.length==0 && (GetOption("expand")==="")) expanditems.push("");

         if (filesdir) {
            for (let i=0;i<filesarr.length;++i) filesarr[i] = filesdir + filesarr[i];
            for (let i=0;i<jsonarr.length;++i) jsonarr[i] = filesdir + jsonarr[i];
         }

         if ((itemsarr.length == 0) && GetOption("item") === "") itemsarr.push("");

         if ((jsonarr.length == 1) && (itemsarr.length == 0) && (expanditems.length==0)) itemsarr.push("");

         if (!this.disp_kind) {
            if ((typeof layout == "string") && (layout.length > 0))
               this.disp_kind = layout;
            else
            switch (itemsarr.length) {
              case 0:
              case 1: this.disp_kind = 'simple'; break;
              case 2: this.disp_kind = 'vert2'; break;
              case 3: this.disp_kind = 'vert21'; break;
              case 4: this.disp_kind = 'vert22'; break;
              case 5: this.disp_kind = 'vert32'; break;
              case 6: this.disp_kind = 'vert222'; break;
              case 7: this.disp_kind = 'vert322'; break;
              case 8: this.disp_kind = 'vert332'; break;
              case 9: this.disp_kind = 'vert333'; break;
              default: this.disp_kind = 'flex';
            }
         }

         if (status==="no")
            status = null;
         else if (status==="off") {
            this.status_disabled = true;
            status = null;
         } else if (status==="on")
            status = true;
         else if (status!==null) {
            statush = parseInt(status);
            if (!Number.isInteger(statush) || (statush < 5)) statush = 0;
            status = true;
         }
         if (this.no_select === "") this.no_select = true;

         if (!browser_kind)
            browser_kind = "fix";
         else if (browser_kind === "no")
            browser_kind = "";
         else if (browser_kind==="off") {
            browser_kind = "";
            status = null;
            this.exclude_browser = true;
         }
         if (GetOption("nofloat")!==null) this.float_browser_disabled = true;

         if (this.start_without_browser) browser_kind = "";

         this._topname = GetOption("topname");

         let openAllFiles = () => {
            let promise;

            if (prereq) {
               promise = JSROOT.require(prereq); prereq = "";
            } else if (load) {
               promise = JSROOT.loadScript(load.split(";")); load = "";
            } else if (browser_kind) {
               promise = this.createBrowser(browser_kind); browser_kind = "";
            } else if (status !== null) {
               promise = this.createStatusLine(statush, status); status = null;
            } else if (jsonarr.length > 0)
               promise = this.openJsonFile(jsonarr.shift());
            else if (filesarr.length > 0)
               promise = this.openRootFile(filesarr.shift());
            else if ((localfile!==null) && (typeof this.selectLocalFile == 'function')) {
               localfile = null; promise = this.selectLocalFile();
            } else if (expanditems.length > 0)
               promise = this.expandItem(expanditems.shift());
            else if (style.length > 0)
               promise = this.applyStyle(style.shift());
            else
               return this.refreshHtml()
                      .then(() => this.displayItems(itemsarr, optionsarr))
                      .then(() => focusitem ? this.focusOnItem(focusitem) : this)
                      .then(() => {
                         this.setMonitoring(monitor);
                         return itemsarr ? this.refreshHtml() : this; // this is final return
                      });

            return promise.then(openAllFiles);
         };

         let h0 = null;
         if (this.is_online) {
            if (typeof GetCachedHierarchy == 'function')
               h0 = GetCachedHierarchy();
            if (typeof h0 !== 'object') h0 = "";

            if ((this.is_online == "draw") && !itemsarr.length)
               itemsarr.push("");
         }

         if (h0 !== null)
            return this.openOnline(h0).then(() => {
               // check if server enables monitoring
               if (!this.exclude_browser && !browser_configured && ('_browser' in this.h)) {
                  browser_kind = this.h._browser;
                  if (browser_kind==="no") browser_kind = ""; else
                  if (browser_kind==="off") { browser_kind = ""; status = null; this.exclude_browser = true; }
               }

               if (('_monitoring' in this.h) && !monitor)
                  monitor = this.h._monitoring;

               if (('_loadfile' in this.h) && (filesarr.length==0))
                  filesarr = parseAsArray(this.h._loadfile);

               if (('_drawitem' in this.h) && (itemsarr.length==0)) {
                  itemsarr = parseAsArray(this.h._drawitem);
                  optionsarr = parseAsArray(this.h._drawopt);
               }

               if (('_layout' in this.h) && !layout && ((this.is_online != "draw") || (itemsarr.length > 1)))
                  this.disp_kind = this.h._layout;

               if (('_toptitle' in this.h) && this.exclude_browser && document)
                  document.title = this.h._toptitle;

               if (gui_div)
                  this.prepareGuiDiv(gui_div.attr('id'), this.disp_kind);

               return openAllFiles();
            });

         if (gui_div)
            this.prepareGuiDiv(gui_div.attr('id'), this.disp_kind);

         return openAllFiles();
      }

      /** @summary Prepare div element - create layout and buttons
        * @private */
      prepareGuiDiv(myDiv, layout) {

         this.gui_div = (typeof myDiv == "string") ? myDiv : myDiv.attr('id');

         this.brlayout = new BrowserLayout(this.gui_div, this);

         this.brlayout.create(!this.exclude_browser);

         if (!this.exclude_browser) {
            let btns = this.brlayout.createBrowserBtns();

            JSROOT.require(['interactive']).then(inter => {
               inter.ToolbarIcons.createSVG(btns, inter.ToolbarIcons.diamand, 15, "toggle fix-pos browser")
                                  .style("margin","3px").on("click", () => this.createBrowser("fix", true));

               if (!this.float_browser_disabled)
                  inter.ToolbarIcons.createSVG(btns, inter.ToolbarIcons.circle, 15, "toggle float browser")
                                     .style("margin","3px").on("click", () => this.createBrowser("float", true));

               if (!this.status_disabled)
                  inter.ToolbarIcons.createSVG(btns, inter.ToolbarIcons.three_circles, 15, "toggle status line")
                                     .style("margin","3px").on("click", () => this.createStatusLine(0, "toggle"));
             });
         }

         this.setDisplay(layout, this.brlayout.drawing_divid());
      }

      /** @summary Create status line
        * @param {number} [height] - size of the status line
        * @param [mode] - false / true / "toggle"
        * @returns {Promise} when ready */
      createStatusLine(height, mode) {
         if (this.status_disabled || !this.gui_div || !this.brlayout)
            return Promise.resolve("");
         return this.brlayout.createStatusLine(height, mode);
      }

      /** @summary Redraw hierarchy
        * @desc works only when inspector or streamer info is displayed
        * @private */
      redrawObject(obj) {
         if (!this._inspector && !this._streamer_info) return false;
         if (this._streamer_info)
            this.h = createStreamerInfoContent(obj)
         else
            this.h = createInspectorContent(obj);
         return this.refreshHtml().then(() => { this.setTopPainter(); });
      }

      /** @summary Create browser elements
        * @returns {Promise} when completed */
      createBrowser(browser_kind, update_html) {

         if (!this.gui_div || this.exclude_browser || !this.brlayout)
            return Promise.resolve(false);

         let main = d3.select("#" + this.gui_div + " .jsroot_browser");

         // one requires top-level container
         if (main.empty())
            return Promise.resolve(false);

         if ((browser_kind==="float") && this.float_browser_disabled) browser_kind = "fix";

         if (!main.select('.jsroot_browser_area').empty()) {
            // this is case when browser created,
            // if update_html specified, hidden state will be toggled

            if (update_html) this.brlayout.toggleKind(browser_kind);

            return Promise.resolve(true);
         }

         let guiCode = `<p class="jsroot_browser_version"><a href="https://root.cern/js/">JSROOT</a> version <span style="color:green"><b>${JSROOT.version}</b></span></p>`;

         if (this.is_online) {
            guiCode += '<p> Hierarchy in <a href="h.json">json</a> and <a href="h.xml">xml</a> format</p>' +
                       '<div style="display:inline; vertical-align:middle; white-space: nowrap;">' +
                       '<label style="margin-right:5px"><input type="checkbox" name="monitoring" class="gui_monitoring"/>Monitoring</label>';
         } else if (!this.no_select) {
            let myDiv = d3.select("#"+this.gui_div),
                files = myDiv.attr("files") || "../files/hsimple.root",
                path = JSROOT.decodeUrl().get("path") || myDiv.attr("path") || "",
                arrFiles = files.split(';');

            guiCode += '<input type="text" value="" style="width:95%; margin:5px;border:2px;" class="gui_urlToLoad" title="input file name"/>' +
                       '<div style="display:flex;flex-direction:row;padding-top:5px">' +
                       '<select class="gui_selectFileName" style="flex:1;padding:2px;" title="select file name"' +
                       '<option value="" selected="selected"></option>';
            arrFiles.forEach(fname => { guiCode += `<option value="${path + fname}">${fname}</option>`; });
            guiCode += '</select>' +
                       '<input type="file" class="gui_localFile" accept=".root" style="display:none"/><output id="list" style="display:none"></output>' +
                       '<input type="button" value="..." class="gui_fileBtn" style="min-width:3em;padding:3px;margin-left:5px;margin-right:5px;" title="select local file for reading"/><br/>' +
                       '</div>' +
                       '<p id="gui_fileCORS"><small><a href="https://github.com/root-project/jsroot/blob/master/docs/JSROOT.md#reading-root-files-from-other-servers">Read docu</a>' +
                       ' how to open files from other servers.</small></p>' +
                       '<div style="display:flex;flex-direction:row">' +
                       '<input style="padding:3px;margin-right:5px" class="gui_ReadFileBtn" type="button" title="Read the Selected File" value="Load"/>' +
                       '<input style="padding:3px;margin-right:5px" class="gui_ResetUIBtn" type="button" title="Close all opened files and clear drawings" value="Reset"/>';
         } else if (this.no_select == "file") {
            guiCode += '<div style="display:flex;flex-direction:row">';
         }

         if (this.is_online || !this.no_select || this.no_select=="file")
            guiCode += '<select style="padding:2px;margin-right:5px;" title="layout kind" class="gui_layout"></select>' +
                       '</div>';

         guiCode += `<div id="${this.gui_div}_browser_hierarchy" class="jsroot_browser_hierarchy"></div>`;

         this.brlayout.setBrowserContent(guiCode);

         if (this.is_online)
             this.brlayout.setBrowserTitle('ROOT online server');
          else
             this.brlayout.setBrowserTitle('Read a ROOT file');

         let localfile_read_callback = null;

         if (!this.is_online && !this.no_select) {

            this.readSelectedFile = function() {
               let filename = main.select(".gui_urlToLoad").property('value').trim();
               if (!filename) return;

               if (filename.toLowerCase().lastIndexOf(".json") == filename.length-5)
                  this.openJsonFile(filename);
               else
                  this.openRootFile(filename);
            };

            main.select(".gui_selectFileName").property("value", "")
                 .on("change", evnt => main.select(".gui_urlToLoad").property("value", evnt.target.value));
            main.select(".gui_fileBtn").on("click", () => main.select(".gui_localFile").node().click());

            main.select(".gui_ReadFileBtn").on("click", () => this.readSelectedFile());

            main.select(".gui_ResetUIBtn").on("click", () => this.clearHierarchy(true));

            main.select(".gui_urlToLoad").on("keyup", evnt => {
               if (evnt.keyCode == 13) this.readSelectedFile();
            });

            main.select(".gui_localFile").on('change', evnt => {
               let files = evnt.target.files, promises = [];

               for (let n = 0; n < files.length; ++n) {
                  let f = files[n];
                  main.select(".gui_urlToLoad").property('value', f.name);
                  promises.push(this.openRootFile(f));
               }

               Promise.all(promises).then(() => {
                  if (localfile_read_callback) {
                     localfile_read_callback();
                     localfile_read_callback = null;
                  }
               });
            });

            this.selectLocalFile = function() {
               return new Promise(resolveFunc => {
                  localfile_read_callback = resolveFunc;
                  main.select(".gui_localFile").node().click();
               });
            };
         }

         let layout = main.select(".gui_layout");
         if (!layout.empty()) {
            ['simple', 'vert2', 'vert3', 'vert231', 'horiz2', 'horiz32', 'flex',
             'grid 2x2', 'grid 1x3', 'grid 2x3', 'grid 3x3', 'grid 4x4'].forEach(kind => layout.append("option").attr("value", kind).html(kind));

            layout.on('change', ev => this.setDisplay(ev.target.value || 'flex', this.gui_div + "_drawing"));
         }

         this.setDom(this.gui_div + '_browser_hierarchy');

         if (update_html) {
            this.refreshHtml();
            this.initializeBrowser();
         }

         return this.brlayout.toggleBrowserKind(browser_kind || "fix");
      }

      /** @summary Initialize browser elements */
      initializeBrowser() {

         let main = d3.select("#" + this.gui_div + " .jsroot_browser");
         if (main.empty() || !this.brlayout) return;

         if (this.brlayout) this.brlayout.adjustBrowserSize();

         let selects = main.select(".gui_layout").node();

         if (selects) {
            let found = false;
            for (let i in selects.options) {
               let s = selects.options[i].text;
               if (typeof s !== 'string') continue;
               if ((s == this.getLayout()) || (s.replace(/ /g,"") == this.getLayout())) {
                  selects.selectedIndex = i; found = true;
                  break;
               }
            }
            if (!found) {
               let opt = document.createElement('option');
               opt.innerHTML = opt.value = this.getLayout();
               selects.appendChild(opt);
               selects.selectedIndex = selects.options.length-1;
            }
         }

         if (this.is_online) {
            if (this.h && this.h._toptitle)
               this.brlayout.setBrowserTitle(this.h._toptitle);
            main.select(".gui_monitoring")
              .property('checked', this.isMonitoring())
              .on("click", evnt => {
                  this.enableMonitoring(evnt.target.checked);
                  this.updateItems();
               });
         } else if (!this.no_select) {
            let fname = "";
            this.forEachRootFile(item => { if (!fname) fname = item._fullurl; });
            main.select(".gui_urlToLoad").property("value", fname);
         }
      }

      /** @summary Enable monitoring mode */
      enableMonitoring(on) {
         this.setMonitoring(undefined, on);

         let chkbox = d3.select("#" + this.gui_div + " .jsroot_browser .gui_monitoring");
         if (!chkbox.empty() && (chkbox.property('checked') !== on))
            chkbox.property('checked', on);
      }

   } // class HierarchyPainter

   // ======================================================================================

   /** @summary tag item in hierarchy painter as streamer info
     * @desc this function used on THttpServer to mark streamer infos list
     * as fictional TStreamerInfoList class, which has special draw function
     * @private */
   JSROOT.markAsStreamerInfo = function(h,item,obj) {
      if (obj && (obj._typename=='TList'))
         obj._typename = 'TStreamerInfoList';
   }

   /** @summary Build gui without visible hierarchy browser
     * @private */
   JSROOT.buildNobrowserGUI = function(gui_element, gui_kind) {

      let myDiv = (typeof gui_element == 'string') ? d3.select('#' + gui_element) : d3.select(gui_element);
      if (myDiv.empty()) {
         alert('no div for simple nobrowser gui found');
         return Promise.resolve(null);
      }

      let online = false, drawing = false;
      if (gui_kind == 'online')
         online = true;
      else if (gui_kind == 'draw')
         online = drawing = true;

      if (myDiv.attr("ignoreurl") === "true")
         JSROOT.settings.IgnoreUrlOptions = true;

      jsrp.readStyleFromURL();

      let d = JSROOT.decodeUrl(), guisize = d.get("divsize");
      if (guisize) {
         guisize = guisize.split("x");
         if (guisize.length != 2) guisize = null;
      }

      if (guisize) {
         myDiv.style('position',"relative").style('width', guisize[0] + "px").style('height', guisize[1] + "px");
      } else {
         d3.select('html').style('height','100%');
         d3.select('body').style('min-height','100%').style('margin',0).style('overflow',"hidden");
         myDiv.style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('right',0).style('padding',1);
      }

      let hpainter = new HierarchyPainter('root', null);

      if (online) hpainter.is_online = drawing ? "draw" : "online";
      if (drawing) hpainter.exclude_browser = true;

      hpainter.start_without_browser = true; // indicate that browser not required at the beginning

      return hpainter.startGUI(myDiv, () => {
         if (!drawing) return hpainter;
         let func = JSROOT.findFunction('GetCachedObject');
         let obj = (typeof func == 'function') ? JSROOT.parse(func()) : null;
         if (obj) hpainter._cached_draw_object = obj;
         let opt = d.get("opt", "");

         if (d.has("websocket")) opt+=";websocket";

         return hpainter.display("", opt).then(() => hpainter);
      });
   }

   /** @summary Build main JSROOT GUI
     * @returns {Promise} when completed
     * @private  */
   JSROOT.buildGUI = function(gui_element, gui_kind) {
      let myDiv = (typeof gui_element == 'string') ? d3.select('#' + gui_element) : d3.select(gui_element);
      if (myDiv.empty()) return alert('no div for gui found');

      let online = false;
      if (gui_kind == "online") online = true;

      if (myDiv.attr("ignoreurl") === "true")
         JSROOT.settings.IgnoreUrlOptions = true;

      if (JSROOT.decodeUrl().has("nobrowser") || (myDiv.attr("nobrowser") && myDiv.attr("nobrowser")!=="false") || (gui_kind == "draw") || (gui_kind == "nobrowser"))
         return JSROOT.buildNobrowserGUI(gui_element, gui_kind);

      jsrp.readStyleFromURL();

      let hpainter = new JSROOT.HierarchyPainter('root', null);

      hpainter.is_online = online;

      return hpainter.startGUI(myDiv).then(() => {
         hpainter.initializeBrowser();
         return hpainter;
      });
   }


   /** @summary Display streamer info
     * @private */
   jsrp.drawStreamerInfo = function(dom, lst) {
      let painter = new HierarchyPainter('sinfo', dom, 'white');

      // in batch mode HTML drawing is not possible, just keep object reference for a minute
      if (JSROOT.batch_mode) {
         painter.selectDom().property("_json_object_", lst);
         return Promise.resolve(painter);
      }

      painter._streamer_info = true;
      painter.h = createStreamerInfoContent(lst);

      // painter.selectDom().style('overflow','auto');

      return painter.refreshHtml().then(() => {
         painter.setTopPainter();
         return painter;
      });
   }

   // ======================================================================================

   /** @summary Display inspector
     * @private */
   jsrp.drawInspector = function(dom, obj) {

      JSROOT.cleanup(dom);
      let painter = new HierarchyPainter('inspector', dom, 'white');

      // in batch mode HTML drawing is not possible, just keep object reference for a minute
      if (JSROOT.batch_mode) {
         painter.selectDom().property("_json_object_", obj);
         return Promise.resolve(painter);
      }

      painter.default_by_click = "expand"; // default action
      painter.with_icons = false;
      painter._inspector = true; // keep

      if (painter.selectDom().classed("jsroot_inspector"))
         painter.removeInspector = function() {
            this.selectDom().remove();
         }

      painter.fill_context = function(menu, hitem) {
         let sett = jsrp.getDrawSettings(hitem._kind, 'nosame');
         if (sett.opts)
            menu.addDrawMenu("nosub:Draw", sett.opts, function(arg) {
               if (!hitem || !hitem._obj) return;
               let obj = hitem._obj, ddom = this.selectDom().node();
               if (this.removeInspector) {
                  ddom = ddom.parentNode;
                  this.removeInspector();
                  if (arg == "inspect")
                     return this.showInspector(obj);
               }
               JSROOT.cleanup(ddom);
               JSROOT.draw(ddom, obj, arg);
            });
      }

      painter.h = createInspectorContent(obj);

      return painter.refreshHtml().then(() => {
         painter.setTopPainter();
         return painter;
      });
   }

   // ================================================================

   /**
    * @summary Base class to manage multiple document interface for drawings
    *
    * @memberof JSROOT
    * @private
    */

   class MDIDisplay extends JSROOT.BasePainter {
      /** @summary constructor */
      constructor(frameid) {
         super();
         this.frameid = frameid;
         if (frameid != "$batch$") {
            this.setDom(frameid);
            this.selectDom().property('mdi', this);
         }
         this.cleanupFrame = JSROOT.cleanup; // use standard cleanup function by default
         this.active_frame_title = ""; // keep title of active frame
      }

      /** @summary Assign func which called for each newly created frame */
      setInitFrame(func) {
         this.initFrame = func;
         this.forEachFrame(frame => func(frame));
      }

      /** @summary method called before new frame is created */
      beforeCreateFrame(title) { this.active_frame_title = title; }

      /** @summary method called after new frame is created
        * @private */
      afterCreateFrame(frame) {
         if (typeof this.initFrame == 'function')
            this.initFrame(frame);
         return frame;
      }

      /** @summary method dedicated to iterate over existing panels
        * @param {function} userfunc is called with arguments (frame)
        * @param {boolean} only_visible let select only visible frames */
      forEachFrame(userfunc, only_visible) {
         console.warn(`forEachFrame not implemented in MDIDisplay ${typeof userfunc} ${only_visible}`);
      }

      /** @summary method dedicated to iterate over existing panles
        * @param {function} userfunc is called with arguments (painter, frame)
        * @param {boolean} only_visible let select only visible frames */
      forEachPainter(userfunc, only_visible) {
         this.forEachFrame(frame => {
            new JSROOT.ObjectPainter(frame).forEachPainter(painter => userfunc(painter, frame));
         }, only_visible);
      }

      /** @summary Returns total number of drawings */
      numDraw() {
         let cnt = 0;
         this.forEachFrame(() => ++cnt);
         return cnt;
      }

      /** @summary Serach for the frame using item name */
      findFrame(searchtitle, force) {
         let found_frame = null;

         this.forEachFrame(frame => {
            if (d3.select(frame).attr('frame_title') == searchtitle)
               found_frame = frame;
         });

         if (!found_frame && force)
            found_frame = this.createFrame(searchtitle);

         return found_frame;
      }

      /** @summary Activate frame */
      activateFrame(frame) { this.active_frame_title = d3.select(frame).attr('frame_title'); }

      /** @summary Return active frame */
      getActiveFrame() { return this.findFrame(this.active_frame_title); }

      /** @summary perform resize for each frame
        * @protected */
      checkMDIResize(only_frame_id, size) {

         let resized_frame = null;

         this.forEachPainter((painter, frame) => {

            if (only_frame_id && (d3.select(frame).attr('id') != only_frame_id)) return;

            if ((painter.getItemName()!==null) && (typeof painter.checkResize == 'function')) {
               // do not call resize for many painters on the same frame
               if (resized_frame === frame) return;
               painter.checkResize(size);
               resized_frame = frame;
            }
         });
      }

      /** @summary Cleanup all drawings */
      cleanup() {
         this.active_frame_title = "";

         this.forEachFrame(this.cleanupFrame);

         this.selectDom().html("").property('mdi', null);
      }

   } // class MDIDisplay


   // ==================================================

   /**
    * @summary Custom MDI display
    *
    * @memberof JSROOT
    * @desc All HTML frames should be created before and add via {@link CustomDisplay.addFrame} calls
    * @private
    */

   class CustomDisplay extends MDIDisplay {
      constructor() {
         super("dummy");
         this.frames = {}; // array of configured frames
      }

      addFrame(divid, itemname) {
         if (!(divid in this.frames)) this.frames[divid] = "";

         this.frames[divid] += (itemname + ";");
      }

      forEachFrame(userfunc) {
         let ks = Object.keys(this.frames);
         for (let k = 0; k < ks.length; ++k) {
            let node = d3.select("#"+ks[k]);
            if (!node.empty())
               userfunc(node.node());
         }
      }

      createFrame(title) {
         this.beforeCreateFrame(title);

         let ks = Object.keys(this.frames);
         for (let k = 0; k < ks.length; ++k) {
            let items = this.frames[ks[k]];
            if (items.indexOf(title+";") >= 0)
               return d3.select("#"+ks[k]).node();
         }
         return null;
      }

      cleanup() {
         super.cleanup();
         this.forEachFrame(frame => d3.select(frame).html(""));
      }

   } // class CustomDisplay

   // ================================================

   /**
    * @summary Generic grid MDI display
    *
    * @memberof JSROOT
    * @private
    */

   class GridDisplay extends MDIDisplay {

    /** @summary Create GridDisplay instance
      * @param {string} frameid - where grid display is created
      * @param {string} kind - kind of grid
      * @desc  following kinds are supported
      *    - vertical or horizontal - only first letter matters, defines basic orientation
      *    - 'x' in the name disable interactive separators
      *    - v4 or h4 - 4 equal elements in specified direction
      *    - v231 -  created 3 vertical elements, first divided on 2, second on 3 and third on 1 part
      *    - v23_52 - create two vertical elements with 2 and 3 subitems, size ratio 5:2
      *    - gridNxM - normal grid layout without interactive separators
      *    - gridiNxM - grid layout with interactive separators
      *    - simple - no layout, full frame used for object drawings */
      constructor(frameid, kind, kind2) {

         super(frameid);

         this.framecnt = 0;
         this.getcnt = 0;
         this.groups = [];
         this.vertical = kind && (kind[0] == 'v');
         this.use_separarators = !kind || (kind.indexOf("x")<0);
         this.simple_layout = false;

         this.selectDom().style('overflow','hidden');

         if (kind === "simple") {
            this.simple_layout = true;
            this.use_separarators = false;
            this.framecnt = 1;
            return;
         }

         let num = 2, arr = undefined, sizes = undefined;

         if ((kind.indexOf("grid") == 0) || kind2) {
            if (kind2) kind = kind + "x" + kind2;
                  else kind = kind.substr(4).trim();
            this.use_separarators = false;
            if (kind[0] === "i") {
               this.use_separarators = true;
               kind = kind.substr(1);
            }

            let separ = kind.indexOf("x"), sizex, sizey;

            if (separ > 0) {
               sizey = parseInt(kind.substr(separ + 1));
               sizex = parseInt(kind.substr(0, separ));
            } else {
               sizex = sizey = parseInt(kind);
            }

            if (!Number.isInteger(sizex)) sizex = 3;
            if (!Number.isInteger(sizey)) sizey = 3;

            if (sizey > 1) {
               this.vertical = true;
               num = sizey;
               if (sizex > 1)
                  arr = new Array(num).fill(sizex);
            } else if (sizex > 1) {
               this.vertical = false;
               num = sizex;
            } else {
               this.simple_layout = true;
               this.use_separarators = false;
               this.framecnt = 1;
               return;
            }
            kind = "";
         }

         if (kind && kind.indexOf("_") > 0) {
            let arg = parseInt(kind.substr(kind.indexOf("_")+1), 10);
            if (Number.isInteger(arg) && (arg > 10)) {
               kind = kind.substr(0, kind.indexOf("_"));
               sizes = [];
               while (arg>0) {
                  sizes.unshift(Math.max(arg % 10, 1));
                  arg = Math.round((arg-sizes[0])/10);
                  if (sizes[0]===0) sizes[0]=1;
               }
            }
         }

         kind = kind ? parseInt(kind.replace( /^\D+/g, ''), 10) : 0;
         if (Number.isInteger(kind) && (kind > 1)) {
            if (kind < 10) {
               num = kind;
            } else {
               arr = [];
               while (kind>0) {
                  arr.unshift(kind % 10);
                  kind = Math.round((kind-arr[0])/10);
                  if (arr[0]==0) arr[0]=1;
               }
               num = arr.length;
            }
         }

         if (sizes && (sizes.length!==num)) sizes = undefined;

         if (!this.simple_layout)
            this.createGroup(this, this.selectDom(), num, arr, sizes);
      }

      /** @summary Create frames group
        * @private */
      createGroup(handle, main, num, childs, sizes) {

         if (!sizes) sizes = new Array(num);
         let sum1 = 0, sum2 = 0;
         for (let n = 0; n < num; ++n)
            sum1 += (sizes[n] || 1);
         for (let n = 0; n < num; ++n) {
            sizes[n] = Math.round(100 * (sizes[n] || 1) / sum1);
            sum2 += sizes[n];
            if (n==num-1) sizes[n] += (100-sum2); // make 100%
         }

         for (let cnt = 0; cnt < num; ++cnt) {
            let group = { id: cnt, drawid: -1, position: 0, size: sizes[cnt] };
            if (cnt > 0) group.position = handle.groups[cnt-1].position + handle.groups[cnt-1].size;
            group.position0 = group.position;

            if (!childs || !childs[cnt] || childs[cnt]<2) group.drawid = this.framecnt++;

            handle.groups.push(group);

            let elem = main.append("div").attr('groupid', group.id);

            if (handle.vertical)
               elem.style('float', 'bottom').style('height',group.size+'%').style('width','100%');
            else
               elem.style('float', 'left').style('width',group.size+'%').style('height','100%');

            if (group.drawid>=0) {
               elem.classed('jsroot_newgrid', true);
               if (typeof this.frameid === 'string')
                  elem.attr('id', this.frameid + "_" + group.drawid);
            } else {
               elem.style('display','flex').style('flex-direction', handle.vertical ? "row" : "column");
            }

            if (childs && (childs[cnt]>1)) {
               group.vertical = !handle.vertical;
               group.groups = [];
               elem.style('overflow','hidden');
               this.createGroup(group, elem, childs[cnt]);
            }
         }

         if (this.use_separarators && this.createSeparator)
            for (let cnt = 1; cnt < num; ++cnt)
               this.createSeparator(handle, main, handle.groups[cnt]);
      }

      /** @summary Handle interactive sepearator movement
        * @private */
      handleSeparator(elem, action) {
         let separ = d3.select(elem),
             parent = elem.parentNode,
             handle = separ.property('handle'),
             id = separ.property('separator_id'),
             group = handle.groups[id];

          const findGroup = grid => {
            let chld = parent.firstChild;
            while (chld) {
               if (chld.getAttribute("groupid") == grid)
                  return d3.select(chld);
               chld = chld.nextSibling;
            }
            // should never happen, but keep it here like
            return d3.select(parent).select(`[groupid='${grid}']`);
          }, setGroupSize = grid => {
             let name = handle.vertical ? 'height' : 'width',
                 size = handle.groups[grid].size+'%';
             findGroup(grid).style(name, size)
                            .selectAll(".jsroot_separator").style(name, size);
          }, resizeGroup = grid => {
             let sel = findGroup(grid);
             if (!sel.classed('jsroot_newgrid')) sel = sel.select(".jsroot_newgrid");
             sel.each(function() { JSROOT.resize(this); });
          };

         if (action == "start") {
            group.startpos = group.position;
            group.acc_drag = 0;
            return;
         }

         if (action == "end") {
            if (Math.abs(group.startpos - group.position) >= 0.5) {
               resizeGroup(id-1);
               resizeGroup(id);
             }
             return;
         }

         let pos;
         if (action == "restore") {
             pos = group.position0;
         } else if (handle.vertical) {
             group.acc_drag += action.dy;
             pos = group.startpos + ((group.acc_drag + 2) / parent.clientHeight) * 100;
         } else {
             group.acc_drag += action.dx;
             pos = group.startpos + ((group.acc_drag + 2) / parent.clientWidth) * 100;
         }

         let diff = group.position - pos;

         if (Math.abs(diff) < 0.3) return; // if no significant change, do nothing

         // do not change if size too small
         if (Math.min(handle.groups[id-1].size-diff, group.size+diff) < 3) return;

         handle.groups[id-1].size -= diff;
         group.size += diff;
         group.position = pos;

         separ.style(handle.vertical ? 'top' : 'left', `calc(${pos}% - 2px)`);

         setGroupSize(id-1);
         setGroupSize(id);

         if (action == "restore") {
             resizeGroup(id-1);
             resizeGroup(id);
         }

      }

      /** @summary Create group separator
        * @private */
      createSeparator(handle, main, group) {
         let separ = main.append("div");

         separ.classed('jsroot_separator', true)
              .classed(handle.vertical ? 'jsroot_hline' : 'jsroot_vline', true)
              .property('handle', handle)
              .property('separator_id', group.id)
              .style('position', 'absolute')
              .style(handle.vertical ? 'top' : 'left', `calc(${group.position}% - 2px)`)
              .style(handle.vertical ? 'width' : 'height', (handle.size || 100)+"%")
              .style(handle.vertical ? 'height' : 'width', '5px')
              .style('cursor', handle.vertical ? "ns-resize" : "ew-resize");

         let pthis = this, drag_move =
           d3.drag().on("start", function() { pthis.handleSeparator(this, "start"); })
                    .on("drag", function(evnt) { pthis.handleSeparator(this, evnt); })
                    .on("end", function() { pthis.handleSeparator(this, "end"); });

         separ.call(drag_move).on("dblclick", function() { pthis.handleSeparator(this, "restore"); });

         // need to get touches events handling in drag
         if (JSROOT.browser.touches && !main.on("touchmove"))
            main.on("touchmove", function() { });
      }


      /** @summary Call function for each frame */
      forEachFrame(userfunc) {
         if (this.simple_layout)
            userfunc(this.getGridFrame());
         else
            this.selectDom().selectAll('.jsroot_newgrid').each(function() {
               userfunc(this);
            });
      }

      /** @summary Returns active frame */
      getActiveFrame() {
         if (this.simple_layout)
            return this.getGridFrame();

         let found = super.getActiveFrame();
         if (found) return found;

         this.forEachFrame(frame => { if (!found) found = frame; });

         return found;
      }

      /** @summary Returns number of frames in grid layout */
      numGridFrames() { return this.framecnt; }

      /** @summary Return grid frame by its id */
      getGridFrame(id) {
         if (this.simple_layout)
            return this.selectDom('origin').node();
         let res = null;
         this.selectDom().selectAll('.jsroot_newgrid').each(function() {
            if (id-- === 0) res = this;
         });
         return res;
      }

      /** @summary Create new frame */
      createFrame(title) {
         this.beforeCreateFrame(title);

         let frame = null, maxloop = this.framecnt || 2;

         while (!frame && maxloop--) {
            frame = this.getGridFrame(this.getcnt);
            if (!this.simple_layout && this.framecnt)
               this.getcnt = (this.getcnt+1) % this.framecnt;

            if (d3.select(frame).classed("jsroot_fixed_frame")) frame = null;
         }

         if (frame) {
            this.cleanupFrame(frame);
            d3.select(frame).attr('frame_title', title);
         }

         return this.afterCreateFrame(frame);
      }

   } // class GridDisplay

   // ================================================

   /**
    * @summary Generic flexible MDI display
    *
    * @memberof JSROOT
    * @private
    */

   class FlexibleDisplay extends MDIDisplay {

      constructor(frameid) {
         super(frameid);
         this.cnt = 0; // use to count newly created frames
         this.selectDom().on('contextmenu', evnt => this.showContextMenu(evnt))
                         .style('overflow', 'auto');
      }

      /** @summary Cleanup all drawings */
      cleanup() {
         this.selectDom().style('overflow', null)
                         .on('contextmenu', null);
         this.cnt = 0;
         super.cleanup();
      }

      /** @summary call function for each frame */
      forEachFrame(userfunc,  only_visible) {
         if (typeof userfunc != 'function') return;

         let mdi = this, top = this.selectDom().select('.jsroot_flex_top');

         top.selectAll(".jsroot_flex_draw").each(function() {
            // check if only visible specified
            if (only_visible && (mdi.getFrameState(this) == "min")) return;

            userfunc(this);
         });
      }

      /** @summary return active frame */
      getActiveFrame() {
         let found = super.getActiveFrame();
         if (found && d3.select(found.parentNode).property("state") != "min") return found;

         found = null;
         this.forEachFrame(frame => { found = frame; }, true);
         return found;
      }

      /** @summary actiavte frame */
      activateFrame(frame) {
         if ((frame === 'first') || (frame === 'last')) {
            let res = null;
            this.forEachFrame(f => { if (frame == 'last' || !res) res = f; }, true);
            frame = res;
         }
         if (!frame) return;
         if (frame.getAttribute("class") != "jsroot_flex_draw") return;

         if (this.getActiveFrame() === frame) return;

         super.activateFrame(frame);

         let main = frame.parentNode;
         main.parentNode.append(main);

         if (this.getFrameState(frame) != "min") {
            jsrp.selectActivePad({ pp: jsrp.getElementCanvPainter(frame), active: true });
            JSROOT.resize(frame);
         }
      }

      /** @summary get frame state */
      getFrameState(frame) {
         let main = d3.select(frame.parentNode);
         return main.property("state");
      }

      /** @summary returns frame rect */
      getFrameRect(frame) {
         if (this.getFrameState(frame) == "max") {
            let top = this.selectDom().select('.jsroot_flex_top');
            return { x: 0, y: 0, w: top.node().clientWidth, h: top.node().clientHeight };
         }

         let main = d3.select(frame.parentNode), left = main.style('left'), top = main.style('top');

         return { x: parseInt(left.substr(0, left.length-2)), y: parseInt(top.substr(0, top.length-2)),
                  w: main.node().clientWidth, h: main.node().clientHeight };
      }

      /** @summary change frame state */
      changeFrameState(frame, newstate,no_redraw) {
         let main = d3.select(frame.parentNode),
             state = main.property("state"),
             top = this.selectDom().select('.jsroot_flex_top');

         if (state == newstate)
            return false;

         if (state == "normal")
             main.property('original_style', main.attr('style'));

         // clear any previous settings
         top.style('overflow', null);

         switch (newstate) {
            case "min":
               main.style("height","auto").style("width", "auto");
               main.select(".jsroot_flex_draw").style("display","none");
               break;
            case "max":
               main.style("height","100%").style("width", "100%").style('left','').style('top','');
               main.select(".jsroot_flex_draw").style("display", null);
               top.style('overflow', 'hidden');
               break;
            default:
               main.select(".jsroot_flex_draw").style("display", null);
               main.attr("style", main.property("original_style"));
         }

         main.select(".jsroot_flex_header").selectAll("button").each(function(d) {
            let btn = d3.select(this);
            if (((d.t == "minimize") && (newstate == "min")) ||
                ((d.t == "maximize") && (newstate == "max")))
                  btn.html("&#x259E;").attr("title", "restore");
            else
               btn.html(d.n).attr("title", d.t);
         });

         main.property("state", newstate);
         main.select(".jsroot_flex_resize").style("display", (newstate == "normal") ? null : "none");

         // adjust position of new minified rect
         if (newstate == "min") {
            const rect = this.getFrameRect(frame),
                  top = this.selectDom().select('.jsroot_flex_top'),
                  ww = top.node().clientWidth,
                  hh = top.node().clientHeight,
                  arr = [], step = 4,
                  crossX = (r1,r2) => ((r1.x <= r2.x) && (r1.x + r1.w >= r2.x)) || ((r2.x <= r1.x) && (r2.x + r2.w >= r1.x)),
                  crossY = (r1,r2) => ((r1.y <= r2.y) && (r1.y + r1.h >= r2.y)) || ((r2.y <= r1.y) && (r2.y + r2.h >= r1.y));

            this.forEachFrame(f => { if ((f!==frame) && (this.getFrameState(f) == "min")) arr.push(this.getFrameRect(f)); });

            rect.y = hh;
            do {
               rect.x = step;
               rect.y -= rect.h + step;
               let maxx = step, iscrossed = false;
               arr.forEach(r => {
                  if (crossY(r,rect)) {
                     maxx = Math.max(maxx, r.x + r.w + step);
                     if (crossX(r,rect)) iscrossed = true;
                  }
               });
               if (iscrossed) rect.x = maxx;
            } while ((rect.x + rect.w > ww - step) && (rect.y > 0));
            if (rect.y < 0) { rect.x = step; rect.y = hh - rect.h - step; }

            main.style("left", rect.x + "px").style("top", rect.y + "px");
         } else if (!no_redraw) {
            JSROOT.resize(frame);
         }

         return true;
      }

      /** @summary handle button click
        * @private */
      _clickButton(btn) {
         let kind = d3.select(btn).datum(),
             main = d3.select(btn.parentNode.parentNode),
             frame = main.select(".jsroot_flex_draw").node();

         if (kind.t == "close") {
            this.cleanupFrame(frame);
            main.remove();
            this.activateFrame('last'); // set active as last non-minfied window
            return;
         }

         let state = main.property("state"), newstate;
         if (kind.t == "maximize")
            newstate = (state == "max") ? "normal" : "max";
         else
            newstate = (state == "min") ? "normal" : "min";

         if (this.changeFrameState(frame, newstate))
            this.activateFrame(newstate != "min" ? frame : 'last');
      }

      /** @summary create new frame */
      createFrame(title) {

         this.beforeCreateFrame(title);

         let mdi = this,
             dom = this.selectDom(),
             top = dom.select(".jsroot_flex_top");

         if (top.empty())
            top = dom.append("div").classed("jsroot_flex_top", true);

         let w = top.node().clientWidth,
             h = top.node().clientHeight,
             main = top.append('div');

         main.html(`<div class="jsroot_flex_header"><p>${title}</p></div>
                    <div id="${this.frameid}_cont${this.cnt}" class="jsroot_flex_draw"></div>
                    <div class="jsroot_flex_resize">&#x25FF;</div>`);

         main.attr("class", "jsroot_flex_frame")
            .style("position", "absolute")
            .style('left', Math.round(w * (this.cnt % 5)/10) + "px")
            .style('top', Math.round(h * (this.cnt % 5)/10) + "px")
            .style('width', Math.round(w * 0.58) + "px")
            .style('height', Math.round(h * 0.58) + "px")
            .property("state", "normal")
            .select(".jsroot_flex_header")
            .on("click", function() { mdi.activateFrame(d3.select(this.parentNode).select(".jsroot_flex_draw").node()); })
            .selectAll("button")
            .data([{ n: '&#x2715;', t: "close" }, { n: '&#x2594;', t: "maximize" }, { n: '&#x2581;', t: "minimize" }])
            .enter()
            .append("button")
            .attr("type", "button")
            .attr("class", "jsroot_flex_btn")
            .attr("title", d => d.t)
            .html(d => d.n)
            .on("click", function() { mdi._clickButton(this); });

         const detectRightButton = event => {
            if ('buttons' in event) return event.buttons === 2;
            if ('which' in event) return event.which === 3;
            if ('button' in event) return event.button === 2;
            return false;
         };

         let moving_frame = null, moving_div = null, doing_move = false,
             drag_object = d3.drag().subject(Object), current = [];
         drag_object.on("start", function(evnt) {
            if (evnt.sourceEvent.target.type == "button")
               return mdi._clickButton(evnt.sourceEvent.target);

            if (detectRightButton(evnt.sourceEvent)) return;

            let main = d3.select(this.parentNode);
            if(!main.classed("jsroot_flex_frame") || (main.property("state") == "max")) return;

            doing_move = !d3.select(this).classed("jsroot_flex_resize");
            if (!doing_move && (main.property("state") == "min")) return;

            mdi.activateFrame(main.select(".jsroot_flex_draw").node());

            moving_div = top.append('div').classed("jsroot_flex_resizable_helper", true);

            moving_div.attr("style", main.attr("style"));

            if (main.property("state") == "min")
               moving_div.style("width", main.node().clientWidth + "px")
                         .style("height", main.node().clientHeight + "px");

            evnt.sourceEvent.preventDefault();
            evnt.sourceEvent.stopPropagation();

            moving_frame = main;
            current = [];

         }).on("drag", function(evnt) {
            if (!moving_div) return;
            evnt.sourceEvent.preventDefault();
            evnt.sourceEvent.stopPropagation();
            let changeProp = (i,name,dd) => {
               if (i >= current.length) {
                  let v = moving_div.style(name);
                  current[i] = parseInt(v.substr(0,v.length-2));
               }
               current[i] += dd;
               moving_div.style(name, Math.max(0, current[i])+"px");
            };
            if (doing_move) {
               changeProp(0, "left", evnt.dx);
               changeProp(1, "top", evnt.dy);
            } else {
               changeProp(0, "width", evnt.dx);
               changeProp(1, "height", evnt.dy);
            }
         }).on("end", function(evnt) {
            if (!moving_div) return;
            evnt.sourceEvent.preventDefault();
            evnt.sourceEvent.stopPropagation();
            if (doing_move) {
               moving_frame.style("left", moving_div.style("left"));
               moving_frame.style("top", moving_div.style("top"));
            } else {
               moving_frame.style("width", moving_div.style("width"));
               moving_frame.style("height", moving_div.style("height"));
            }
            moving_div.remove();
            moving_div = null;
            if (!doing_move)
               JSROOT.resize(moving_frame.select(".jsroot_flex_draw").node());
         });

         main.select(".jsroot_flex_header").call(drag_object);
         main.select(".jsroot_flex_resize").call(drag_object);

         let draw_frame = main.select('.jsroot_flex_draw')
                              .attr('frame_title', title)
                              .property('frame_cnt', this.cnt++)
                              .node();

         return this.afterCreateFrame(draw_frame);
      }

      /** @summary minimize all frames */
      minimizeAll() {
         this.forEachFrame(frame => this.changeFrameState(frame, "min"));
      }

      /** @summary close all frames */
      closeAllFrames() {
         let arr = [];
         this.forEachFrame(frame => arr.push(frame));
         arr.forEach(frame => {
            this.cleanupFrame(frame);
            d3.select(frame.parentNode).remove();
         });
      }

      /** @summary cascade frames */
      sortFrames(kind) {
         let arr = [];
         this.forEachFrame(frame => {
            let state = this.getFrameState(frame);
            if (state=="min") return;
            if (state == "max") this.changeFrameState(frame, "normal", true);
            arr.push(frame);
         });

         if (arr.length == 0) return;

         let top = this.selectDom(),
             w = top.node().clientWidth,
             h = top.node().clientHeight,
             dx = Math.min(40, Math.round(w*0.4/arr.length)),
             dy = Math.min(40, Math.round(h*0.4/arr.length)),
             nx = Math.ceil(Math.sqrt(arr.length)), ny = nx;

         // calculate number of divisions for "tile" sorting
         if ((nx > 1) && (nx*(nx-1) >= arr.length))
           if (w > h) ny--; else nx--;

         arr.forEach((frame,i) => {
            let main = d3.select(frame.parentNode);
            if (kind == "cascade")
               main.style('left', (i*dx) + "px")
                   .style('top', (i*dy) + "px")
                   .style('width', Math.round(w * 0.58) + "px")
                   .style('height', Math.round(h * 0.58) + "px");
            else
               main.style('left', Math.round(w/nx*(i%nx)) + "px")
                   .style('top', Math.round(h/ny*((i-i%nx)/nx)) + "px")
                   .style('width', Math.round(w/nx - 4) + "px")
                   .style('height', Math.round(h/ny - 4) + "px");
            JSROOT.resize(frame);
         });
      }

      /** @summary context menu */
      showContextMenu(evnt) {
         // handle context menu only for MDI area
         if ((evnt.target.getAttribute("class") != "jsroot_flex_top") || (this.numDraw() == 0)) return;

         evnt.preventDefault();

         let arr = [];
         this.forEachFrame(f => arr.push(f));
         let active = this.getActiveFrame();
         arr.sort((f1,f2) => { return  d3.select(f1).property('frame_cnt') < d3.select(f2).property('frame_cnt') ? -1 : 1; });

         jsrp.createMenu(evnt, this).then(menu => {
            menu.add("header:Flex");
            menu.add("Cascade", () => this.sortFrames("cascade"));
            menu.add("Tile", () => this.sortFrames("tile"));
            menu.add("Minimize all", () => this.minimizeAll());
            menu.add("Close all", () => this.closeAllFrames());
            menu.add("separator");

            arr.forEach((f,i) => menu.addchk((f===active), ((this.getFrameState(f) == "min") ? "[min] " : "") + d3.select(f).attr("frame_title"), i,
                         arg => {
                           let frame = arr[arg];
                           if (this.getFrameState(frame) == "min")
                              this.changeFrameState(frame, "normal");
                           this.activateFrame(frame);
                         }));

            menu.show();
         });
      }

   } // class FlexibleDisplay

   // ==================================================

   /**
    * @summary Batch MDI display
    *
    * @memberof JSROOT
    * @desc Can be used together with hierarchy painter in node.js
    * @private
    */

   class BatchDisplay extends MDIDisplay {

      constructor(width, height) {
         super("$batch$");
         this.frames = []; // array of configured frames
         this.width = width || 1200;
         this.height = height || 800;
      }

      forEachFrame(userfunc) {
         this.frames.forEach(userfunc)
      }

      createFrame(title) {
         this.beforeCreateFrame(title);

         let frame;
         if (!JSROOT.nodejs) {
            frame = d3.select('body').append("div").style("visible", "hidden");
         } else {
            if (!JSROOT._.nodejs_document) {
             // use eval while old minifier is not able to parse newest Node.js syntax
              const { JSDOM } = require("jsdom");
              JSROOT._.nodejs_window = (new JSDOM("<!DOCTYPE html>hello")).window;
              JSROOT._.nodejs_document = JSROOT._.nodejs_window.document; // used with three.js
              JSROOT._.nodejs_window.d3 = d3.select(JSROOT._.nodejs_document); //get d3 into the dom
            }
            frame = JSROOT._.nodejs_window.d3.select('body').append('div');
         }

         if (this.frames.length == 0)
            JSROOT._.svg_3ds = undefined;

         frame.attr("width", this.width).attr("height", this.height);
         frame.style("width", this.width + "px").style("height", this.height + "px");
         frame.attr("id","jsroot_batch_" + this.frames.length);
         frame.attr('frame_title', title);
         this.frames.push(frame.node());

         return this.afterCreateFrame(frame.node());
      }

      /** @summary Returns number of created frames */
      numFrames() { return this.frames.length; }

      /** @summary returns JSON representation if any
        * @desc Now works only for inspector, can be called once */
      makeJSON(id, spacing) {
         let frame = this.frames[id];
         if (!frame) return;
         let obj = d3.select(frame).property('_json_object_');
         if (obj) {
            d3.select(frame).property('_json_object_', null);
            return JSROOT.toJSON(obj, spacing);
         }
      }

      /** @summary Create SVG for specified frame id */
      makeSVG(id) {
         let frame = this.frames[id];
         if (!frame) return;
         let main = d3.select(frame);
         let has_workarounds = JSROOT._.svg_3ds && jsrp.processSvgWorkarounds;
         main.select('svg')
             .attr("xmlns", "http://www.w3.org/2000/svg")
             .attr("xmlns:xlink", "http://www.w3.org/1999/xlink")
             .attr("width", this.width)
             .attr("height", this.height)
             .attr("title", null).attr("style", null).attr("class", null).attr("x", null).attr("y", null);

         let svg = main.html();
         if (has_workarounds)
            svg = jsrp.processSvgWorkarounds(svg, id != this.frames.length-1);

         svg = jsrp.compressSVG(svg);

         main.remove();
         return svg;
      }

   } // class BatchDisplay

   // ===========================================================================================================

   /** @summary Create painter to perform tree drawing on server side
     * @private */
   JSROOT.createTreePlayer = function(player) {

      player.draw_first = true;

      player.configureOnline = function(itemname, url, askey, root_version, dflt_expr) {
         this.setItemName(itemname, "", this);
         this.url = url;
         this.root_version = root_version;
         this.askey = askey;
         this.dflt_expr = dflt_expr;
      }

      player.configureTree = function(tree) {
         this.local_tree = tree;
      }

      player.showExtraButtons = function(args) {
         let main = this.selectDom(),
            numentries = this.local_tree ? this.local_tree.fEntries : 0;

         main.select('.treedraw_more').remove(); // remove more button first

         main.select(".treedraw_buttons").node().innerHTML +=
             'Cut: <input class="treedraw_cut ui-corner-all ui-widget" style="width:8em;margin-left:5px" title="cut expression"></input>'+
             'Opt: <input class="treedraw_opt ui-corner-all ui-widget" style="width:5em;margin-left:5px" title="histogram draw options"></input>'+
             `Num: <input class="treedraw_number" type="number" min="0" max="${numentries}" step="1000" style="width:7em;margin-left:5px" title="number of entries to process (default all)"></input>`+
             `First: <input class="treedraw_first" type="number" min="0" max="${numentries}" step="1000" style="width:7em;margin-left:5px" title="first entry to process (default first)"></input>`+
             '<button class="treedraw_clear" title="Clear drawing">Clear</button>';

         main.select('.treedraw_exe').on("click", () => this.performDraw());
         main.select(".treedraw_cut").property("value", args && args.parse_cut ? args.parse_cut : "").on("change", () => this.performDraw());
         main.select(".treedraw_opt").property("value", args && args.drawopt ? args.drawopt : "").on("change", () => this.performDraw());
         main.select(".treedraw_number").attr("value", args && args.numentries ? args.numentries : ""); // .on("change", () => this.performDraw());
         main.select(".treedraw_first").attr("value", args && args.firstentry ? args.firstentry : ""); // .on("change", () => this.performDraw());
         main.select(".treedraw_clear").on("click", () => JSROOT.cleanup(this.drawid));
      }

      player.showPlayer = function(args) {

         let main = this.selectDom();

         this.drawid = "jsroot_tree_player_" + JSROOT._.id_counter++ + "_draw";

         let show_extra = args && (args.parse_cut || args.numentries || args.firstentry);

         main.html('<div style="display:flex; flex-flow:column; height:100%; width:100%;">'+
                      '<div class="treedraw_buttons" style="flex: 0 1 auto;margin-top:0.2em;">' +
                         '<button class="treedraw_exe" title="Execute draw expression" style="margin-left:0.5em">Draw</button>' +
                         'Expr:<input class="treedraw_varexp treedraw_varexp_info" style="width:12em;margin-left:5px" title="draw expression"></input>'+
                         '<label class="treedraw_varexp_info">\u24D8</label>' +
                        '<button class="treedraw_more">More</button>' +
                      '</div>' +
                      '<div style="flex: 0 1 auto"><hr/></div>' +
                      `<div id="${this.drawid}" style="flex: 1 1 auto; overflow:hidden;"></div>` +
                   '</div>');

         // only when main html element created, one can set painter
         // ObjectPainter allow such usage of methods from BasePainter
         this.setTopPainter();

         if (this.local_tree)
            main.select('.treedraw_buttons')
                .attr("title", "Tree draw player for: " + this.local_tree.fName);
         main.select('.treedraw_exe').on("click", () => this.performDraw());
         main.select('.treedraw_varexp')
             .attr("value", args && args.parse_expr ? args.parse_expr : (this.dflt_expr || "px:py"))
             .on("change", () => this.performDraw());
         main.select('.treedraw_varexp_info')
             .attr('title', "Example of valid draw expressions:\n" +
                            "  px - 1-dim draw\n" +
                            "  px:py - 2-dim draw\n" +
                            "  px:py:pz - 3-dim draw\n" +
                            "  px+py:px-py - use any expressions\n" +
                            "  px:py>>Graph - create and draw TGraph\n" +
                            "  px:py>>dump - dump extracted variables\n" +
                            "  px:py>>h(50,-5,5,50,-5,5) - custom histogram\n" +
                            "  px:py;hbins:100 - custom number of bins");

         if (show_extra)
            this.showExtraButtons(args);
         else
            main.select('.treedraw_more').on("click", () => this.showExtraButtons(args));

         this.checkResize();

         jsrp.registerForResize(this);
      }

      player.getValue = function(sel) {
         const elem = this.selectDom().select(sel);
         if (elem.empty()) return;
         const val = elem.property("value");
         if (val !== undefined) return val;
         return elem.attr("value");
      }

      player.performLocalDraw = function() {
         if (!this.local_tree) return;

         const frame = this.selectDom(),
               args = { expr: this.getValue('.treedraw_varexp') };

         if (frame.select('.treedraw_more').empty()) {
            args.cut = this.getValue('.treedraw_cut');
            if (!args.cut) delete args.cut;

            args.drawopt = this.getValue('.treedraw_opt');
            if (args.drawopt === "dump") { args.dump = true; args.drawopt = ""; }
            if (!args.drawopt) delete args.drawopt;

            args.numentries = parseInt(this.getValue('.treedraw_number'));
            if (!Number.isInteger(args.numentries)) delete args.numentries;

            args.firstentry = parseInt(this.getValue('.treedraw_first'));
            if (!Number.isInteger(args.firstentry)) delete args.firstentry;
         }

         if (args.drawopt) JSROOT.cleanup(this.drawid);

         const process_result = obj => JSROOT.redraw(this.drawid, obj);

         args.progress = process_result;

         this.local_tree.Draw(args).then(process_result);
      }

      player.getDrawOpt = function() {
         let res = "player",
             expr = this.getValue('.treedraw_varexp')
         if (expr) res += ":" + expr;
         return res;
      }

      player.performDraw = function() {

         if (this.local_tree)
            return this.performLocalDraw();

         let frame = this.selectDom(),
             url = this.url + '/exe.json.gz?compact=3&method=Draw',
             expr = this.getValue('.treedraw_varexp'),
             hname = "h_tree_draw", option = "",
             pos = expr.indexOf(">>");

         if (pos < 0) {
            expr += ">>" + hname;
         } else {
            hname = expr.substr(pos+2);
            if (hname[0]=='+') hname = hname.substr(1);
            let pos2 = hname.indexOf("(");
            if (pos2 > 0) hname = hname.substr(0, pos2);
         }

         if (frame.select('.treedraw_more').empty()) {
            let cut = this.getValue('.treedraw_cut'),
                nentries = this.getValue('.treedraw_number'),
                firstentry = this.getValue('.treedraw_first');

            option = this.getValue('.treedraw_opt');

            url += `&prototype="const char*,const char*,Option_t*,Long64_t,Long64_t"&varexp="${expr}"&selection="${cut}"`;

            // provide all optional arguments - default value kMaxEntries not works properly in ROOT6
            if (nentries=="") nentries = (this.root_version >= 394499) ? "TTree::kMaxEntries": "1000000000"; // kMaxEntries available since ROOT 6.05/03
            if (firstentry=="") firstentry = "0";
            url += `&option="${option}"&nentries=${nentries}&firstentry=${firstentry}`;
         } else {
            url += `&prototype="Option_t*"&opt="${expr}"`;
         }
         url += '&_ret_object_=' + hname;

         const submitDrawRequest = () => {
            JSROOT.httpRequest(url, 'object').then(res => {
               JSROOT.cleanup(this.drawid);
               JSROOT.draw(this.drawid, res, option);
            });
         };

         if (this.askey) {
            // first let read tree from the file
            this.askey = false;
            JSROOT.httpRequest(this.url + "/root.json", 'text').then(submitDrawRequest);
         } else {
            submitDrawRequest();
         }
      }

      player.checkResize = function(/*arg*/) {
         JSROOT.resize(this.drawid);
      }

      return player;
   }

   /** @summary function used with THttpServer to assign player for the TTree object
     * @private */
   JSROOT.drawTreePlayer = function(hpainter, itemname, askey, asleaf) {

      let item = hpainter.findItem(itemname),
          top = hpainter.getTopOnlineItem(item),
          draw_expr = "", leaf_cnt = 0;
      if (!item || !top) return null;

      if (asleaf) {
         draw_expr = item._name;
         while (item && !item._ttree) item = item._parent;
         if (!item) return null;
         itemname = hpainter.itemFullName(item);
      }

      let url = hpainter.getOnlineItemUrl(itemname);
      if (!url) return null;

      let root_version = top._root_version ? parseInt(top._root_version) : 396545; // by default use version number 6-13-01

      let mdi = hpainter.getDisplay();
      if (!mdi) return null;

      let frame = mdi.findFrame(itemname, true);
      if (!frame) return null;

      let divid = d3.select(frame).attr('id'),
          player = new JSROOT.BasePainter(divid);

      if (item._childs && !asleaf)
         for (let n=0;n<item._childs.length;++n) {
            let leaf = item._childs[n];
            if (leaf && leaf._kind && (leaf._kind.indexOf("ROOT.TLeaf")==0) && (leaf_cnt<2)) {
               if (leaf_cnt++ > 0) draw_expr+=":";
               draw_expr+=leaf._name;
            }
         }

      JSROOT.createTreePlayer(player);
      player.configureOnline(itemname, url, askey, root_version, draw_expr);
      player.showPlayer();

      return player;
   }

   /** @summary function used with THttpServer when tree is not yet loaded
     * @private */
   JSROOT.drawTreePlayerKey = function(hpainter, itemname) {
      return JSROOT.drawTreePlayer(hpainter, itemname, true);
   }

   /** @summary function used with THttpServer when tree is not yet loaded
     * @private */
   JSROOT.drawLeafPlayer = function(hpainter, itemname) {
      return JSROOT.drawTreePlayer(hpainter, itemname, false, true);
   }

   // export all functions and classes

   jsrp.drawList = drawList;

   jsrp.folderHierarchy = folderHierarchy;
   jsrp.taskHierarchy = taskHierarchy;
   jsrp.listHierarchy = listHierarchy;
   jsrp.objectHierarchy = objectHierarchy;
   jsrp.keysHierarchy = keysHierarchy;

   JSROOT.BrowserLayout = BrowserLayout;
   JSROOT.HierarchyPainter = HierarchyPainter;
   JSROOT.MDIDisplay = MDIDisplay;
   JSROOT.CustomDisplay = CustomDisplay;
   JSROOT.GridDisplay = GridDisplay;
   JSROOT.FlexibleDisplay = FlexibleDisplay;
   JSROOT.BatchDisplay = BatchDisplay;

   if (JSROOT.nodejs) module.exports = JSROOT;
   return JSROOT;

});
