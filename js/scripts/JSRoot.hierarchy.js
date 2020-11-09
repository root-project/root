/// @file JSRoot.hierarchy.js
/// Hierarchy display functionality

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   // ===========================================================================================

   /// function use to draw all items from TList or TObjArray inserted into the TCanvas list of primitives
   function drawList(divid, lst, opt, callback) {
      if (!lst || !lst.arr) return JSROOT.callBack(callback);

      let obj = {
        divid: divid,
        lst: lst,
        opt: opt,
        indx: -1,
        callback: callback,
        draw_next: function() {
           while (++this.indx < this.lst.arr.length) {
              let item = this.lst.arr[this.indx],
                  opt = (this.lst.opt && this.lst.opt[this.indx]) ? this.lst.opt[this.indx] : this.opt;
              if (!item) continue;
              return JSROOT.draw(this.divid, item, opt).then(this.draw_bind); // reenter loop via callback
           }

           return JSROOT.callBack(this.callback);
        }
      }

      obj.draw_bind = obj.draw_next.bind(obj);

      obj.draw_next();
   }

   // ===================== hierarchy scanning functions ==================================

   function FolderHierarchy(item, obj) {

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

   function TaskHierarchy(item, obj) {
      // function can be used for different derived classes
      // we show not only child tasks, but all complex data members

      if (!obj || !('fTasks' in obj) || (obj.fTasks === null)) return false;

      ObjectHierarchy(item, obj, { exclude: ['fTasks', 'fName'] } );

      if ((obj.fTasks.arr.length===0) && (item._childs.length==0)) { item._more = false; return true; }

      // item._childs = [];

      for ( let i = 0; i < obj.fTasks.arr.length; ++i) {
         let chld = obj.fTasks.arr[i];
         item._childs.push( {
            _name : chld.fName,
            _kind : "ROOT." + chld._typename,
            _obj : chld
         });
      }
      return true;
   }

   function ListHierarchy(folder, lst) {
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
      for ( let i = 0; i < lst.arr.length; ++i) {
         let obj = ismap ? lst.arr[i].first : lst.arr[i];

         let item;

         if (!obj || !obj._typename) {
            item = {
               _name: i.toString(),
               _kind: "ROOT.NULL",
               _title: "NULL",
               _value: "null",
               _obj: null
            }
         } else {
           item = {
             _name: obj.fName || obj.name,
             _kind: "ROOT." + obj._typename,
             _title: (obj.fTitle || "") + " type:"  +  obj._typename,
             _obj: obj
           };

           switch(obj._typename) {
              case 'TColor': item._value = jsrp.getRGBfromTColor(obj); break;
              case 'TText': item._value = obj.fTitle; break;
              case 'TLatex': item._value = obj.fTitle; break;
              case 'TObjString': item._value = obj.fString; break;
              default: if (lst.opt && lst.opt[i] && lst.opt[i].length) item._value = lst.opt[i];
           }

           if (do_context && JSROOT.canDraw(obj._typename)) item._direct_context = true;

           // if name is integer value, it should match array index
           if (!item._name || (!isNaN(parseInt(item._name)) && (parseInt(item._name)!==i))
               || (lst.arr.indexOf(obj)<i)) {
              item._name = i.toString();
           } else {
              // if there are several such names, add cycle number to the item name
              let indx = names.indexOf(obj.fName);
              if ((indx>=0) && (cnt[indx]>1)) {
                 item._cycle = cycle[indx]++;
                 item._keyname = item._name;
                 item._name = item._keyname + ";" + item._cycle;
              }
           }
         }

         folder._childs.push(item);
      }
      return true;
   }

   function KeysHierarchy(folder, keys, file, dirname) {

      if (keys === undefined) return false;

      folder._childs = [];

      for (let i = 0; i < keys.length; ++i) {
         let key = keys[i];

         let item = {
            _name : key.fName + ";" + key.fCycle,
            _cycle : key.fCycle,
            _kind : "ROOT." + key.fClassName,
            _title : key.fTitle,
            _keyname : key.fName,
            _readobj : null,
            _parent : folder
         };

         if (key.fObjlen > 1e5) item._title += ' (size: ' + (key.fObjlen/1e6).toFixed(1) + 'MB)';

         if ('fRealName' in key)
            item._realname = key.fRealName + ";" + key.fCycle;

         if (key.fClassName == 'TDirectory' || key.fClassName == 'TDirectoryFile') {
            let dir = null;
            if (dirname && file) dir = file.GetDir(dirname + key.fName);
            if (!dir) {
               item._more = true;
               item._expand = function(node, obj) {
                  // one can get expand call from child objects - ignore them
                  return KeysHierarchy(node, obj.fKeys);
               }
            } else {
               // remove cycle number - we have already directory
               item._name = key.fName;
               KeysHierarchy(item, dir.fKeys, file, dirname + key.fName + "/");
            }
         } else
         if ((key.fClassName == 'TList') && (key.fName == 'StreamerInfo')) {
            item._name = 'StreamerInfo';
            item._kind = "ROOT.TStreamerInfoList";
            item._title = "List of streamer infos for binary I/O";
            item._readobj = file.fStreamerInfos;
         }

         folder._childs.push(item);
      }

      return true;
   }

   function ObjectHierarchy(top, obj, args) {
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

         for (let k=0;k<obj.byteLength;++k) {
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

      let isarray = (proto.lastIndexOf('Array]') == proto.length-6) && (proto.indexOf('[object')==0) && !isNaN(obj.length),
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
      else
      if (top._obj !== obj) alert('object missmatch');

      if (!top._title) {
         if (obj._typename)
            top._title = "ROOT." + obj._typename;
         else
         if (isarray) top._title = "Array len: " + obj.length;
      }

      if (arrcompress) {
         for (let k=0;k<obj.length;) {

            let nextk = Math.min(k+10,obj.length), allsame = true, prevk = k;

            while (allsame) {
               allsame = true;
               for (let d=prevk;d<nextk;++d)
                  if (obj[k]!==obj[d]) allsame = false;

               if (allsame) {
                  if (nextk===obj.length) break;
                  prevk = nextk;
                  nextk = Math.min(nextk+10,obj.length);
               } else
               if (prevk !== k) {
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
            if (cnt>0) lastitem._name += ".." + lastkey;
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

            if ((proto.lastIndexOf('Array]') == proto.length-6) && (proto.indexOf('[object')==0)) {
               item._title = "array len=" + fld.length;
               simple = (proto != '[object Array]');
               if (fld.length === 0) {
                  item._value = "[ ]";
                  item._more = false; // hpainter will not try to expand again
               } else {
                  item._value = "[...]";
                  item._more = true;
                  item._expand = ObjectHierarchy;
                  item._obj = fld;
               }
            } else
            if (proto === "[object DataView]") {
               item._title = 'DataView len=' + fld.byteLength;
               item._value = "[...]";
               item._more = true;
               item._expand = ObjectHierarchy;
               item._obj = fld;
            }  else
            if (proto === "[object Date]") {
               item._more = false;
               item._title = 'Date';
               item._value = fld.toString();
               item._vclass = 'h_value_num';
            } else {

               if (fld.$kind || fld._typename)
                  item._kind = item._title = "ROOT." + (fld.$kind || fld._typename);

               if (fld._typename) {
                  item._title = fld._typename;
                  if (do_context && JSROOT.canDraw(fld._typename)) item._direct_context = true;
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
                     case 'TText': item._value = fld.fTitle; break;
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
         } else
         if ((typeof fld === 'number') || (typeof fld === 'boolean')) {
            simple = true;
            if (key == 'fBits')
               item._value = "0x" + fld.toString(16);
            else
               item._value = fld.toString();
            item._vclass = 'h_value_num';
         } else
         if (typeof fld === 'string') {
            simple = true;
            item._value = '&quot;' + fld.replace(/\&/g, '&amp;').replace(/\"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '&quot;';
            item._vclass = 'h_value_str';
         } else
         if (typeof fld === 'undefined') {
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

   /** @summary Parse string value as array.
    * @desc It could be just simple string:  "value" or
    * array with or without string quotes:  [element], ['elem1',elem2]
    * @private */
   let ParseAsArray = val => {

      let res = [];

      if (typeof val != 'string') return res;

      val = val.trim();
      if (val=="") return res;

      // return as array with single element
      if ((val.length<2) || (val[0]!='[') || (val[val.length-1]!=']')) {
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
                  let sub =  val.substring(last, indx).trim();
                  if ((sub.length>1) && (sub[0]==sub[sub.length-1]) && ((sub[0]=='"') || (sub[0]=="'")))
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
     * @summary special layout with three different areas for browser (left), status line (bottom) and central drawing
     * Main application is normal browser in JSROOT, but later one should be able to use it in ROOT6 canvas
     *
     * @class
     * @memberof JSROOT
     * @private
     */

   function BrowserLayout(id, hpainter, objpainter) {
      this.gui_div = id;
      this.hpainter = hpainter; // painter for brwoser area (if any)
      this.objpainter = objpainter; // painter for object area (if any)
      this.browser_kind = null; // should be 'float' or 'fix'
   }

   BrowserLayout.prototype.main = function() {
      return d3.select("#" + this.gui_div);
   }

   BrowserLayout.prototype.drawing_divid = function() {
      return this.gui_div + "_drawing";
   }

   BrowserLayout.prototype.CheckResize = function() {
      if (this.hpainter && (typeof this.hpainter.CheckResize == 'function'))
         this.hpainter.CheckResize();
      else if (this.objpainter && (typeof this.objpainter.CheckResize == 'function')) {
         this.objpainter.CheckResize(true);
      }
   }

   /** @summary method used to create basic elements
     * @desc should be called only once */
   BrowserLayout.prototype.Create = function(with_browser) {
      let main = this.main();

      main.append("div").attr("id", this.drawing_divid())
                        .classed("jsroot_draw_area", true)
                        .style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('right',0);

      if (with_browser) main.append("div").classed("jsroot_browser", true);
   }

   BrowserLayout.prototype.CreateBrowserBtns = function() {
      let br = this.main().select(".jsroot_browser");
      if (br.empty()) return;
      let btns = br.append("div").classed("jsroot_browser_btns", true).classed("jsroot", true);
      btns.style('position',"absolute").style("left","7px").style("top","7px");
      if (JSROOT.browser.touches) btns.style('opacity','0.2'); // on touch devices should be always visible
      return btns;
   }

   BrowserLayout.prototype.RemoveBrowserBtns = function() {
      this.main().select(".jsroot_browser").select(".jsroot_browser_btns").remove();
   }

   BrowserLayout.prototype.SetBrowserContent = function(guiCode) {
      let main = d3.select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return;

      main.insert('div', ".jsroot_browser_btns").classed('jsroot_browser_area', true)
          .style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('width','250px')
          .style('padding-left','5px')
          .style('display','flex').style('flex-direction', 'column')   /* use the flex model */
          .html("<p class='jsroot_browser_title'>title</p>" +  guiCode);
   }

   BrowserLayout.prototype.HasContent = function() {
      let main = d3.select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return false;
      return !main.select(".jsroot_browser_area").empty();
   }

   BrowserLayout.prototype.DeleteContent = function() {
      let main = d3.select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return;

      main.selectAll("*").remove();
      delete this.browser_visible;
   }

   BrowserLayout.prototype.HasStatus = function() {
      let main = d3.select("#"+this.gui_div+" .jsroot_browser");
      if (main.empty()) return false;

      let id = this.gui_div + "_status",
          line = d3.select("#"+id);

      return !line.empty();
   }

   BrowserLayout.prototype.CreateStatusLine = function(height, mode) {
      if (!this.gui_div) return '';
      JSROOT.require('jq2d').then(() => this.CreateStatusLine(height, mode));
      return this.gui_div + "_status";
   }

   // ==============================================================================


   /** @summary central function for expand of all online items
     * @private */
   function OnlineHierarchy(node, obj) {
      if (obj && node && ('_childs' in obj)) {

         for (let n=0;n<obj._childs.length;++n)
            if (obj._childs[n]._more || obj._childs[n]._childs)
               obj._childs[n]._expand = OnlineHierarchy;

         node._childs = obj._childs;
         obj._childs = null;
         return true;
      }

      return false;
   }

   // ==============================================================

   JSROOT.hpainter = null; // global pointer

   /**
     * @summary Painter of hierarchical structures
     *
     * @class
     * @memberof JSROOT
     * @extends JSROOT.BasePainter
     * @param {string} name - painter name
     * @param {string|object} frameid - place when painter drawn
     * @param {string} backgr - background color
     * @private
     */

   function HierarchyPainter(name, frameid, backgr) {
      JSROOT.BasePainter.call(this);
      this.name = name;
      this.h = null; // hierarchy
      this.with_icons = true;
      this.background = backgr;
      this.files_monitoring = !frameid; // by default files monitored when nobrowser option specified
      this.nobrowser = (frameid === null);
      if (!this.nobrowser) this.SetDivId(frameid); // this is required to be able cleanup painter

      // remember only very first instance
      if (!JSROOT.hpainter)
         JSROOT.hpainter = this;
   }

   HierarchyPainter.prototype = Object.create(JSROOT.BasePainter.prototype);

   /** @summary Cleanup hierarchy painter */
   HierarchyPainter.prototype.Cleanup = function() {
      // clear drawing and browser
      this.clear(true);

      JSROOT.BasePainter.prototype.Cleanup.call(this);

      if (JSROOT.hpainter === this)
         JSROOT.hpainter = null;
   }

   /** @summary Create file hierarchy */
   HierarchyPainter.prototype.FileHierarchy = function(file) {
      let painter = this;

      let folder = {
         _name : file.fFileName,
         _title : (file.fTitle ? (file.fTitle + ", path ") : "")  + file.fFullURL,
         _kind : "ROOT.TFile",
         _file : file,
         _fullurl : file.fFullURL,
         _localfile : file.fLocalFile,
         _had_direct_read : false,
         // this is central get method, item or itemname can be used
         _get : function(item, itemname, callback) {

            let fff = this; // file item

            if (item && item._readobj)
               return JSROOT.callBack(callback, item, item._readobj);

            if (item!=null) itemname = painter.itemFullName(item, fff);

            function ReadFileObject(file) {
               if (!fff._file) fff._file = file;

               if (!file) return JSROOT.callBack(callback, item, null);

               file.ReadObject(itemname).then(obj => {

                  // if object was read even when item did not exist try to reconstruct new hierarchy
                  if (!item && obj) {
                     // first try to found last read directory
                     let d = painter.Find({name:itemname, top:fff, last_exists:true, check_keys:true });
                     if ((d!=null) && ('last' in d) && (d.last!=fff)) {
                        // reconstruct only subdir hierarchy
                        let dir = file.GetDir(painter.itemFullName(d.last, fff));
                        if (dir) {
                           d.last._name = d.last._keyname;
                           let dirname = painter.itemFullName(d.last, fff);
                           KeysHierarchy(d.last, dir.fKeys, file, dirname + "/");
                        }
                     } else {
                        // reconstruct full file hierarchy
                        KeysHierarchy(fff, file.fKeys, file, "");
                     }
                     item = painter.Find({name:itemname, top: fff});
                  }

                  if (item) {
                     item._readobj = obj;
                     // remove cycle number for objects supporting expand
                     if ('_expand' in item) item._name = item._keyname;
                  }

                  JSROOT.callBack(callback, item, obj);
               });
            }

            if (fff._file) ReadFileObject(fff._file); else
            if (fff._localfile) JSROOT.openFile(fff._localfile).then(f => ReadFileObject(f)); else
            if (fff._fullurl) JSROOT.openFile(fff._fullurl).then(f => ReadFileObject(f));
         }
      };

      KeysHierarchy(folder, file.fKeys, file, "");

      return folder;
   }

   /** @summary Iterate over all items in hierarchy
    * @param {function} callback - function called for every item
    * @param {object} [top = null] - top item to start from
    * @private */
   HierarchyPainter.prototype.ForEach = function(callback, top) {

      if (!top) top = this.h;
      if (!top || (typeof callback != 'function')) return;
      function each_item(item) {
         callback(item);
         if ('_childs' in item)
            for (let n = 0; n < item._childs.length; ++n) {
               item._childs[n]._parent = item;
               each_item(item._childs[n]);
            }
      }

      each_item(top);
   }

   /** @summary Ssearch item in the hierarchy
    * @param {object|string} arg - item name or object with arguments
    * @param {string} arg.name -  item to search
    * @param {boolean} [arg.force = false] - specified elements will be created when not exists
    * @param {boolean} [arg.last_exists = false] -  when specified last parent element will be returned
    * @param {boolean} [arg.check_keys = false] - check TFile keys with cycle suffix
    * @param {object} [arg.top = null] - element to start search from
    * @private */
   HierarchyPainter.prototype.Find = function(arg) {

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
                   !isNaN(parseInt(localname.substr(1,localname.length-2)))) {
                  allow_index = true;
                  localname = localname.substr(1,localname.length-2);
               }

               // when search for the elements it could be allowed to check index
               if (allow_index) {
                  let indx = parseInt(localname);
                  if (!isNaN(indx) && (indx>=0) && (indx<top._childs.length))
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

   HierarchyPainter.prototype.itemFullName = function(node, uptoparent, compact) {

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

    /** @summary Eexecutes item marked as 'Command'
      * @desc If command requires additional arguments, they could be specified as extra arguments
      * Or they will be requested interactive
      * @param {String} itemname - name of command item
      * @param {Function} callback - function called with command result */
   HierarchyPainter.prototype.ExecuteCommand = function(itemname, callback) {

      let hitem = this.Find(itemname),
          url = this.GetOnlineItemUrl(hitem) + "/cmd.json",
          d3node = d3.select((typeof callback == 'function') ? undefined : callback);

      if ('_numargs' in hitem)
         for (let n = 0; n < hitem._numargs; ++n) {
            let argname = "arg" + (n+1), argvalue = null;
            if (n+2<arguments.length) argvalue = arguments[n+2];
            if (!argvalue && (typeof callback == 'object'))
               argvalue = prompt("Input argument " + argname + " for command " + hitem._name, "");
            if (!argvalue) return;
            url += ((n==0) ? "?" : "&") + argname + "=" + argvalue;
         }

      if (!d3node.empty()) {
         d3node.style('background','yellow');
         if (hitem && hitem._title) d3node.attr('title', "Executing " + hitem._title);
      }

      if (typeof callback != 'function')
         callback = res => {
            if (d3node.empty()) return;
            let col = ((res!=null) && (res!='false')) ? 'green' : 'red';
            if (hitem && hitem._title) d3node.attr('title', hitem._title + " lastres=" + res);
            d3node.style('background', col);
            setTimeout(() => d3node.style('background', ''), 2000);
            if ((col == 'green') && ('_hreload' in hitem)) this.reload();
            if ((col == 'green') && ('_update_item' in hitem)) this.updateItems(hitem._update_item.split(";"));
         }

      JSROOT.httpRequest(url, 'text').then(callback, callback);
   }

   HierarchyPainter.prototype.RefreshHtml = function(callback) {
      if (!this.divid) return JSROOT.callBack(callback);
      JSROOT.require('jq2d').then(() => this.RefreshHtml(callback));
   }

   /** @summary Get object item with specified name
     * @desc depending from provided option, same item can generate different object types */
   HierarchyPainter.prototype.get = function(arg, call_back, options) {

      if (arg===null) return JSROOT.callBack(call_back, null, null);

      let itemname, item;

      if (typeof arg === 'string') {
         itemname = arg;
      } else if (typeof arg === 'object') {
         if ((arg._parent!==undefined) && (arg._name!==undefined) && (arg._kind!==undefined)) item = arg; else
         if (arg.name!==undefined) itemname = arg.name; else
         if (arg.arg!==undefined) itemname = arg.arg; else
         if (arg.item!==undefined) item = arg.item;
      }

      if ((typeof itemname == 'string') && (itemname.indexOf("img:")==0))
         return JSROOT.callBack(call_back, null, {
            _typename: "TJSImage", // artificial class, can be created by users
            fName: itemname.substr(4)
         });

      if (item) itemname = this.itemFullName(item);
           else item = this.Find( { name: itemname, allow_index: true, check_keys: true } );

      // if item not found, try to find nearest parent which could allow us to get inside
      let d = (item!=null) ? null : this.Find({ name: itemname, last_exists: true, check_keys: true, allow_index: true });

      // if item not found, try to expand hierarchy central function
      // implements not process get in central method of hierarchy item (if exists)
      // if last_parent found, try to expand it
      if ((d !== null) && ('last' in d) && (d.last !== null)) {
         let parentname = this.itemFullName(d.last);

         // this is indication that expand does not give us better path to searched item
         if ((typeof arg == 'object') && ('rest' in arg))
            if ((arg.rest == d.rest) || (arg.rest.length <= d.rest.length))
               return JSROOT.callBack(call_back);

         return this.expand(parentname, res => {
            if (!res) JSROOT.callBack(call_back);
            let newparentname = this.itemFullName(d.last);
            if (newparentname.length>0) newparentname+="/";
            this.get( { name: newparentname + d.rest, rest: d.rest }, call_back, options);
         }, null, true);
      }

      if ((item !== null) && (typeof item._obj == 'object'))
         return JSROOT.callBack(call_back, item, item._obj);

      // normally search _get method in the parent items
      let curr = item;
      while (curr) {
         if (('_get' in curr) && (typeof curr._get == 'function'))
            return curr._get(item, null, call_back, options);
         curr = ('_parent' in curr) ? curr._parent : null;
      }

      JSROOT.callBack(call_back, item, null);
   }

   HierarchyPainter.prototype.draw = function(divid, obj, drawopt) {
      // just envelope, one should be able to redefine it for sub-classes
      return JSROOT.draw(divid, obj, drawopt);
   }

   HierarchyPainter.prototype.redraw = function(divid, obj, drawopt) {
      // just envelope, one should be able to redefine it for sub-classes
      return JSROOT.redraw(divid, obj, drawopt);
   }

   HierarchyPainter.prototype.player = function(itemname, option, call_back) {
      let item = this.Find(itemname);

      if (!item || !item._player) return JSROOT.callBack(call_back, null);

      JSROOT.require(item._prereq || '').then(() => {

         let player_func = JSROOT.findFunction(item._player);
         if (!player_func) return JSROOT.callBack(call_back, null);

         this.CreateDisplay(mdi => {
            let res = mdi ? player_func(this, itemname, option) : null;
            JSROOT.callBack(call_back, res);
         });
      });
   }

   HierarchyPainter.prototype.canDisplay = function(item, drawopt) {
      if (!item) return false;
      if (item._player) return true;
      if (item._can_draw !== undefined) return item._can_draw;
      if (drawopt == 'inspect') return true;
      let handle = JSROOT.getDrawHandle(item._kind, drawopt);
      return handle && (('func' in handle) || ('draw_field' in handle));
   }

   HierarchyPainter.prototype.isItemDisplayed = function(itemname) {
      let mdi = this.GetDisplay();
      if (!mdi) return false;

      return mdi.FindFrame(itemname) !== null;
   }

   HierarchyPainter.prototype.display = function(itemname, drawopt, call_back) {
      let h = this,
          painter = null,
          updating = false,
          item = null,
          display_itemname = itemname,
          frame_name = itemname,
          marker = "::_display_on_frame_::",
          p = drawopt ? drawopt.indexOf(marker) : -1;

      if (p>=0) {
         frame_name = drawopt.substr(p + marker.length);
         drawopt = drawopt.substr(0, p);
      }

      function display_callback(respainter) {
         if (!updating) JSROOT.progress();

         if (respainter && (typeof respainter === 'object') && (typeof respainter.SetItemName === 'function')) {
            respainter.SetItemName(display_itemname, updating ? null : drawopt, h); // mark painter as created from hierarchy
            if (item && !item._painter) item._painter = respainter;
         }

         // prevent calling many times
         let f = call_back;
         call_back = null;
         JSROOT.callBack(f, respainter || painter, display_itemname);
      }

      h.CreateDisplay(function(mdi) {

         if (!mdi) return display_callback();

         item = h.Find(display_itemname);

         if (item && ('_player' in item))
            return h.player(display_itemname, drawopt, display_callback);

         updating = (typeof(drawopt)=='string') && (drawopt.indexOf("update:")==0);

         if (updating) {
            drawopt = drawopt.substr(7);
            if (!item || item._doing_update) return display_callback();
            item._doing_update = true;
         }

         if (item && !h.canDisplay(item, drawopt)) return display_callback();

         let divid = "";
         if ((typeof(drawopt)=='string') && (drawopt.indexOf("divid:")>=0)) {
            let pos = drawopt.indexOf("divid:");
            divid = drawopt.slice(pos+6);
            drawopt = drawopt.slice(0, pos);
         }

         if (!updating) JSROOT.progress("Loading " + display_itemname);

         h.get(display_itemname, (resitem, obj) => {

            if (!updating) JSROOT.progress();

            if (!item) item = resitem;

            if (updating && item) delete item._doing_update;
            if (!obj) return display_callback();

            if (!updating) JSROOT.progress("Drawing " + display_itemname);

            if (divid.length > 0) {
               let draw_func = updating ? JSROOT.redraw : JSROOT.draw;
               return draw_func(divid, obj, drawopt).then(display_callback).catch(() => display_callback(null));
            }

            mdi.ForEachPainter((p, frame) => {
               if (p.GetItemName() != display_itemname) return;
               // verify that object was drawn with same option as specified now (if any)
               if (!updating && (drawopt!=null) && (p.GetItemDrawOpt()!=drawopt)) return;
               mdi.ActivateFrame(frame);

               let handle = null;
               if (obj._typename) handle = JSROOT.getDrawHandle("ROOT." + obj._typename);
               if (handle && handle.draw_field && obj[handle.draw_field])
                  obj = obj[handle.draw_field];

               if (p.RedrawObject(obj)) painter = p;
            });

            if (painter) return display_callback();

            if (updating) {
               console.warn(`something went wrong - did not found painter when doing update of ${display_itemname}`);
               return display_callback();
            }

            let frame = mdi.FindFrame(frame_name, true);
            d3.select(frame).html("");
            mdi.ActivateFrame(frame);

            JSROOT.draw(d3.select(frame).attr("id"), obj, drawopt).then(display_callback).catch(() => display_callback(null));

            if (JSROOT.settings.DragAndDrop)
               h.enable_dropping(frame, display_itemname);

         }, drawopt);
      });
   }

   HierarchyPainter.prototype.enable_dragging = function(/*element, itemname*/) {
      // here is not defined - implemented with jquery
   }

   HierarchyPainter.prototype.enable_dropping = function(/*frame, itemname*/) {
      // here is not defined - implemented with jquery
   }

   HierarchyPainter.prototype.dropitem = function(itemname, divid, opt, call_back) {
      if (opt && typeof opt === 'function') { call_back = opt; opt = ""; }
      if (opt===undefined) opt = "";

      let drop_callback = drop_painter => {
         if (drop_painter && (typeof drop_painter === 'object')) drop_painter.SetItemName(itemname, null, this);
         JSROOT.callBack(call_back);
      }

      if (itemname == "$legend")
         return JSROOT.require("hist").then(() => {
            let res = jsrp.produceLegend(divid, opt);
            JSROOT.callBack(drop_callback, res);
         });

      this.get(itemname, (item, obj) => {

         if (!obj) return JSROOT.callBack(call_back);

         let main_painter = JSROOT.get_main_painter(divid);

         if (main_painter && (typeof main_painter.PerformDrop === 'function'))
            return main_painter.PerformDrop(obj, itemname, item, opt).then(drop_callback);

         if (main_painter && main_painter.accept_drops)
            return JSROOT.draw(divid, obj, "same " + opt).then(drop_callback);

         this.CleanupFrame(divid);
         return JSROOT.draw(divid, obj, opt).then(drop_callback);
      });

      return true;
   }

   /** @summary Update items
     * @desc Argument is item name or array of string with items name
     * only already drawn items will be update with same draw option */
   HierarchyPainter.prototype.updateItems = function(items) {

      if (!this.disp || !items) return;

      let draw_items = [], draw_options = [];

      this.disp.ForEachPainter(p => {
         let itemname = p.GetItemName();
         if (!itemname || (draw_items.indexOf(itemname)>=0)) return;
         if (typeof items == 'array') {
            if (items.indexOf(itemname) < 0) return;
         } else {
            if (items != itemname) return;
         }
         draw_items.push(itemname);
         draw_options.push("update:" + p.GetItemDrawOpt());
      }, true); // only visible panels are considered

      if (draw_items.length > 0)
         this.displayAll(draw_items, draw_options);
   }


   /** @summary Update all elements
     * @desc Method can be used to fetch new objects and update all existing drawings
     * if only_auto_items specified, only automatic items will be updated */
   HierarchyPainter.prototype.updateAll = function(only_auto_items /*, only_items*/) {

      if (!this.disp) return;

      if (only_auto_items === "monitoring") only_auto_items = !this._monitoring_on;

      let allitems = [], options = [];

      // first collect items
      this.disp.ForEachPainter(p => {
         let itemname = p.GetItemName(),
             drawopt = p.GetItemDrawOpt();
         if ((typeof itemname != 'string') || (allitems.indexOf(itemname)>=0)) return;

         let item = this.Find(itemname), forced = false;
         if (!item || ('_not_monitor' in item) || ('_player' in item)) return;

         if ('_always_monitor' in item) {
            forced = true;
         } else {
            let handle = JSROOT.getDrawHandle(item._kind);
            if (handle && ('monitor' in handle)) {
               if ((handle.monitor===false) || (handle.monitor=='never')) return;
               if (handle.monitor==='always') forced = true;
            }
         }

         if (forced || !only_auto_items) {
            allitems.push(itemname);
            options.push("update:" + drawopt);
         }
      }, true); // only visible panels are considered

      // force all files to read again (normally in non-browser mode)
      if (this.files_monitoring && !only_auto_items)
         this.ForEachRootFile(item => {
            this.ForEach(fitem => { delete fitem._readobj; }, item);
            delete item._file;
         });

      if (allitems.length > 0)
         this.displayAll(allitems, options);
   }

   /** @summary Display all provided elements */
   HierarchyPainter.prototype.displayAll = function(items, options, call_back) {

      if (!items || (items.length == 0)) return JSROOT.callBack(call_back);

      let h = this;

      if (!options) options = [];
      while (options.length < items.length)
         options.push("");

      if ((options.length == 1) && (options[0] == "iotest")) {
         h.clear();
         d3.select("#" + h.disp_frameid).html("<h2>Start I/O test</h2>")

         let tm0 = new Date();
         return h.get(items[0], function(/*item, obj*/) {
            let tm1 = new Date();
            d3.select("#" + h.disp_frameid).append("h2").html("Item " + items[0] + " reading time = " + (tm1.getTime() - tm0.getTime()) + "ms");
            return JSROOT.callBack(call_back);
         });
      }

      let dropitems = new Array(items.length), dropopts = new Array(items.length), images = new Array(items.length);

      // First of all check that items are exists, look for cycle extension and plus sign
      for (let i = 0; i < items.length; ++i) {
         dropitems[i] = dropopts[i] = null;

         let item = items[i], can_split = true;

         if (item && item.indexOf("img:")==0) { images[i] = true; continue; }

         if (item && (item.length>1) && (item[0]=='\'') && (item[item.length-1]=='\'')) {
            items[i] = item.substr(1, item.length-2);
            can_split = false;
         }

         let elem = h.Find({ name: items[i], check_keys: true });
         if (elem) { items[i] = h.itemFullName(elem); continue; }

         if (can_split && (items[i][0]=='[') && (items[i][items[i].length-1]==']')) {
            dropitems[i] = ParseAsArray(items[i]);
            items[i] = dropitems[i].shift();
         } else
         if (can_split && (items[i].indexOf("+") > 0)) {
            dropitems[i] = items[i].split("+");
            items[i] = dropitems[i].shift();
         }

         if (dropitems[i] && dropitems[i].length > 0) {
            // allow to specify _same_ item in different file
            for (let j = 0; j < dropitems[i].length; ++j) {
               let pos = dropitems[i][j].indexOf("_same_");
               if ((pos>0) && (h.Find(dropitems[i][j])===null))
                  dropitems[i][j] = dropitems[i][j].substr(0,pos) + items[i].substr(pos);

               elem = h.Find({ name: dropitems[i][j], check_keys: true });
               if (elem) dropitems[i][j] = h.itemFullName(elem);
            }

            if ((options[i][0] == "[") && (options[i][options[i].length-1] == "]")) {
               dropopts[i] = ParseAsArray(options[i]);
               options[i] = dropopts[i].shift();
            } else
            if (options[i].indexOf("+") > 0) {
               dropopts[i] = options[i].split("+");
               options[i] = dropopts[i].shift();
            } else {
               dropopts[i] = [];
            }

            while (dropopts[i].length < dropitems[i].length) dropopts[i].push("");
         }

         // also check if subsequent items has _same_, than use name from first item
         let pos = items[i].indexOf("_same_");
         if ((pos>0) && !h.Find(items[i]) && (i>0))
            items[i] = items[i].substr(0,pos) + items[0].substr(pos);

         elem = h.Find({ name: items[i], check_keys: true });
         if (elem) items[i] = h.itemFullName(elem);
      }

      // now check that items can be displayed
      for (let n = items.length-1; n>=0; --n) {
         if (images[n]) continue;
         let hitem = h.Find(items[n]);
         if (!hitem || h.canDisplay(hitem, options[n])) continue;
         // try to expand specified item
         h.expand(items[n], null, null, true);
         items.splice(n, 1);
         options.splice(n, 1);
         dropitems.splice(n, 1);
      }

      if (items.length == 0) return JSROOT.callBack(call_back);

      let frame_names = new Array(items.length), items_wait = new Array(items.length);
      for (let n=0; n < items.length;++n) {
         items_wait[n] = 0;
         let fname = items[n], k = 0;
         if (items.indexOf(fname) < n) items_wait[n] = true; // if same item specified, one should wait first drawing before start next
         let p = options[n].indexOf("frameid:");
         if (p>=0) {
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
      for (let n=0; n<items.length;++n) {
         if (items_wait[n] !== 0) continue;
         let found_main = n;
         for (let k=0; k<items.length;++k)
            if ((items[n]===items[k]) && (options[k].indexOf('main')>=0)) found_main = k;
         for (let k=0; k<items.length;++k)
            if (items[n]===items[k]) items_wait[k] = (found_main != k);
      }

      h.CreateDisplay(function(mdi) {
         if (!mdi) return JSROOT.callBack(call_back);

         // Than create empty frames for each item
         for (let i = 0; i < items.length; ++i)
            if (options[i].indexOf('update:')!==0) {
               mdi.CreateFrame(frame_names[i]);
               options[i] += "::_display_on_frame_::"+frame_names[i];
            }

         function DropNextItem(indx, painter) {
            if (painter && dropitems[indx] && (dropitems[indx].length>0))
               return h.dropitem(dropitems[indx].shift(), painter.divid, dropopts[indx].shift(), DropNextItem.bind(h, indx, painter));

            dropitems[indx] = null; // mark that all drop items are processed
            items[indx] = null; // mark item as ready

            let isany = false;

            for (let cnt = 0; cnt < items.length; ++cnt) {
               if (dropitems[cnt]) isany = true;
               if (items[cnt]===null) continue; // ignore completed item
               isany = true;
               if (items_wait[cnt] && items.indexOf(items[cnt])===cnt) {
                  items_wait[cnt] = false;
                  h.display(items[cnt], options[cnt], DropNextItem.bind(h,cnt));
               }
            }

            // only when items drawn and all sub-items dropped, one could perform call-back
            if (!isany && call_back) {
               JSROOT.callBack(call_back);
               call_back = null;
            }
         }

         // We start display of all items parallel, but only if they are not the same
         for (let i = 0; i < items.length; ++i)
            if (!items_wait[i])
               h.display(items[i], options[i], DropNextItem.bind(h,i));
      });
   }

   /** @summary Reload hierarhcy and refresh html code */
   HierarchyPainter.prototype.reload = function() {
      if ('_online' in this.h)
         this.OpenOnline(this.h._online,() => this.RefreshHtml());
   }

   HierarchyPainter.prototype.UpdateTreeNode = function() {
      // dummy function, will be redefined when jquery part loaded
   }

   /** @summary activate (select) specified item
     * @param {boolean} [force] - if specified, all required sub-levels will be opened */
   HierarchyPainter.prototype.activate = function(items, force) {

      if (typeof items == 'string') items = [ items ];

      let active = [],  // array of elements to activate
          update = []; // array of elements to update
      this.ForEach(item => { if (item._background) { active.push(item); delete item._background; } });

      let mark_active = () => {
         if (typeof this.UpdateBackground !== 'function') return;

         for (let n=update.length-1;n>=0;--n)
            this.UpdateTreeNode(update[n]);

         for (let n=0;n<active.length;++n)
            this.UpdateBackground(active[n], force);
      }

      let find_next = (itemname, prev_found) => {
         if (itemname === undefined) {
            // extract next element
            if (items.length == 0) return mark_active();
            itemname = items.shift();
         }

         let hitem = this.Find(itemname);

         if (!hitem) {
            let d = this.Find({ name: itemname, last_exists: true, check_keys: true, allow_index: true });
            if (!d || !d.last) return find_next();
            d.now_found = this.itemFullName(d.last);

            if (force) {

               // if after last expand no better solution found - skip it
               if ((prev_found!==undefined) && (d.now_found === prev_found)) return find_next();

               return this.expand(d.now_found, res => {
                  if (!res) return find_next();
                  let newname = this.itemFullName(d.last);
                  if (newname.length>0) newname+="/";
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

            hitem._background = 'grey';
            if (active.indexOf(hitem)<0) active.push(hitem);
         }

         find_next();
      }

      if (force && this.brlayout) {
         if (!this.brlayout.browser_kind) return this.CreateBrowser('float', true, find_next);
         if (!this.brlayout.browser_visible) this.brlayout.ToggleBrowserVisisbility();
      }

      // use recursion
      find_next();
   }

   /** @summary expand specified item */
   HierarchyPainter.prototype.expand = function(itemname, call_back, d3cont, silent) {
      let hitem = this.Find(itemname);

      if (!hitem && d3cont) return JSROOT.callBack(call_back);

      let DoExpandItem = (_item, _obj, _name) => {
         if (!_name) _name = this.itemFullName(_item);

         let handle = _item._expand ? null : JSROOT.getDrawHandle(_item._kind, "::expand");

         if (_obj && handle && handle.expand_item) {
            _obj = _obj[handle.expand_item]; // just take specified field from the object
            if (_obj && _obj._typename)
               handle = JSROOT.getDrawHandle("ROOT."+_obj._typename, "::expand");
         }

         if (handle && handle.expand) {
            JSROOT.require(handle.prereq).then(() => {
               _item._expand = JSROOT.findFunction(handle.expand);
               if (_item._expand) return DoExpandItem(_item, _obj, _name);
               JSROOT.callBack(call_back);
            });
            return true;
         }

         // try to use expand function
         if (_obj && _item && (typeof _item._expand === 'function')) {
            if (_item._expand(_item, _obj)) {
               _item._isopen = true;
               if (_item._parent && !_item._parent._isopen) {
                  _item._parent._isopen = true; // also show parent
                  if (!silent) this.UpdateTreeNode(_item._parent);
               } else {
                  if (!silent) this.UpdateTreeNode(_item, d3cont);
               }
               JSROOT.callBack(call_back, _item);
               return true;
            }
         }

         if (_obj && ObjectHierarchy(_item, _obj)) {
            _item._isopen = true;
            if (_item._parent && !_item._parent._isopen) {
               _item._parent._isopen = true; // also show parent
               if (!silent) this.UpdateTreeNode(_item._parent);
            } else {
               if (!silent) this.UpdateTreeNode(_item, d3cont);
            }
            JSROOT.callBack(call_back, _item);
            return true;
         }

         return false;
      }

      if (hitem) {
         // item marked as it cannot be expanded, also top item cannot be changed
         if ((hitem._more === false) || (!hitem._parent && hitem._childs)) return JSROOT.callBack(call_back);

         if (hitem._childs && hitem._isopen) {
            hitem._isopen = false;
            if (!silent) this.UpdateTreeNode(hitem, d3cont);
            return JSROOT.callBack(call_back);
         }

         if (hitem._obj && DoExpandItem(hitem, hitem._obj, itemname)) return;
      }

      JSROOT.progress("Loading " + itemname);

      this.get(itemname, (item, obj) => {

         JSROOT.progress();

         if (obj && DoExpandItem(item, obj)) return;

         JSROOT.callBack(call_back);
      }, "hierarchy_expand" ); // indicate that we getting element for expand, can handle it differently

   }

   HierarchyPainter.prototype.GetTopOnlineItem = function(item) {
      if (item) {
         while (item && (!('_online' in item))) item = item._parent;
         return item;
      }

      if (!this.h) return null;
      if ('_online' in this.h) return this.h;
      if (this.h._childs && ('_online' in this.h._childs[0])) return this.h._childs[0];
      return null;
   }


   HierarchyPainter.prototype.ForEachJsonFile = function(call_back) {
      if (!this.h) return;
      if ('_jsonfile' in this.h)
         return JSROOT.callBack(call_back, this.h);

      if (this.h._childs)
         for (let n = 0; n < this.h._childs.length; ++n) {
            let item = this.h._childs[n];
            if ('_jsonfile' in item) JSROOT.callBack(call_back, item);
         }
   }

   HierarchyPainter.prototype.OpenJsonFile = function(filepath, call_back) {
      let isfileopened = false;
      this.ForEachJsonFile(function(item) { if (item._jsonfile==filepath) isfileopened = true; });
      if (isfileopened) return JSROOT.callBack(call_back);

      JSROOT.httpRequest(filepath, 'object').then(res => {
         let h1 = { _jsonfile: filepath, _kind: "ROOT." + res._typename, _jsontmp: res, _name: filepath.split("/").pop() };
         if (res.fTitle) h1._title = res.fTitle;
         h1._get = function(item,itemname,callback) {
            if (item._jsontmp)
               return JSROOT.callBack(callback, item, item._jsontmp);
            JSROOT.httpRequest(item._jsonfile, 'object').then(function(res) {
               item._jsontmp = res;
               JSROOT.callBack(callback, item, item._jsontmp);
            });
         }
         if (!this.h) this.h = h1; else
         if (this.h._kind == 'TopFolder') this.h._childs.push(h1); else {
            let h0 = this.h, topname = ('_jsonfile' in h0) ? "Files" : "Items";
            this.h = { _name: topname, _kind: 'TopFolder', _childs : [h0, h1] };
         }

         this.RefreshHtml(call_back);
      }).catch(function() { JSROOT.callBack(call_back); });
   }

   HierarchyPainter.prototype.ForEachRootFile = function(call_back) {
      if (!this.h) return;
      if ((this.h._kind == "ROOT.TFile") && this.h._file)
         return JSROOT.callBack(call_back, this.h);

      if (this.h._childs)
         for (let n = 0; n < this.h._childs.length; ++n) {
            let item = this.h._childs[n];
            if ((item._kind == 'ROOT.TFile') && ('_fullurl' in item))
               JSROOT.callBack(call_back, item);
         }
   }

   HierarchyPainter.prototype.OpenRootFile = function(filepath, call_back) {
      // first check that file with such URL already opened

      let isfileopened = false;
      this.ForEachRootFile(function(item) { if (item._fullurl===filepath) isfileopened = true; });
      if (isfileopened) return JSROOT.callBack(call_back);

      JSROOT.progress("Opening " + filepath + " ...");

      JSROOT.openFile(filepath).then(file => {

         let h1 = this.FileHierarchy(file);
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

         this.RefreshHtml(call_back);
      }).catch(() => {
         // make CORS warning
         if (!d3.select("#gui_fileCORS").style("background","red").empty())
             setTimeout(function() { d3.select("#gui_fileCORS").style("background",''); }, 5000);
         JSROOT.callBack(call_back, false);
      }).finally(() => JSROOT.progress());
   }

   HierarchyPainter.prototype.ApplyStyle = function(style, call_back) {
      if (!style)
         return JSROOT.callBack(call_back);

      if (typeof style === 'object') {
         if (style._typename === "TStyle")
            JSROOT.extend(JSROOT.gStyle, style);
         return JSROOT.callBack(call_back);
      }

      if (typeof style === 'string') {

         let item = this.Find({ name: style, allow_index: true, check_keys: true });

         if (item!==null)
            return this.get(item, (item2, obj) => this.ApplyStyle(obj, call_back));

         if (style.indexOf('.json') > 0)
            return JSROOT.httpRequest(style, 'object')
                         .then(res => this.ApplyStyle(res, call_back));
      }

      return JSROOT.callBack(call_back);
   }

   HierarchyPainter.prototype.GetFileProp = function(itemname) {
      let item = this.Find(itemname);
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

   HierarchyPainter.prototype.GetOnlineItemUrl = function(item) {
      // returns URL, which could be used to request item from the online server
      if (typeof item == "string") item = this.Find(item);
      let prnt = item;
      while (prnt && (prnt._online===undefined)) prnt = prnt._parent;
      return prnt ? (prnt._online + this.itemFullName(item, prnt)) : null;
   }

   HierarchyPainter.prototype.isOnlineItem = function(item) {
      return this.GetOnlineItemUrl(item)!==null;
   }

   HierarchyPainter.prototype.GetOnlineItem = function(item, itemname, callback, option) {
      // method used to request object from the http server

      let url = itemname, h_get = false, req = "", req_kind = "object", draw_handle = null;

      if (option === 'hierarchy_expand') { h_get = true; option = undefined; }

      if (item) {
         url = this.GetOnlineItemUrl(item);
         let func = null;
         if ('_kind' in item) draw_handle = JSROOT.getDrawHandle(item._kind);

         if (h_get) {
            req = 'h.json?compact=3';
            item._expand = OnlineHierarchy; // use proper expand function
         } else if ('_make_request' in item) {
            func = JSROOT.findFunction(item._make_request);
         } else if ((draw_handle!=null) && ('make_request' in draw_handle)) {
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

         if ((req.length==0) && (item._kind.indexOf("ROOT.")!=0))
           req = 'item.json.gz?compact=3';
      }

      if (!itemname && item && ('_cached_draw_object' in this) && (req.length == 0)) {
         // special handling for drawGUI when cashed
         let obj = this._cached_draw_object;
         delete this._cached_draw_object;
         return JSROOT.callBack(callback, item, obj);
      }

      if (req.length == 0) req = 'root.json.gz?compact=23';

      if (url.length > 0) url += "/";
      url += req;

      let itemreq = JSROOT.NewHttpRequest(url, req_kind, obj => {

         let func = null;

         if (!h_get && item && ('_after_request' in item)) {
            func = JSROOT.findFunction(item._after_request);
         } else if (draw_handle && ('after_request' in draw_handle))
            func = draw_handle.after_request;

         if (typeof func == 'function') {
            let res = func(this, item, obj, option, itemreq);
            if ((res!=null) && (typeof res == "object")) obj = res;
         }

         JSROOT.callBack(callback, item, obj);
      });

      itemreq.send(null);
   }

   HierarchyPainter.prototype.OpenOnline = function(server_address, user_callback) {
      let AdoptHierarchy = result => {
         this.h = result;
         if (!result) return;

         if (('_title' in this.h) && (this.h._title!='')) document.title = this.h._title;

         result._isopen = true;

         // mark top hierarchy as online data and
         this.h._online = server_address;

         this.h._get = (item, itemname, callback, option) => {
            this.GetOnlineItem(item, itemname, callback, option);
         }

         this.h._expand = OnlineHierarchy;

         let scripts = [], modules = [];
         this.ForEach(function(item) {
            if ('_childs' in item) item._expand = OnlineHierarchy;

            if ('_autoload' in item) {
               let arr = item._autoload.split(";");
               for (let n = 0; n < arr.length; ++n)
                  if ((arr[n].length>3) &&
                      ((arr[n].lastIndexOf(".js")==arr[n].length-3) ||
                      (arr[n].lastIndexOf(".css")==arr[n].length-4))) {
                     if (!scripts.find(elem => elem == arr[n])) scripts.push(arr[n]);
                  } else {
                     if (arr[n] && !modules.find(elem => elem ==arr[n])) modules.push(arr[n]);
                  }
            }
         });

         JSROOT.require(modules)
               .then(() => JSROOT.loadScript(scripts))
               .then(() => {
                  this.ForEach(item => {
                     if (!('_drawfunc' in item) || !('_kind' in item)) return;
                     let typename = "kind:" + item._kind;
                     if (item._kind.indexOf('ROOT.')==0) typename = item._kind.slice(5);
                     let drawopt = item._drawopt;
                     if (!JSROOT.canDraw(typename) || drawopt)
                        JSROOT.addDrawFunc({ name: typename, func: item._drawfunc, script: item._drawscript, opt: drawopt });
                  });

                  JSROOT.callBack(user_callback, this);
               });
      }

      if (!server_address) server_address = "";

      if (typeof server_address == 'object') {
         let h = server_address;
         server_address = "";
         return AdoptHierarchy(h);
      }

      JSROOT.httpRequest(server_address + "h.json?compact=3", 'object').then(AdoptHierarchy);
   }

   HierarchyPainter.prototype.GetOnlineProp = function(itemname) {
      let item = this.Find(itemname);
      if (!item) return null;

      let subname = item._name;
      while (item._parent) {
         item = item._parent;

         if ('_online' in item) {
            return {
               server : item._online,
               itemname : subname
            };
         }
         subname = item._name + "/" + subname;
      }

      return null;
   }

   HierarchyPainter.prototype.FillOnlineMenu = function(menu, onlineprop, itemname) {

      let node = this.Find(itemname),
          sett = JSROOT.getDrawSettings(node._kind, 'nosame;noinspect'),
          handle = JSROOT.getDrawHandle(node._kind),
          root_type = (typeof node._kind == 'string') ? node._kind.indexOf("ROOT.") == 0 : false;

      if (sett.opts && (node._can_draw !== false)) {
         sett.opts.push('inspect');
         menu.addDrawMenu("Draw", sett.opts, arg => this.display(itemname, arg));
      }

      if (!node._childs && (node._more !== false) && (node._more || root_type || sett.expand))
         menu.add("Expand", () => this.expand(itemname));

      if (handle && ('execute' in handle))
         menu.add("Execute", () => this.ExecuteCommand(itemname, menu.tree_node));

      let drawurl = onlineprop.server + onlineprop.itemname + "/draw.htm", separ = "?";
      if (this.IsMonitoring()) {
         drawurl += separ + "monitoring=" + this.MonitoringInterval();
         separ = "&";
      }

      if (sett.opts && (node._can_draw !== false))
         menu.addDrawMenu("Draw in new window", sett.opts, function(arg) { window.open(drawurl+separ+"opt=" +arg); });

      if (sett.opts && (sett.opts.length > 0) && root_type && (node._can_draw !== false))
         menu.addDrawMenu("Draw as png", sett.opts, function(arg) {
            window.open(onlineprop.server + onlineprop.itemname + "/root.png?w=400&h=300&opt=" + arg);
         });

      if ('_player' in node)
         menu.add("Player", () => this.player(itemname));
   }

   HierarchyPainter.prototype.Adopt = function(h) {
      this.h = h;
      this.RefreshHtml();
   }

   /** @summary Configures monitoring interval
    * @param interval - repetition interval in ms
    * @param flag - initial monitoring state
    * @private */
   HierarchyPainter.prototype.SetMonitoring = function(interval, monitor_on) {

      this._runMonitoring("cleanup");

      if (interval) {
         interval = parseInt(interval);
         if (!isNaN(interval) && (interval > 0)) {
            this._monitoring_interval = Math.max(100,interval);
            monitor_on = true;
         } else {
            this._monitoring_interval = 3000;
         }
      }

      this._monitoring_on = monitor_on;

      if (this.IsMonitoring())
         this._runMonitoring();
   }

   /** @summary Runs monitoring event loop
     * @private */
   HierarchyPainter.prototype._runMonitoring = function(arg) {
      if ((arg == "cleanup") || !this.IsMonitoring()) {
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
         this.updateAll("monitoring");
      }

      this._monitoring_handle = setTimeout(this._runMonitoring.bind(this,"frame"), this.MonitoringInterval());
   }

   /** @summary Returns configured monitoring interval in ms */
   HierarchyPainter.prototype.MonitoringInterval = function() {
      return this._monitoring_interval || 3000;
   }

   /** @summary Enable/disable monitoring */
   HierarchyPainter.prototype.EnableMonitoring = function(on) {
      this.SetMonitoring(undefined, on);
   }

   /** @summary Returns true when monitoring is enabled */
   HierarchyPainter.prototype.IsMonitoring = function() {
      return this._monitoring_on;
   }

   HierarchyPainter.prototype.SetDisplay = function(layout, frameid) {
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
         JSROOT.RegisterForResize(this);
      }
   }

   HierarchyPainter.prototype.GetLayout = function() {
      return this.disp_kind;
   }

   HierarchyPainter.prototype.ClearPainter = function(obj_painter) {
      this.ForEach(function(item) {
         if (item._painter === obj_painter) delete item._painter;
      });
   }

   HierarchyPainter.prototype.clear = function(withbrowser) {
      if (this.disp) {
         this.disp.Reset();
         delete this.disp;
      }

      let plainarr = [];

      this.ForEach(function(item) {
         delete item._painter; // remove reference on the painter
         // when only display cleared, try to clear all browser items
         if (!withbrowser && (typeof item.clear=='function')) item.clear();
         if (withbrowser) plainarr.push(item);
      });

      if (withbrowser) {
         // cleanup all monitoring loops
         this.EnableMonitoring(false);
         // simplify work for javascript and delete all (ok, most of) cross-references
         this.select_main().html("");
         plainarr.forEach(d => { delete d._parent; delete d._childs; delete d._obj; delete d._d3cont; });
         delete this.h;
      }
   }

   HierarchyPainter.prototype.GetDisplay = function() {
      return ('disp' in this) ? this.disp : null;
   }

   HierarchyPainter.prototype.CleanupFrame = function(divid) {
      // hook to perform extra actions when frame is cleaned

      let lst = JSROOT.cleanup(divid);

      // we remove all painters references from items
      if (lst && (lst.length>0))
         this.ForEach(function(item) {
            if (item._painter && lst.indexOf(item._painter)>=0) delete item._painter;
         });
   }

   /** @summary Creates configured JSROOT.MDIDisplay object
    * @param callback - called when mdi object created */
   HierarchyPainter.prototype.CreateDisplay = function(callback) {

      if ('disp' in this) {
         if ((this.disp.NumDraw() > 0) || (this.disp_kind == "custom")) return JSROOT.callBack(callback, this.disp);
         this.disp.Reset();
         delete this.disp;
      }

      // check that we can found frame where drawing should be done
      if (!document.getElementById(this.disp_frameid))
         return JSROOT.callBack(callback, null);

      if ((this.disp_kind == "simple") ||
          ((this.disp_kind.indexOf("grid") == 0) && (this.disp_kind.indexOf("gridi") < 0)))
           this.disp = new GridDisplay(this.disp_frameid, this.disp_kind);
      else
         return JSROOT.require('jq2d').then(this.CreateDisplay.bind(this, callback));

      if (this.disp)
         this.disp.CleanupFrame = this.CleanupFrame.bind(this);

      JSROOT.callBack(callback, this.disp);
   }

   /** @summary If possible, creates custom JSROOT.MDIDisplay for given item
   * @param itemname - name of item, for which drawing is created
   * @param custom_kind - display kind
   * @param callback - callback function, called when mdi object created */
   HierarchyPainter.prototype.CreateCustomDisplay = function(itemname, custom_kind, callback) {

      if (this.disp_kind != "simple")
         return this.CreateDisplay(callback);

      this.disp_kind = custom_kind;

      // check if display can be erased
      if (this.disp) {
         let num = this.disp.NumDraw();
         if ((num>1) || ((num==1) && !this.disp.FindFrame(itemname)))
            return this.CreateDisplay(callback);
         this.disp.Reset();
         delete this.disp;
      }

      this.CreateDisplay(callback);
   }

   HierarchyPainter.prototype.updateOnOtherFrames = function(painter, obj) {
      // function should update object drawings for other painters
      let mdi = this.disp, handle = null, isany = false;
      if (!mdi) return false;

      if (obj._typename) handle = JSROOT.getDrawHandle("ROOT." + obj._typename);
      if (handle && handle.draw_field && obj[handle.draw_field])
         obj = obj[handle.draw_field];

      mdi.ForEachPainter((p, frame) => {
         if ((p===painter) || (p.GetItemName() != painter.GetItemName())) return;
         mdi.ActivateFrame(frame);
         if (p.RedrawObject(obj)) isany = true;
      });
      return isany;
   }

   HierarchyPainter.prototype.CheckResize = function(size) {
      if (this.disp) this.disp.CheckMDIResize(null, size);
   }

   HierarchyPainter.prototype.StartGUI = function(gui_div, gui_call_back, url) {

      let d = JSROOT.decodeUrl(url);

      let GetOption = (opt) => {
         let res = d.get(opt, null);
         if (!res && gui_div && !gui_div.empty() && gui_div.node().hasAttribute(opt)) res = gui_div.attr(opt);
         return res;
      }

      let GetUrlOptionAsArray = (opt) => {

         let res = [];

         while (opt.length>0) {
            let separ = opt.indexOf(";");
            let part = (separ>0) ? opt.substr(0, separ) : opt;

            if (separ>0) opt = opt.substr(separ+1); else opt = "";

            let canarray = true;
            if (part[0]=='#') { part = part.substr(1); canarray = false; }

            let val = d.get(part,null);

            if (canarray) res = res.concat(ParseAsArray(val));
                     else if (val!==null) res.push(val);
         }
         return res;
      }

      let GetOptionAsArray = (opt) => {
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

            if (canarray) res = res.concat(ParseAsArray(val));
            else if (val!==null) res.push(val);
         }
         return res;
      }

      let prereq = GetOption('prereq') || "",
          filesdir = d.get("path") || "", // path used in normal gui
          filesarr = GetOptionAsArray("#file;files"),
          localfile = GetOption("localfile"),
          jsonarr = GetOptionAsArray("#json;jsons"),
          expanditems = GetOptionAsArray("expand"),
          itemsarr = GetOptionAsArray("#item;items"),
          optionsarr = GetOptionAsArray("#opt;opts"),
          monitor = GetOption("monitoring"),
          layout = GetOption("layout"),
          style = GetOptionAsArray("#style"),
          statush = 0, status = GetOption("status"),
          browser_kind = GetOption("browser"),
          browser_configured = !!browser_kind,
          title = GetOption("title");

      if (GetOption("float")!==null) { browser_kind = 'float'; browser_configured = true; } else
      if (GetOption("fix")!==null) { browser_kind = 'fix'; browser_configured = true; }

      this.no_select = GetOption("noselect");

      if (GetOption('files_monitoring')!==null) this.files_monitoring = true;

      if (title) document.title = title;

      let load = GetOption("load");
      if (load) prereq += ";io;gpad;";

      if (expanditems.length==0 && (GetOption("expand")==="")) expanditems.push("");

      if (filesdir) {
         for (let i=0;i<filesarr.length;++i) filesarr[i] = filesdir + filesarr[i];
         for (let i=0;i<jsonarr.length;++i) jsonarr[i] = filesdir + jsonarr[i];
      }

      if ((itemsarr.length==0) && GetOption("item")==="") itemsarr.push("");

      if ((jsonarr.length==1) && (itemsarr.length==0) && (expanditems.length==0)) itemsarr.push("");

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

      if (status==="no") status = null; else
      if (status==="off") { this.status_disabled = true; status = null; } else
      if (status==="on") status = true; else
      if (status!==null) { statush = parseInt(status); if (isNaN(statush) || (statush<5)) statush = 0; status = true; }
      if (this.no_select==="") this.no_select = true;

      if (!browser_kind) browser_kind = "fix"; else
      if (browser_kind==="no") browser_kind = ""; else
      if (browser_kind==="off") { browser_kind = ""; status = null; this.exclude_browser = true; }
      if (GetOption("nofloat")!==null) this.float_browser_disabled = true;

      if (this.start_without_browser) browser_kind = "";

      if (status || browser_kind) prereq += "jq2d;";

      this._topname = GetOption("topname");

      let OpenAllFiles = () => {
         if (prereq || load) {
            // load first prerequicities and then special script
            let req = prereq;
            prereq = "";
            if (!req) { req = load; load = ""; }
            return JSROOT.require(req).then(OpenAllFiles);
         }

         if (browser_kind) { this.CreateBrowser(browser_kind); browser_kind = ""; }
         if (status!==null) { this.CreateStatusLine(statush, status); status = null; }
         if (jsonarr.length > 0)
            this.OpenJsonFile(jsonarr.shift(), OpenAllFiles);
         else if (filesarr.length > 0)
            this.OpenRootFile(filesarr.shift(), OpenAllFiles);
         else if ((localfile!==null) && (typeof this.SelectLocalFile == 'function')) {
            localfile = null; this.SelectLocalFile(OpenAllFiles);
         } else if (expanditems.length > 0)
            this.expand(expanditems.shift(), OpenAllFiles);
         else if (style.length > 0)
            this.ApplyStyle(style.shift(), OpenAllFiles);
         else {
            this.RefreshHtml();
            this.displayAll(itemsarr, optionsarr, () => {
               if (itemsarr) this.RefreshHtml();
               this.SetMonitoring(monitor);
               JSROOT.callBack(gui_call_back);
           });
         }
      }

      let AfterOnlineOpened = () => {
         // check if server enables monitoring

         if (!this.exclude_browser && !browser_configured && ('_browser' in this.h)) {
            browser_kind = this.h._browser;
            if (browser_kind==="no") browser_kind = ""; else
            if (browser_kind==="off") { browser_kind = ""; status = null; this.exclude_browser = true; }
         }

         if (('_monitoring' in this.h) && !monitor)
            monitor = this.h._monitoring;

         if (('_loadfile' in this.h) && (filesarr.length==0))
            filesarr = ParseAsArray(this.h._loadfile);

         if (('_drawitem' in this.h) && (itemsarr.length==0)) {
            itemsarr = ParseAsArray(this.h._drawitem);
            optionsarr = ParseAsArray(this.h._drawopt);
         }

         if (('_layout' in this.h) && !layout && ((this.is_online != "draw") || (itemsarr.length > 1)))
            this.disp_kind = this.h._layout;

         if (('_toptitle' in this.h) && this.exclude_browser && document)
            document.title = this.h._toptitle;

         if (gui_div)
            this.PrepareGuiDiv(gui_div, this.disp_kind);

         OpenAllFiles();
      }

      let h0 = null;
      if (this.is_online) {
         if (typeof GetCachedHierarchy == 'function') h0 = GetCachedHierarchy();
         if (typeof h0 !== 'object') h0 = "";
      }

      if (h0 !== null)
         return this.OpenOnline(h0, AfterOnlineOpened);

      if (gui_div)
         this.PrepareGuiDiv(gui_div, this.disp_kind);

      OpenAllFiles();
   }

   HierarchyPainter.prototype.PrepareGuiDiv = function(myDiv, layout) {

      this.gui_div = myDiv.attr('id');

      this.brlayout = new BrowserLayout(this.gui_div, this);

      this.brlayout.Create(!this.exclude_browser);

      if (!this.exclude_browser) {
         let btns = this.brlayout.CreateBrowserBtns();

         JSROOT.require(['interactive']).then(inter => {
            inter.ToolbarIcons.CreateSVG(btns, inter.ToolbarIcons.diamand, 15, "toggle fix-pos browser")
                               .style("margin","3px").on("click", () => this.CreateBrowser("fix", true));

            if (!this.float_browser_disabled)
               inter.ToolbarIcons.CreateSVG(btns, inter.ToolbarIcons.circle, 15, "toggle float browser")
                                  .style("margin","3px").on("click", () => this.CreateBrowser("float", true));

            if (!this.status_disabled)
               inter.ToolbarIcons.CreateSVG(btns, inter.ToolbarIcons.three_circles, 15, "toggle status line")
                                  .style("margin","3px").on("click", () => this.CreateStatusLine(0, "toggle"));
          });
      }

      this.SetDisplay(layout, this.brlayout.drawing_divid());
   }

   HierarchyPainter.prototype.CreateStatusLine = function(height, mode) {
      if (this.status_disabled || !this.gui_div || !this.brlayout) return '';
      return this.brlayout.CreateStatusLine(height, mode);
   }

   HierarchyPainter.prototype.CreateBrowser = function(browser_kind, update_html, call_back) {
      if (this.gui_div)
         JSROOT.require('jq2d').then(() => this.CreateBrowser(browser_kind, update_html, call_back));
   }

   // ======================================================================================

   /** @summary tag item in hierarchy painter as streamer info
     * @desc this function used on THttpServer to mark streamer infos list
     * as fictional TStreamerInfoList class, which has special draw function
     * @private */
   JSROOT.MarkAsStreamerInfo = function(h,item,obj) {
      if (obj && (obj._typename=='TList'))
         obj._typename = 'TStreamerInfoList';
   }

   /** @summary Build gui without visisble hierarchy browser
     * @desc avoid loading of jquery part
     * @private */
   JSROOT.BuildNobrowserGUI = function() {
      let myDiv = d3.select('#simpleGUI'),
          online = false, drawing = false;

      if (myDiv.empty()) {
         online = true;
         myDiv = d3.select('#onlineGUI');
         if (myDiv.empty()) { myDiv = d3.select('#drawGUI'); drawing = true; }
         if (myDiv.empty()) return alert('no div for simple nobrowser gui found');
      }

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

      let hpainter = new JSROOT.HierarchyPainter('root', null);

      if (online) hpainter.is_online = drawing ? "draw" : "online";
      if (drawing) hpainter.exclude_browser = true;

      hpainter.start_without_browser = true; // indicate that browser not required at the beginning

      return new Promise(resolveFunc => {

         hpainter.StartGUI(myDiv, () => {
            if (!drawing) return resolveFunc(hpainter);

            let func = JSROOT.findFunction('GetCachedObject');
            let obj = (typeof func == 'function') ? JSROOT.parse(func()) : null;
            if (obj) hpainter._cached_draw_object = obj;
            let opt = d.get("opt", "");

            if (d.has("websocket")) opt+=";websocket";

            hpainter.display("", opt, () => resolveFunc(hpainter));
         });
      });
   }

   /** @summary Display streamer info
     * @private */
   jsrp.drawStreamerInfo = function(divid, lst) {
      let painter = new JSROOT.HierarchyPainter('sinfo', divid, 'white');

      painter.h = { _name : "StreamerInfo", _childs : [] };

      for (let i = 0; i < lst.arr.length; ++i) {
         let entry = lst.arr[i]

         if (entry._typename == "TList") continue;

         if (typeof (entry.fName) == 'undefined') {
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

         painter.h._childs.push(item);

         if (typeof entry.fElements == 'undefined') continue;
         for ( let l = 0; l < entry.fElements.arr.length; ++l) {
            let elem = entry.fElements.arr[l];
            if (!elem || !elem.fName) continue;
            let info = elem.fTypeName + " " + elem.fName,
                title = elem.fTypeName + " type:" + elem.fType;
            if (elem.fArrayDim===1)
               info += "[" + elem.fArrayLength + "]";
            else
               for (let dim=0;dim<elem.fArrayDim;++dim)
                  info+="[" + elem.fMaxIndex[dim] + "]";
            if (elem.fBaseVersion===4294967295) info += ":-1"; else
            if (elem.fBaseVersion!==undefined) info += ":" + elem.fBaseVersion;
            info += ";";
            if (elem.fTitle != '') info += " // " + elem.fTitle;

            item._childs.push({ _name : info, _title: title, _kind: elem.fTypeName, _icon: (elem.fTypeName == 'BASE') ? "img_class" : "img_member" });
         }
         if (item._childs.length == 0) delete item._childs;
      }

      // painter.select_main().style('overflow','auto');

      painter.RefreshHtml(function() {
         painter.SetDivId(divid);
         painter.DrawingReady();
      });

      return painter;
   }

   // ======================================================================================

   /** @summary Display inspector
     * @private */
   jsrp.drawInspector = function(divid, obj) {

      JSROOT.cleanup(divid);

      let painter = new JSROOT.HierarchyPainter('inspector', divid, 'white');
      painter.default_by_click = "expand"; // default action
      painter.with_icons = false;
      painter.h = { _name: "Object", _title: "", _click_action: "expand", _nosimple: false, _do_context: true };
      if ((typeof obj.fTitle === 'string') && (obj.fTitle.length>0))
         painter.h._title = obj.fTitle;

      if (painter.select_main().classed("jsroot_inspector"))
         painter.removeInspector = function() {
            this.select_main().remove();
         }

      if (obj._typename)
         painter.h._title += "  type:" + obj._typename;

      if ((typeof obj.fName === 'string') && (obj.fName.length > 0))
         painter.h._name = obj.fName;

      // painter.select_main().style('overflow','auto');

      painter.fill_context = function(menu, hitem) {
         let sett = JSROOT.getDrawSettings(hitem._kind, 'nosame');
         if (sett.opts)
            menu.addDrawMenu("nosub:Draw", sett.opts, function(arg) {
               if (!hitem || !hitem._obj) return;
               let obj = hitem._obj, divid = this.divid;
               if (this.removeInspector) {
                  divid = this.select_main().node().parentNode;
                  this.removeInspector();
                  if (arg == "inspect")
                     return this.ShowInspector(obj);
               }
               JSROOT.cleanup(divid);
               JSROOT.draw(divid, obj, arg);
            });
      }

      if (JSROOT.isRootCollection(obj)) {
         painter.h._name = obj.name || obj._typename;
         ListHierarchy(painter.h, obj);
      } else {
         ObjectHierarchy(painter.h, obj);
      }
      painter.RefreshHtml(function() {
         painter.SetDivId(divid);
         painter.DrawingReady();
      });

      return painter.Promise();
   }

   // ================================================================

   // MDIDisplay - class to manage multiple document interface for drawings

   class MDIDisplay extends JSROOT.BasePainter {
      constructor(frameid) {
         super();
         this.frameid = frameid;
         this.SetDivId(frameid);
         this.select_main().property('mdi', this);
         this.CleanupFrame = JSROOT.cleanup; // use standard cleanup function by default
         this.active_frame_title = ""; // keep title of active frame
      }

      BeforeCreateFrame(title) { this.active_frame_title = title; }

      /** method dedicated to iterate over existing panels
        * @param {function} userfunc is called with arguments (frame)
        * @param {boolean} only_visible let select only visible frames */
      ForEachFrame(/* userfunc, only_visible */) { console.warn("ForEachFrame not implemented in MDIDisplay"); }

      /** method dedicated to iterate over existing panles
        * @param {function} userfunc is called with arguments (painter, frame)
        * @param {boolean} only_visible let select only visible frames */
      ForEachPainter(userfunc, only_visible) {

         this.ForEachFrame(frame => {
            let dummy = new JSROOT.ObjectPainter();
            dummy.SetDivId(frame, -1);
            dummy.ForEachPainter(painter => userfunc(painter, frame));
         }, only_visible);
      }

      NumDraw() {
         let cnt = 0;
         this.ForEachFrame(() => ++cnt);
         return cnt;
      }

      FindFrame(searchtitle, force) {
         let found_frame = null;

         this.ForEachFrame(frame => {
            if (d3.select(frame).attr('frame_title') == searchtitle)
               found_frame = frame;
         });

         if (!found_frame && force)
            found_frame = this.CreateFrame(searchtitle);

         return found_frame;
      }

      ActivateFrame(frame) { this.active_frame_title = d3.select(frame).attr('frame_title'); }

      GetActiveFrame() { return this.FindFrame(this.active_frame_title); }

      CheckMDIResize(only_frame_id, size) {
         // perform resize for each frame
         let resized_frame = null;

         this.ForEachPainter((painter, frame) => {

            if (only_frame_id && (d3.select(frame).attr('id') != only_frame_id)) return;

            if ((painter.GetItemName()!==null) && (typeof painter.CheckResize == 'function')) {
               // do not call resize for many painters on the same frame
               if (resized_frame === frame) return;
               painter.CheckResize(size);
               resized_frame = frame;
            }
         });
      }

      Reset() {
         this.active_frame_title = "";

         this.ForEachFrame(this.CleanupFrame);

         this.select_main().html("").property('mdi', null);
      }

      Draw(title, obj, drawopt) {
         // draw object with specified options
         if (!obj) return;

         if (!JSROOT.canDraw(obj._typename, drawopt)) return;

         let frame = this.FindFrame(title, true);

         this.ActivateFrame(frame);

         return JSROOT.redraw(frame, obj, drawopt);
      }
   } // class MDIDisplay


   // ==================================================

   class CustomDisplay extends MDIDisplay {
      constructor() {
         super("dummy");
         this.frames = {}; // array of configured frames
      }

      AddFrame(divid, itemname) {
         if (!(divid in this.frames)) this.frames[divid] = "";

         this.frames[divid] += (itemname + ";");
      }

      ForEachFrame(userfunc /* ,  only_visible */) {
         let ks = Object.keys(this.frames);
         for (let k = 0; k < ks.length; ++k) {
            let node = d3.select("#"+ks[k]);
            if (!node.empty())
               JSROOT.callBack(userfunc, node.node());
         }
      }

      CreateFrame(title) {
         this.BeforeCreateFrame(title);

         let ks = Object.keys(this.frames);
         for (let k = 0; k < ks.length; ++k) {
            let items = this.frames[ks[k]];
            if (items.indexOf(title+";")>=0)
               return d3.select("#"+ks[k]).node();
         }
         return null;
      }

      Reset() {
         super.Reset();
         this.ForEachFrame(frame => d3.select(frame).html(""));
      }
   } // class CustomDisplay

   // ================================================

   class GridDisplay extends MDIDisplay {

      constructor(frameid, kind, kind2) {
         // following kinds are supported
         //  vertical or horizontal - only first letter matters, defines basic orientation
         //   'x' in the name disable interactive separators
         //   v4 or h4 - 4 equal elements in specified direction
         //   v231 -  created 3 vertical elements, first divided on 2, second on 3 and third on 1 part
         //   v23_52 - create two vertical elements with 2 and 3 subitems, size ratio 5:2
         //   gridNxM - normal grid layout without interactive separators
         //   gridiNxM - grid layout with interactive separators
         //   simple - no layout, full frame used for object drawings

         super(frameid);

         this.framecnt = 0;
         this.getcnt = 0;
         this.groups = [];
         this.vertical = kind && (kind[0] == 'v');
         this.use_separarators = !kind || (kind.indexOf("x")<0);
         this.simple_layout = false;

         this.select_main().style('overflow','hidden');

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
            if (kind[0]==="i") {
               this.use_separarators = true;
               kind = kind.substr(1);
            }

            let separ = kind.indexOf("x"), sizex = 3, sizey = 3;

            if (separ > 0) {
               sizey = parseInt(kind.substr(separ + 1));
               sizex = parseInt(kind.substr(0, separ));
            } else {
               sizex = sizey = parseInt(kind);
            }

            if (isNaN(sizex)) sizex = 3;
            if (isNaN(sizey)) sizey = 3;

            if (sizey>1) {
               this.vertical = true;
               num = sizey;
               if (sizex>1) {
                  arr = new Array(num);
                  for (let k=0;k<num;++k) arr[k] = sizex;
               }
            } else
            if (sizex > 1) {
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

         if (kind && kind.indexOf("_")>0) {
            let arg = parseInt(kind.substr(kind.indexOf("_")+1), 10);
            if (!isNaN(arg) && (arg>10)) {
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
         if (kind && (kind>1)) {
            if (kind<10) {
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
            this.CreateGroup(this, this.select_main(), num, arr, sizes);
      }

      CreateGroup(handle, main, num, childs, sizes) {

         if (!sizes) sizes = new Array(num);
         let sum1 = 0, sum2 = 0;
         for (let n=0;n<num;++n) sum1 += (sizes[n] || 1);
         for (let n=0;n<num;++n) {
            sizes[n] = Math.round(100 * (sizes[n] || 1) / sum1);
            sum2 += sizes[n];
            if (n==num-1) sizes[n] += (100-sum2); // make 100%
         }

         for (let cnt = 0; cnt<num; ++cnt) {
            let group = { id: cnt, drawid: -1, position: 0, size: sizes[cnt] };
            if (cnt>0) group.position = handle.groups[cnt-1].position + handle.groups[cnt-1].size;
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
               this.CreateGroup(group, elem, childs[cnt]);
            }
         }

         if (this.use_separarators && this.CreateSeparator)
            for (let cnt=1;cnt<num;++cnt)
               this.CreateSeparator(handle, main, handle.groups[cnt]);
      }

      ForEachFrame(userfunc /*, only_visible */) {
         if (this.simple_layout)
            userfunc(this.GetFrame());
         else
            this.select_main().selectAll('.jsroot_newgrid').each(function() {
               userfunc(d3.select(this).node());
            });
      }

      GetActiveFrame() {
         if (this.simple_layout) return this.GetFrame();

         let found = super.GetActiveFrame();
         if (found) return found;

         this.ForEachFrame(frame => { if (!found) found = frame; }, true);

         return found;
      }

      ActivateFrame(frame) { this.active_frame_title = d3.select(frame).attr('frame_title'); }

      GetFrame(id) {
         if (this.simple_layout)
            return this.select_main('origin').node();
         let res = null;
         this.select_main().selectAll('.jsroot_newgrid').each(function() {
            if (id-- === 0) res = this;
         });
         return res;
      }

      NumGridFrames() { return this.framecnt; }

      CreateFrame(title) {
         this.BeforeCreateFrame(title);

         let frame = null, maxloop = this.framecnt || 2;

         while (!frame && maxloop--) {
            frame = this.GetFrame(this.getcnt);
            if (!this.simple_layout && this.framecnt)
               this.getcnt = (this.getcnt+1) % this.framecnt;

            if (d3.select(frame).classed("jsroot_fixed_frame")) frame = null;
         }

         if (frame) {
            this.CleanupFrame(frame);
            d3.select(frame).attr('frame_title', title);
         }

         return frame;
      }

   } // class GridDisplay


   // export all functions and classes

   jsrp.drawList = drawList;

   jsrp.FolderHierarchy = FolderHierarchy;
   jsrp.ObjectHierarchy = ObjectHierarchy;
   jsrp.TaskHierarchy = TaskHierarchy;
   jsrp.ListHierarchy = ListHierarchy;
   jsrp.KeysHierarchy = KeysHierarchy;

   JSROOT.BrowserLayout = BrowserLayout;
   JSROOT.HierarchyPainter = HierarchyPainter;

   JSROOT.MDIDisplay = MDIDisplay;
   JSROOT.CustomDisplay = CustomDisplay;
   JSROOT.GridDisplay = GridDisplay;

   return JSROOT;

});
