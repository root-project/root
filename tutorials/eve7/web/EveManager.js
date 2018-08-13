/// @file EveManager.js

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootCore'], factory );
   } else
      if (typeof exports === 'object' && typeof module !== 'undefined') {
	 factory(require("./JSRootCore.js"));
      } else {

	 if (typeof JSROOT == 'undefined')
            throw new Error('JSROOT is not defined', 'JSRootPainter.more.js');

	 factory(JSROOT);
      }
} (function(JSROOT) {

   "use strict";

   // JSROOT.sources.push("evemgr");

   /** @namespace JSROOT.EVE */
   /// Holder of all TGeo-related functions and classes
   JSROOT.EVE = {};

   function EveManager() {
      this.map = {}; // use object, not array
      this.childs = [];
      this.last_json = null;
      this.scene_changed = null;

      this.hrecv = []; // array of receivers of highlight messages

       this.EChangeBits = { "kCBColorSelection":1, "kCBTransBBox":2, "kCBObjProps":4, "kCBVisibility":8};
   }

   /** Returns element with given ID */
   EveManager.prototype.GetElement = function(id) {
      return this.map[id];
   }

   /** Configure dependency for given element id - invoke function when element changed */
   EveManager.prototype.Register = function(id, receiver, func_name) {
      var elem = this.GetElement(id);

      if (!elem) return;

      if (!elem.$receivers) elem.$receivers = [];

      elem.$receivers.push({obj:receiver, func:func_name});
   }

   /** returns master id for given element id
    * master id used for highlighting element in all dependent views */
   EveManager.prototype.GetMasterId = function(elemid) {
      var elem = this.map[elemid];
      if (!elem) return elemid;
      return elem.fMasterId || elemid;
   }

   EveManager.prototype.RegisterHighlight = function(receiver, func_name) {
      for (var n=0;n<this.hrecv.length;++n) {
         var el = this.hrecv[n];
         if (el.obj === receiver) {
            el.func = func_name;
            return;
         }
      }
      console.log("ADDDD ENTRY", func_name, receiver);
      this.hrecv.push({obj:receiver, func:func_name});
   }

   /** Invoke highlight on all dependent views.
    * One specifies element id and on/off state.
    * If timeout configured, actual execution will be postponed on given time interval */

   EveManager.prototype.ProcessHighlight = function(sender, masterid, timeout) {
      if (this.highligt_timer) {
         clearTimeout(this.highligt_timer);
         delete this.highligt_timer;
      }

      if (timeout) {
         this.highligt_timer = setTimeout(this.ProcessHighlight.bind(this, sender, masterid), timeout);
         return;
      }

      for (var n=0; n<this.hrecv.length; ++n) {
         var el = this.hrecv[n];
         if (el.obj!==sender)
            el.obj[el.func](masterid);
      }
   }

   EveManager.prototype.Unregister = function(receiver) {
      for (var n=0;n<this.hrecv.length;++n) {
         var el = this.hrecv[n];
         if (el.obj===receiver)
            this.hrecv.splice(n, 1);
      }
      // TODO: cleanup object from all receivers
   }

   // mark object and all its parents as modified
   EveManager.prototype.MarkModified = function(id) {
      while (id) {
         var elem = this.GetElement(id);
         if (!elem) return;
         if (elem.$receivers) elem.$modified = true; // mark only elements which have receivers
         id = elem.fMotherId;
      }
   }

   EveManager.prototype.ProcessModified = function() {
      for (var id in this.map) {
         var elem = this.map[id];
         if (!elem || !elem.$modified) continue;

         for (var k=0;k<elem.$receivers.length;++k) {
            var f = elem.$receivers[k];
            f.obj[f.func](id, elem);
         }

         delete elem.$modified;
      }
   }

   EveManager.prototype.ProcessData = function(arr) {
      if (!arr) return;

      if (arr[0].content == "TEveScene::StreamElements")
         return this.Update(arr);

      if (arr[0].content == "TEveManager::DestroyElementsOf")
         return this.DestroyElements(arr);
   }

   EveManager.prototype.Update = function(arr) {
      this.last_json = null;
      // console.log("JSON", arr[0]);

      if (arr[0].fTotalBinarySize)
         this.last_json = arr;

      for (var n=1; n<arr.length;++n) {
         var elem = arr[n];

         var obj = this.map[elem.fElementId];

         if (!obj) {
            // element was not exists up to now
            var parent = null;
            if (elem.fMotherId !== 0) {
               parent = this.map[elem.fMotherId];
            } else {
               parent = this;
            }
            if (!parent) {
               console.error('Parent object ' + elem.fMotherId + ' does not exists - why?');
               return;
            }

            if (parent.childs === undefined)
               parent.childs = [];

            parent.childs.push(elem);

            obj = this.map[elem.fElementId] = elem;

         } else {
            // update existing element

            // just copy all properties from new object info to existing one
            JSROOT.extend(obj, elem);

            //obj.fMotherId = elem.fMotherId;

         }

         this.MarkModified(elem.fElementId);
      }
   }

   EveManager.prototype.SceneChanged = function(arr) {
      this.last_json = null;
      this.scene_changes = arr;
      console.log("JSON sceneChanged", arr[0]);

      if (arr[0].fTotalBinarySize) {
          this.last_json = arr;

      }
      else {
         this.PostProcessSceneChanges();
      }
   }


   EveManager.prototype.PostProcessSceneChanges = function() {
      var arr = this.scene_changes;
      var head = arr[0];
      var scene = this.GetElement(head.fSceneId);
      // console.log("PostProcessSceneChanges ", scene, arr);
      for (var i=0; i != scene.$receivers.length; i++)
      {
         var receiver = scene.$receivers[i].obj;

         for (var n=1; n<arr.length;++n) {
            var em = arr[n];
            console.log("PostProcessSceneChanges message ", em);
            var obj = this.map[em.fElementId];

            if (em.changeBit & this.EChangeBits.kCBVisibility) {
               obj.fRnrSelf = em.fRnrSelf;
               receiver.visibilityChanged(obj, em);
            }

            if (em.changeBit & this.EChangeBits.kCBColorSelection) {
               receiver.volorChanged(obj, em);
            }
            if (em.changeBit & this.EChangeBits.kCBObjProps) {
               receiver.renderDataChanged(obj, em);
            }
         }
      }

      this.scene_changes = null;
   }

   EveManager.prototype.DeleteChildsOf = function(elem) {
      if (!elem || !elem.childs) return;
      for (var n=0;n<elem.childs.length;++n) {
         var sub = elem.childs[n];
         this.DeleteChildsOf(sub);
         delete sub.childs;
         var id = sub.fElementId;
         if ((id !== undefined) && this.map[id])
            delete this.map[id];
      }
      delete elem.childs;
   }

   EveManager.prototype.DestroyElements = function(arr) {
      var ids = arr[0].element_ids;
      if (!ids) return;

      for (var n=0;n<ids.length;++n) {
         var element = this.map[ids[n]];
         if (!element) {
            console.log("try to delete non-existing element with id", ids[n]);
            continue;
         }
         this.DeleteChildsOf(element);
         element.$modified = true;
      }
   }

   EveManager.prototype.FindViewers = function(chlds) {
      if (chlds === undefined) chlds = this.childs;

      for (var k=0;k<chlds.length;++k) {
         if (!chlds[k].childs) continue;
         if (chlds[k]._typename == "ROOT::Experimental::TEveViewerList") return chlds[k].childs;
         var res = this.FindViewers(chlds[k].childs);
         if (res) return res;
      }
   }

   EveManager.prototype.UpdateBinary = function(rawdata, offset) {
      if (!this.last_json) return;

      if (!rawdata.byteLength) return;

      // console.log("GOT binary", rawdata.byteLength - offset);

      var arr = this.last_json;
      this.last_json = null;

      var lastoff = 0;

      for (var n=1; n<arr.length;++n)
      {
         var elem = arr[n];

         // console.log('elem', elem.fName, elem.rnr_offset);

         if (!elem.render_data) continue;

         var rd = elem.render_data;
         var off = offset + rd.rnr_offset;

         var obj = this.GetElement(elem.fElementId);

         // console.log('elem', elem.fName, off, rawdata.byteLength);

         if (off !== lastoff)
            console.error('Element', elem.fName, 'offset mismatch', off, lastoff);

         if (rd.vert_size) {
            rd.vtxBuff = new Float32Array(rawdata, off, rd.vert_size);
            off += rd.vert_size*4;
            // console.log('elems', elem.fName, elem.fVertexBuffer);
         }

         if (rd.norm_size) {
            rd.nrmBuff = new Float32Array(rawdata, off, rd.norm_size);
            off += rd.norm_size*4;
         }

         if (rd.index_size) {
            rd.idxBuff = new Uint32Array(rawdata, off, rd.index_size);
            off += rd.index_size*4;
         }

         lastoff = off;
      }

      if (lastoff !== rawdata.byteLength)
         console.error('Raw data decoding error - length mismatch', lastoff, rawdata.byteLength);

      if (arr[0].content == "ElementsRepresentaionChanges") { PostProcessSceneChanges();}
   }

   EveManager.prototype.CanEdit = function(elem) {
      // AMT this should be decided by the Summary controller
      if (elem._typename=="ROOT::Experimental::TEvePointSet") return true;
      if (elem._typename=="ROOT::Experimental::TEveJetCone") return true;
      if (elem._typename=="ROOT::Experimental::TEveTrack") return true;
      if (elem._typename=="ROOT::Experimental::TEveDataCollection") return true;
      if (elem._typename=="ROOT::Experimental::TEveDataItem") return true;
      return false;
   }

   EveManager.prototype.AnyVisible = function(arr) {
      if (!arr) return false;
      for (var k=0;k<arr.length;++k) {
         if (arr[k].fName) return true;
      }
      return false;
   }

   /** Create model, which can be used in TreeView */
   EveManager.prototype.CreateSummaryModel = function(tgt, src) {

      if (tgt === undefined) {
         tgt = [];
         src = this.childs;
         // console.log('original model', src);
      }

      for (var n=0;n<src.length;++n) {
         var elem = src[n];

         var newelem = { fName: elem.fName, id: elem.fElementId };

         if (this.CanEdit(elem))
            newelem.fType = "DetailAndActive";
         else
            newelem.fType = "Active";

         newelem.masterid = elem.fMasterId || elem.fElementId;

         tgt.push(newelem);
         if ((elem.childs !== undefined) && this.AnyVisible(elem.childs))
            newelem.childs = this.CreateSummaryModel([], elem.childs);
      }

      return tgt;
   }

   JSROOT.EVE.EveManager = EveManager;

   return JSROOT;

}));
