/// @file EveManager.js
/// used only together with OpenUI5

// TODO: add dependency from JSROOT components

sap.ui.define([], function() {

   "use strict";

   class EveManager {

      constructor()
      {
         this.map    = {}; // use object, not array
         this.childs = [];
         this.last_json = null;
         this.scene_changes = null;

         this.hrecv = []; // array of receivers of highlight messages

         this.scenes = [];  // list of scene objects

         // Set through Update Trigger UT_Refresh_Selection_State and name match
         // upon arrival of EveWorld. See also comments there.
         this.global_selection_id = null;
         this.global_highlight_id = null;

         this.EChangeBits = { "kCBColorSelection": 1, "kCBTransBBox": 2, "kCBObjProps": 4, "kCBVisibility": 8,  "kCBAdded": 16 };

         this.controllers = [];
         this.gl_controllers = [];
         this.selection_change_foos = [];

         this.initialized = false;
         this.busyProcessingChanges = false;
      }

      /** Returns element with given ID */
      GetElement(id)
      {
         return this.map[id];
      }

      globExceptionHandler(msg, url, lineNo, columnNo, error) {
         // NOTE: currently NOT connected, see onWebsocketOpened() below.

         console.error("EveManager got global error", msg, url, lineNo, columnNo, error);

         let suppress_alert = false;
         return suppress_alert;
      }

      /** Attach websocket handle to manager, all communication runs through manager */
      UseConnection(handle)
      {
         this.handle = handle;
         this.is_rcore = (handle.getUserArgs("GLViewer") == "RCore");

         handle.setReceiver(this);
         handle.connect();
      }

      onWebsocketClosed() {
         this.controllers.forEach(ctrl => {
            if (typeof ctrl.onDisconnect === "function")
                ctrl.onDisconnect();
         });
      }

      /** Checks if number of credits on the connection below threshold */
      CheckSendThreshold() {
         if (!this.handle) return false;
         let value = this.handle.getRelCanSend();
         let below = (value <= 0.2);
         if (this.credits_below_threshold === undefined)
            this.credits_below_threshold = false;
         if (this.credits_below_threshold === below)
            return below;

         this.credits_below_threshold = below;
         this.controllers.forEach(ctrl => {
            if (typeof ctrl.onSendThresholdChanged === "function")
                ctrl.onSendThresholdChanged(below, value);
         });
      }


      onWebsocketOpened() {
         // console.log("EveManager web socket opened.");

         // Presumably not needed at this point - known places where issues
         // can cause server-client protocol breach are handled.

         // window.onerror = this.globExceptionHandler.bind(this);
         // console.log("EveManager registered global error handler in window.onerror");
      }

      RegisterController(c)
      {
         this.controllers.push(c);
      }

      RegisterGlController(c)
      {
         this.gl_controllers.push(c);
      }

      RegisterSelectionChangeFoo(arg)
      {
         this.selection_change_foos.push(arg);
      }

      /** Called when data comes via the websocket */
      onWebsocketMsg(handle, msg, offset) {
         // if (this.ignore_all) return;

         try {
            if (typeof msg != "string") {
               // console.log('ArrayBuffer size ',
               // msg.byteLength, 'offset', offset);
               this.ImportSceneBinary(msg, offset);

               return;
            }

            if (EVE.gDebug)
               console.log("OnWebsocketMsg msg len=", msg.length, "txt:", (msg.length < 100) ? msg : (msg.substr(0, 500) + "..."));

            let resp = JSON.parse(msg);

            if (resp === undefined) {
               console.log("OnWebsocketMsg can't parse json: msg len=", msg.length, " txt:", msg.substr(0, 120), "...");
               return;
            }

            else if (resp[0] && resp[0].content == "REveScene::StreamElements") {
               this.ImportSceneJson(resp);
            }
            else if (resp.header && resp.header.content == "ElementsRepresentaionChanges") {
               this.ImportSceneChangeJson(resp);
            }
            else if (resp.content == "BeginChanges") {
               this.listScenesToRedraw = [];
               this.busyProcessingChanges = true;
            }
            else if (resp.content == "EndChanges") {
               this.ServerEndRedrawCallback();
               if (resp.log) {
                  resp.log.forEach((item) => {
                     // use console error above warning serverity
                     if (item.lvl < 3)
                        console.error(item.msg);
                     else
                        console.log(item.msg);
                  });
               }
            }
            else if (resp.content == "BrowseElement") {
               this.BrowseElement(resp.id);
            } else {
               console.error("OnWebsocketMsg Unhandled message type: msg len=", msg.length, " txt:", msg.substr(0, 120), "...");
            }
         }
         catch (e) {
            console.error("OnWebsocketMsg ", e);
         }
      }

      /** Sending Method Invocation Request
       * Special handling for offline case - some methods can be tried to handle without server */
      SendMIR(mir_call, element_id, element_class)
      {
         if (!mir_call || !this.handle || !element_class) return;

         // if (EVE.gDebug)
            console.log('MIR', mir_call, element_id, element_class);

         if (this.InterceptMIR(mir_call, element_id, element_class))
            return;

         // Sergey: NextEvent() here just to handle data recording in event_demo.C

         if ((this.handle.kind != "file") || (mir_call == "NextEvent()")) {

            let req = {
               "mir" : mir_call,
               "fElementId" : element_id,
               "class" : element_class
            }

            this.handle.send(JSON.stringify(req));
         }
      }


      /** Configure receiver for scene-respective events. Following event used:
       * onSceneChanged */
      RegisterSceneReceiver(id, receiver, func_name) {

         let elem = this.GetElement(id);

         if (!elem) return;

         if (!elem.$receivers) elem.$receivers = [];

         if (elem.$receivers.indexOf(receiver)<0)
            elem.$receivers.push(receiver);
      }

      /** Disconnect scene from the updates */
      UnRegisterSceneReceiver(id, receiver){
         let elem = this.GetElement(id);

         if (!elem) return;
         let idx = elem.$receivers.indexOf(receiver);
         if (idx > -1) { // only splice array when item is found
            console.log("unregister scene receiver");
            elem.$receivers.splice(idx, 1); // 2nd parameter means remove one item only
          }
      }

      /** Returns list of scene elements */
      getSceneElements()
      {
         if (!this.childs) return [];
         return this.childs[0].childs[2].childs;
      }

      /** Invoke function on all receiver of scene events - when such function exists */
      callSceneReceivers(scene, fname, arg) {
         if (scene.$receivers) {
             for (let i=0; i < scene.$receivers.length; i++) {
                 let receiver = scene.$receivers[i];
                 if (typeof receiver[fname] == "function")
                    receiver[fname](arg);
             }
         }
     }

      ImportSceneJson(arr)
      {
         this.last_json = null;

         // remember commands in manager
         if (arr[0].commands && !this.commands)
            this.commands = arr[0].commands;

         if (arr[0].fTotalBinarySize)
            this.last_json = arr;

         for (let n = 1; n < arr.length; ++n)
         {
            let elem = arr[n];

            let obj = this.map[elem.fElementId];

            if ( ! obj) // YYYY isn't it an error if obj exists?
            {
               // element did not exist up to now
               let parent = null;
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

               this.map[elem.fElementId] = elem;
            }

            this.ParseUpdateTriggersAndProcessPostStream(elem);

         }

         if (arr[0].fTotalBinarySize == 0) {
            this.sceneImportComplete(arr[0]);
         }
      }

      //______________________________________________________________________________

      removeElements(ids)
      {
         for (let i = 0; i < ids.length; ++i)
         {
            let elId = ids[i];
            let elem = this.GetElement(elId);
            if (!elem) {
               console.warning("EveManager.removeElements REveElement not found in map, id = ", elId);
               continue;
            }

            // remove from parent list of children
            let mother = this.GetElement(elem.fMotherId);
            if (mother && mother.childs) {
               let mc = mother.childs;
               for (let i = 0; i < mc.length; ++i) {

                  if (mc[i].fElementId === elId) {
                     mc.splice(i, 1);
                  }
               }
            }
            else
               console.warning("EveManager.removeElements can't remove child from mother, mother id = ", elem.fMotherId);

            delete this.map[elId];
         }
      }

      //______________________________________________________________________________

      ImportSceneChangeJson(msg)
      {
         this.last_json = null;
         this.scene_changes = msg;

         let scene = this.GetElement(msg.header.fSceneId);
         // console.log("ImportSceneChange", scene.fName, msg);

         // notify scenes for beginning of changes and
         // notify for element removal
         let removedIds = msg.header["removedElements"];
         if (removedIds.length) {
            this.callSceneReceivers(scene, "elementsRemoved", removedIds);
            this.removeElements(removedIds);
         }

         // wait for binary if needed
         if (msg.header.fTotalBinarySize)
         {
             this.last_json = [1];
             for (let i = 0; i < msg.arr.length; ++i)
                 this.last_json.push(msg.arr[i]);

         } else {
            this.CompleteSceneChanges();
         }
      }

      //______________________________________________________________________________


      CompleteSceneChanges()
      {
         if (!this.scene_changes) return;

         let arr = this.scene_changes.arr;
         let header = this.scene_changes.header;
         let scene = this.GetElement(header.fSceneId);
         let nModified = header["numRepresentationChanged"];

         // first import new elements in element map
         for (let n=0; n<arr.length; ++n)
         {
            let em = arr[n];
            if (em.changeBit & this.EChangeBits.kCBAdded) {
               this.map[em.fElementId] = em;
            }
         }

         for (let n=0; n<arr.length; ++n)
         {
            let em = arr[n];

            em.tag = "changeBit";
            if (em.changeBit & this.EChangeBits.kCBAdded) {
                // create new
               let parent = this.map[em.fMotherId];
               if (!parent.childs) // YYYY do we really need to create arrays here? Can it really be undef?
                  parent.childs = [];
               parent.childs.push(em);

               this.ParseUpdateTriggersAndProcessPostStream(em);
               this.callSceneReceivers(scene, "elementAdded", em);
               continue;
            }

            // merge new and old element
            let obj = this.map[em.fElementId];
            if(!obj) {
               console.log("ERRROR can't find element in map ", em); continue;
            }


            if (em.changeBit & this.EChangeBits.kCBVisibility)
            {
               if (obj.fRnrSelf != em.fRnrSelf) {
                  obj.fRnrSelf = em.fRnrSelf;
                  em.rnr_self_changed = true;
               }
               if (obj.fRnrChildren != em.fRnrChildren) {
                  obj.fRnrChildren = em.fRnrChildren;
                  em.rnr_children_changed = true;
               }
            }

            if (em.changeBit & this.EChangeBits.kCBObjProps) {
               delete obj.render_data;
               // AMT note ... the REveSelection changes fall here
               // I think this should be a separate change bit
               delete obj.sel_list;
               jQuery.extend(obj, em);
               this.ParseUpdateTriggersAndProcessPostStream(obj);
            }
            else if (em.changeBit & this.EChangeBits.kCBColorSelection) {
               delete em.render_data;
               Object.assign(obj, em);
            }

            this.callSceneReceivers(scene, "sceneElementChange", em);
         }

         this.listScenesToRedraw.push(scene);
         this.scene_changes = null;
      }

      //______________________________________________________________________________

      FindViewers(chlds) {
         if (chlds === undefined) chlds = this.childs;

         for (let k=0;k<chlds.length;++k) {
            if (!chlds[k].childs) continue;
            if (chlds[k]._typename == "ROOT::Experimental::REveViewerList") return chlds[k].childs;
            let res = this.FindViewers(chlds[k].childs);
            if (res) return res;
         }
      }

      ImportSceneBinary(rawdata, offset) {

         if (!this.last_json || !rawdata || !rawdata.byteLength) return;

         let arr = this.last_json;
         this.last_json = null;

         let lastoff = offset;

         // Start at 1, EveScene does not have render data.
         for (let n = 1; n < arr.length; ++n)
         {
            let elem = arr[n];

            if (!elem.render_data) continue;

            // in the scene change update check change bits are kCBElementAdded or kCBObjProps
           //  see REveScene::StreamRepresentationChanges binary stream
            if (this.scene_changes) {
               if (!(elem.changeBit & this.EChangeBits.kCBObjProps
                  || elem.changeBit & this.EChangeBits.kCBAdded))
                  continue;
            }

            let rd = elem.render_data,
                off = offset + rd.rnr_offset,
                obj = this.GetElement(elem.fElementId);

            if (off !== lastoff)
               console.error('Element ' + elem.fName + ' offset mismatch ' + off + ' != ' + lastoff);

            if (rd.trans_size) {
               rd.matrix = new Float32Array(rawdata, off, rd.trans_size);
               off += rd.trans_size*4;
            }
            if (rd.vert_size) {
               rd.vtxBuff = new Float32Array(rawdata, off, rd.vert_size);
               off += rd.vert_size*4;
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

         if (this.scene_changes)
            this.CompleteSceneChanges();
         else
            this.sceneImportComplete(arr[0]);
      }


      sceneImportComplete(msg)
      {
         // call controllers when all the last scene is imported
         let world     = this.map[1];
         let scenes    = world.childs[2];
         let lastChild = scenes.childs.length -1;

         if (scenes.childs[lastChild].fElementId == msg.fSceneId)
            this.controllers.forEach(ctrl => {
               if (ctrl.onEveManagerInit)
                  ctrl.onEveManagerInit();
            });
      }

      //------------------------------------------------------------------------------
      // XXXX UT = UpdateTrigger functions. XXXX Can /should we place them
      // somewhere else?
      // ------------------------------------------------------------------------------

      ParseUpdateTriggersAndProcessPostStream(el)
      {
         // console.log("EveManager.ParseUpdateTriggersAndProcessPostStream", el.UT_PostStream, this[el.UT_PostStream]);

         if (el.UT_PostStream !== undefined && typeof this[el.UT_PostStream] == "function")
         {
            this[el.UT_PostStream](el);
            delete el.UT_PostStream
         }
         // XXXX This is called before renderdata. Do we really need post stream or is PostScene
         //      sufficient? Think.

         // XXXX MT check also for PostScene and PostUpdate, put them somewhere and delete them.
      }

      UT_Selection_Refresh_State(sel)
      {
         // sel - rep of a REveSelection object.

         // console.log("UpdateTrigger UT_Selection_Refresh_State called for ", sel.fName);

         // XXXX Hack to assign global selection / highlight ids.
         // This would be more properly done by having REveWorld : public REveScene and store
         // ids there. These will also be needed for viewers, scenes, cameras etc.
         // I somewhat dislike setting this through name match (and class::WriteCoreJson() as
         // it defines the UT function so we sort of know what kind of objects this function
         // will be called for.
         // We shall see -- but it would better be soon as things are getting messy :)
         //
         if (sel._is_registered === undefined)
         {
            if (sel.fName == "Global Selection") this.global_selection_id = sel.fElementId;
            if (sel.fName == "Global Highlight") this.global_highlight_id = sel.fElementId;
            sel._is_registered = true;
            sel.prev_sel_list  = sel.sel_list;
            return;
         }

         let oldMap = new Map();
         sel.prev_sel_list.forEach(function(rec) {
            let iset = new Set(rec.sec_idcs);
            let x    = { "valid": true, "implied": rec.implied, "set": iset, "extra": rec.extra };
            oldMap.set(rec.primary, x);
         });

         // XXXXX MT - make sure to not process elements that do not have a representation in this manager.
         // Both for select and unselect.
         // I probably can't just throw them out, especially not for the selection primary as it is used
         // as a key.

         let newMap = new Map();
         sel.sel_list.forEach(function(rec) {
            let iset = new Set(rec.sec_idcs);
            let x    = { "valid": true, "implied": rec.implied, "set": iset, "extra": rec.extra };
            newMap.set(rec.primary, x);
         });

         // remove identicals from old and new map
         for (let id in oldMap)
         {
            if (id in newMap)
            {
               let oldSet = oldMap[id].set;
               let newSet = newMap[id].set;

               let nm = 0;
               for (let elem of oldSet)
               {
                  if (newSet.delete(elem)) {
                     nm++;
                  }
               }

               // invalidate if sets are empty or identical
               if (nm == oldSet.length && newSet.length == 0)
               {
                  oldMap[id].valid = false;
                  newMap[id].valid = false;
                  // console.log("EveManager.UT_Selection_Refresh_State identical sets for primary", id);
               }
            }
         }

         let changedSet = new Set();
         for (let [id, value] of oldMap.entries())
         {
            if (EVE.DebugSelection)
               console.log("UnSel prim", id, this.GetElement(id), this.GetElement(id).fSceneId);

            this.UnselectElement(sel, id);
            let iel = this.GetElement(id);
            changedSet.add(iel.fSceneId);

            for (let imp of value.implied)
            {
               let impEl = this.GetElement(imp);
               if (impEl) {
                  if (EVE.DebugSelection)
                     console.log("UnSel impl", imp, impEl, impEl.fSceneId);

                  this.UnselectElement(sel, imp);
                  changedSet.add(impEl.fSceneId);
               }
            }
         }

         for (let [id, value] of newMap.entries())
         {
            if (EVE.DebugSelection)
               console.log("Sel prim", id, this.GetElement(id), this.GetElement(id).fSceneId);

            let secIdcs = Array.from(value.set);
            let iel = this.GetElement(id);
            if ( ! iel) {
               console.log("EveManager.UT_Selection_Refresh_State this should not happen ", iel);
               continue;
            }
            changedSet.add(iel.fSceneId);
            this.SelectElement(sel, id, secIdcs, value.extra);

            for (let imp of value.implied)
            {
               if (EVE.DebugSelection)
                  console.log("Sel impl", imp, this.GetElement(imp), this.GetElement(imp).fSceneId);

               this.SelectElement(sel, imp, secIdcs, value.extra);
               changedSet.add(this.GetElement(imp).fSceneId);
            }
         }

         sel.prev_sel_list = sel.sel_list;
         sel.sel_list      = [];

         // XXXXX handle outline visible/hidden color changes
         // Should I check for change? Do I have old values?
         // Also ... check how this is done on init.
         // And think how to do it for multiple selections / connections
         // -----
         for (let foo of this.selection_change_foos)
         {
            // XXXX Need to be smarter, pass which selection has changed.
            // But also need some support for more of them and their handling.
            foo(this);
         }

         // redraw
         for (let item of changedSet)
         {
            let scene = this.GetElement(item);
            // this.callSceneReceivers(scene, "endChanges");
            this.listScenesToRedraw.push(scene);
         }

         // XXXX Oh, blimy, on first arrival, if selection is set, the selected
         // elements have not yet been received and so this will fail. Also true
         // for newly subscribed scenes, once we start supporting this.
         // So, we need something like reapply selections after new scenes arrive.
      }

      SelectElement(selection_obj, element_id, sec_idcs, extra)
      {
         let element = this.GetElement(element_id);
         if ( ! element) return;

         let scene = this.GetElement(element.fSceneId);
         if (scene.$receivers) {
            for (let r of scene.$receivers)
            {
               r.SelectElement(selection_obj, element_id, sec_idcs, extra);
            }
         }

         // console.log("EveManager.SelectElement", element, scene.$receivers[0].viewer.outline_pass.id2obj_map);
      }

      UnselectElement(selection_obj, element_id)
      {
         let element = this.GetElement(element_id);
         if ( ! element) return;

         let scene = this.GetElement(element.fSceneId);

         if (scene.$receivers) {
            for (let r of scene.$receivers)
            {
               r.UnselectElement(selection_obj, element_id);
            }
         }

         // console.log("EveManager.UnselectElement", element, scene.$receivers[0].viewer.outline_pass.id2obj_map);
      }

      ServerEndRedrawCallback()
      {
         // console.log("ServerEndRedrawCallback ", this.listScenesToRedraw);
         let recs = new Set();
         let viewers = new Set();
         for ( let i =0; i < this.listScenesToRedraw.length; i++) {
            let scene = this.listScenesToRedraw[i];
            if (scene.$receivers) {
               for (let r=0; r < scene.$receivers.length; r++) {
                  let sr = scene.$receivers[r];
                  recs.add(sr);
                  if (sr.glctrl) { viewers.add(sr.glctrl.viewer)};
               }
            }
         }

         if (this.is_rcore)
         {
            for (let v of viewers ) {
               v.timeStampAttributesAndTextures();
            }
         }

         for (let item of recs) {
            try {
               item.endChanges();
            } catch (e) {
               console.error("EveManager: Exception caught during update processing", e);
               // XXXX We might want to send e.name, e.message, e.stack back to the server.
            }
         }

         if (this.is_rcore)
         {
            for (let v of viewers ) {
               v.clearAttributesAndTextures();
            }
         }


         if (this.handle.kind != "file")
            this.handle.send("__REveDoneChanges");
         this.busyProcessingChanges = false;
      }

      /** Method invoked from server message to browse to element elid */
      BrowseElement(elid) {
         let scenes = this.getSceneElements();

         for (let i = 0; i < scenes.length; ++i) {
            let scene = this.GetElement(scenes[i].fElementId);
            this.callSceneReceivers(scene, "BrowseElement", elid);
            break; // normally default scene is enough
         }
      }

      /** Returns true if element match to some entry in selection */
      MatchSelection(globalid, eve_el, indx) {
         let so = this.GetElement(globalid);

         let a  = so ? so.prev_sel_list : null;
         if (a && (a.length == 1))
         {
            let h = a[0];
            if (h.primary == eve_el.fElementId || h.primary == eve_el.fMasterId) {
               if (indx) {
                  if (h.sec_idcs && h.sec_idcs[0] == indx) {
                     return true;
                  }
               }
               if ( ! indx && ! h.sec_idcs.length) {
                  return true;
               }
            }
         }
      }


      //==============================================================================
      // Offline handling
      //==============================================================================


      /** find elements ids where fMasterId equal to provided */
      FindElemetsForMaster(elementId, collect_ids) {
         let res = [];

         for (let elid in this.map) {
            let el = this.map[elid];
            if ((el.fMasterId === elementId) && (el.fElementId !== elementId))
               res.push(collect_ids ? el.fElementId : el);
         }

         return res;
      }

      /** used to intercept NewElementPickedStr for hightlight and selection @private */
      _intercept_NewElementPickedStr(elementId) {

         let mirElem = this.GetElement(this._intercept_id);

         let msg1 = { content: "BeginChanges" }, msg3 = { content: "EndChanges" },
             msg2 = { arr: [ Object.assign({UT_PostStream:"UT_Selection_Refresh_State", changeBit: 4}, mirElem) ],
                      header:{ content:"ElementsRepresentaionChanges", fSceneId: mirElem.fSceneId, fTotalBinarySize:0, numRepresentationChanged:1, removedElements:[] }};

         msg2.arr[0].sel_list = elementId ? [{primary: elementId, implied: this.FindElemetsForMaster(elementId, true), sec_idcs:[]}] : [];

         msg2.arr[0].prev_sel_list = undefined;

         this.handle.inject([msg1, msg2, msg3]);
      }

      /** used to intercept BrowseElement call @private */
      _intercept_BrowseElement(elementId) {
         let msg1 = { content: "BrowseElement", id: elementId },
             msg2 = { content: "BeginChanges" },
             msg3 = { content: "EndChanges" };

         this.handle.inject([msg1, msg2, msg3]);
      }

      /** @summary used to intercept SetRnrSelf call
        * @private */
      _intercept_SetRnrSelf(flag) {
         let messages = [{ content: "BeginChanges" }];

         let mirElem = this.GetElement(this._intercept_id);
         let msg = { arr: [{ changeBit:8, fElementId: mirElem.fElementId, fRnrChildren: mirElem.fRnrChildren, fRnrSelf: flag }],
                     header:{ content: "ElementsRepresentaionChanges", fSceneId: mirElem.fSceneId, fTotalBinarySize:0, numRepresentationChanged:1, removedElements:[]}};

         messages.push(msg);

         this.FindElemetsForMaster(this._intercept_id).forEach(function(subElem) {
            msg = { arr: [{ changeBit:8, fElementId: subElem.fElementId, fRnrChildren: subElem.fRnrChildren, fRnrSelf: flag }],
                    header:{ content: "ElementsRepresentaionChanges", fSceneId: subElem.fSceneId, fTotalBinarySize:0, numRepresentationChanged:1, removedElements:[]}};
            messages.push(msg);
         });
         messages.push({ content: "EndChanges" });

         this.handle.inject(messages);
      }

      /** @summary used to intercept SetMainColorRGB
        * @private */
      _intercept_SetMainColorRGB(colr, colg, colb) {
         let messages = [{ content: "BeginChanges" }];

         let newColor = EVE.JSR.addColor("rgb(" + colr + "," + colg + "," + colb + ")");

         let mirElem = this.GetElement(this._intercept_id);
         let msg = { arr: [ Object.assign({changeBit:1}, mirElem) ],
                     header:{ content: "ElementsRepresentaionChanges", fSceneId: mirElem.fSceneId, fTotalBinarySize:0, numRepresentationChanged:1, removedElements:[]}};

         msg.arr[0].fMainColor = newColor;
         msg.arr[0].sel_list = msg.arr[0].prev_sel_list = msg.arr[0].render_data = undefined;

         messages.push(msg);

         this.FindElemetsForMaster(this._intercept_id).forEach(function(subElem) {
            let msg = { arr: [ Object.assign({changeBit:1}, subElem) ],
                  header: { content: "ElementsRepresentaionChanges", fSceneId: subElem.fSceneId, fTotalBinarySize:0, numRepresentationChanged:1, removedElements:[]}};
            msg.arr[0].fMainColor = newColor;
            msg.arr[0].sel_list = msg.arr[0].prev_sel_list = msg.arr[0].render_data = undefined;
            messages.push(msg);
         });
         messages.push({ content: "EndChanges" });

         this.handle.inject(messages);
      }

      /** Handling of MIR calls without sending data to the server.
       * Can be used for debugging or as offline app */
      InterceptMIR(mir_call, element_id, element_class) {

         if (this.handle.kind != "file")
            return false;

         // just do not intercept
         let do_intercept = false;

         if (((mir_call.indexOf("NewElementPickedStr(") == 0) && ((element_id == this.global_highlight_id) || (element_id == this.global_selection_id))) ||
             ((mir_call.indexOf("BrowseElement(") == 0) && (element_id == 0)) ||
             (mir_call.indexOf("SetRnrSelf(") == 0) || (mir_call.indexOf("SetMainColorRGB(") == 0))
            do_intercept = true;

         if (!do_intercept)
            return false;

         this._intercept_id = element_id;
         this._intercept_class = element_class;

         globalThis.$eve7mir = this;

         if (mir_call.indexOf("SetMainColorRGB(") == 0)
            mir_call = mir_call.replace(/\(UChar_t\)/g, '');

         let func = new Function('globalThis.$eve7mir._intercept_' + mir_call);

         try {
            func();
         } catch {
            console.log("Fail to intercept MIR call:", mir_call);
         }

         delete globalThis.$eve7mir;

         return true;
      }

   } // class EveManager


   EVE.EveManager = EveManager;

   EVE.DebugSelection = 0;

   // EVE.gDebug = true;

   return EveManager;

});
