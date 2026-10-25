/// @file EveScene.js

// TODO: add dependency from JSROOT components

sap.ui.define(['rootui5/eve7/lib/EveManager'], function(EveManager) {

   "use strict";

   const kIdentity = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1];

   /// constructor, handle for REveScene class

   class EveScene {

      constructor(mgr, sceneInfo, glctrl)
      {
         this.mgr     = mgr;
         this.id      = sceneInfo.fSceneId;
         this.glctrl  = glctrl;
         this.creator = glctrl.viewer.creator;
         this.id2obj_map  = new Map; // element id -> the object that draws the element
         this.id2node_map = new Map; // element id -> the element's node, see makeNode()

         // Raised when an element is added, removed or rebuilt; the viewer then
         // recomputes this scene's box.
         this.content_stamp = 0;
         // Raised when an element's matrix changes, from the server or by Motion;
         // near and far follow the scene's current box.
         this.trans_stamp = 0;

         this.IsOverlay = mgr.GetElement(sceneInfo.fSceneId).IsOverlay;
         this.first_time = true;

         // register ourself for scene events
         this.mgr.RegisterSceneReceiver(this.id, this);

         if(this.mgr.is_rcore) {
            this.SelectElement = this.SelectElementRCore;
            this.UnselectElement = this.UnselectElementRCore;
         } else {
            this.SelectElement = this.SelectElementStd;
            this.UnselectElement = this.UnselectElementStd;
         }
      }

      //==============================================================================
      // Render object creation / management
      //==============================================================================

      makeGLRepresentation(elem) {
         if (!elem.render_data) return null;

         try {
            let fname = elem.render_data.rnr_func;
            let obj3d = this.creator[fname](elem, elem.render_data);

            if (obj3d) {
               // Used by JSRoot
               obj3d._typename = this.creator.GenerateTypeName(obj3d);

               obj3d.eve_el = elem; // reference to the EveElement
               obj3d.scene = this;  // required for change processing, esp. highlight/selection

               // SL: this is just identifier for highlight, required to show items on other places, set in creator
               obj3d.geo_object = elem.fMasterId || elem.fElementId;
               obj3d.geo_name = elem.fName; // used for highlight

               return obj3d;
            }
         }
         catch (e) {
            console.error("makeGLRepresentation", e);
         }
      }

      getObj3D(elementId)
      {
         return this.id2obj_map.get(elementId);
      }

      //------------------------------------------------------------------------
      // The element tree
      //
      // Each element has a node, in id2node_map, that carries its matrix. A
      // leaf that draws is its own node. Any other element has a group node:
      // its `self` is the element's drawing, at identity, and its other
      // children are the nodes of the element's children. fRnrSelf is the
      // drawing's `visible`, fRnrChildren the group's `visibleChildren`.
      //
      // Motion moves nodes, and a scene's box is the server boxes moved by the
      // world matrices of their nodes.
      //
      // three.js has no `self` and `visibleChildren`. There the node helpers
      // keep the children's nodes in a `kids` object and use its `visible`.
      //------------------------------------------------------------------------

      /** A new group node for `elem`, with its matrix. It has no eve_el, so
        * picking passes it by. */
      makeNode(elem)
      {
         let node = this.glctrl.viewer.make_object("el " + elem.fElementId);
         if (!this.mgr.is_rcore) node.eve_group = true;
         this.setNodeMatrix(node, elem);
         this.nodeSetChildrenVisible(node, elem.fRnrChildren);
         return node;
      }

      /** Add `child` to `node`, a group node or the scene container. */
      nodeAdd(node, child)
      {
         if (this.mgr.is_rcore || !node.eve_group) { node.add(child); return; }
         if (!node.eve_kids) {
            node.eve_kids = this.glctrl.viewer.make_object("kids");
            node.eve_kids.eve_owner = node;
            node.eve_kids.visible = node.eve_vis_children !== false;
            node.add(node.eve_kids);
         }
         node.eve_kids.add(child);
      }

      /** The node `child` belongs to; on three.js the group, not its `kids`. */
      nodeParent(child)
      {
         let p = child.parent;
         return (p && p.eve_owner) || p;
      }

      /** Make `obj` the drawing of group node `node`, at identity, in place of
        * the previous one; null removes it. */
      nodeSetSelf(node, obj)
      {
         if (this.mgr.is_rcore) { node.self = obj; return; }
         if (node.eve_self) node.remove(node.eve_self);
         node.eve_self = obj;
         if (!obj) return;
         node.add(obj);
         obj.matrix.identity();
         obj.matrixAutoUpdate = false;
         obj.updateMatrixWorld(true);
      }

      /** Show or hide the children of group node `node`; not its drawing. */
      nodeSetChildrenVisible(node, vis)
      {
         if (this.mgr.is_rcore) { node.visibleChildren = vis; return; }
         node.eve_vis_children = vis;
         if (node.eve_kids) node.eve_kids.visible = vis;
      }

      /** Replace `obj`, the own node of `elem`, by a group node with `obj` as
        * its drawing. The group takes obj's current matrix, motion included. */
      wrapInGroup(obj, elem)
      {
         let node = this.glctrl.viewer.make_object("el " + elem.fElementId);
         if (this.mgr.is_rcore) {
            node.setMatrixFromArray(obj.matrix.elements);
         } else {
            node.eve_group = true;
            node.matrix.copy(obj.matrix);
            node.matrixAutoUpdate = false;
         }
         this.nodeSetChildrenVisible(node, elem.fRnrChildren);
         let parent = obj.parent;
         if (parent) { parent.remove(obj); parent.add(node); }
         this.nodeSetSelf(node, obj);
         if (!this.mgr.is_rcore) node.updateMatrixWorld(true);
         return node;
      }

      /** Make the node of `elem` under `parentNode` and register it, see
        * "The element tree" above. */
      addElementNode(elem, parentNode)
      {
         let node = null;
         if (!(elem.childs && elem.childs.length)) {
            node = this.makeContent(elem);
            if (node) {
               this.setNodeMatrix(node, elem);
               node.visible = elem.fRnrSelf;
               this.id2obj_map.set(elem.fElementId, node);
            }
         }
         if (!node) {
            node = this.makeNode(elem);
            this.addContent(elem, node);
         }
         this.nodeAdd(parentNode, node);
         this.id2node_map.set(elem.fElementId, node);
         this.applyMotion(elem, node);
         return node;
      }

      /** Register the trajectory of `elem` on `node`, or drop it when `elem`
        * does not move. The node must hold the matrix the server sent, which
        * is the start of the trajectory. */
      applyMotion(elem, node)
      {
         let mo = this.glctrl.viewer.motion;
         if (!mo) return;
         if (elem.motion)
            mo.update(elem.fElementId, node, elem.motion, this);
         else
            mo.remove(elem.fElementId);
      }

      /** Give `elem`, a leaf, a group node, so that it can hold children. */
      promoteToGroup(elem)
      {
         let node = this.wrapInGroup(this.id2node_map.get(elem.fElementId), elem);
         this.id2node_map.set(elem.fElementId, node);

         // The group carries the matrix now, so the trajectory moves it.
         let mo = this.glctrl.viewer.motion;
         if (mo) mo.retarget(elem.fElementId, node);
         return node;
      }

      /** Set the matrix of `elem`'s full state on `node`; identity when it has
        * none. Moves and motion set the node themselves. */
      setNodeMatrix(node, elem)
      {
         this.applyMatrix(node, EveManager.decodeMatrix(elem.matrix) || kIdentity);
      }

      /** Set the 16 values of `m`, column-major, as the matrix of `node`. */
      applyMatrix(node, m)
      {
         if (this.mgr.is_rcore) {
            node.setMatrixFromArray(m);
         } else {
            node.matrix.fromArray(m);
            node.matrixAutoUpdate = false;
            node.updateMatrixWorld(true);
         }
      }

      /** Make the object that draws `elem`, or null if it does not draw. */
      makeContent(elem)
      {
         if (!elem.render_data) return null;
         if (!this.creator[elem.render_data.rnr_func]) {
            console.error("Function " + elem.render_data.rnr_func + " missing in creator");
            return null;
         }
         return this.makeGLRepresentation(elem) || null;
      }

      /** Make the drawing of `elem` and set it as the drawing of group node
        * `node`. Returns it, or null when `elem` does not draw. */
      addContent(elem, node)
      {
         let obj3d = this.makeContent(elem);
         if (!obj3d) return null;
         obj3d.visible = elem.fRnrSelf;
         this.nodeSetSelf(node, obj3d);
         this.id2obj_map.set(elem.fElementId, obj3d);
         return obj3d;
      }

      /** The node to put the node of `elem` under: the mother's, made a group
        * if it is a leaf. Until the mother's node exists, the scene container;
        * the mother adopts it when it is added. */
      parentNodeOf(elem)
      {
         if (elem.fMotherId !== this.id) {
            let pn = this.id2node_map.get(elem.fMotherId);
            if (pn && pn === this.id2obj_map.get(elem.fMotherId))
               pn = this.promoteToGroup(this.mgr.GetElement(elem.fMotherId));
            if (pn) return pn;
         }
         return this.glctrl.getSceneContainer(this);
      }

      /** Build the nodes and drawing objects of `elem` and everything below it. */
      buildSubtree(elem, parentNode)
      {
         let node = this.addElementNode(elem, parentNode);
         if (elem.childs)
            for (let c of elem.childs)
               this.buildSubtree(c, node);
      }

      //==============================================================================

      /** Build the scene's element tree under the scene container. */
      redrawScene()
      {
         let eveScene = this.mgr.GetElement(this.id);
         let childs = (eveScene && eveScene.childs) || [];
         if (!childs.length && this.first_time) return;

         let cont = this.glctrl.getSceneContainer(this);
         while (cont.children.length > 0)
            cont.remove(cont.children[0]);
         this.id2obj_map.clear();
         this.id2node_map.clear();

         for (let c of childs)
            this.buildSubtree(c, cont);
         ++this.content_stamp;

         this.applySelectionOnSceneCreate(this.mgr.global_selection_id);
         this.applySelectionOnSceneCreate(this.mgr.global_highlight_id);

         this.first_time = false;
      }

      removeScene()
      {
         let cont = this.glctrl.getSceneContainer(this);
         while (cont.children.length > 0)
            cont.remove(cont.children[0]);

         this.mgr.UnRegisterSceneReceiver(this.id, this);
         this.first_time = true;
      }

      /** Apply fRnrSelf to the element's drawing and fRnrChildren to its group
        * node. */
      applyVisibility(el)
      {
         let obj3d = this.getObj3D(el.fElementId);
         if (obj3d) obj3d.visible = el.fRnrSelf;
         let node = this.id2node_map.get(el.fElementId);
         if (node && node !== obj3d) this.nodeSetChildrenVisible(node, el.fRnrChildren);
      }

      onSceneCreate(/* id */)
      {
         this.redrawScene();
      }

      //==============================================================================
      // Scene changes processing
      //==============================================================================

      beginChanges()
      {
      }

      endChanges()
      {
         // Recompute the scene box after content changes, not after moves: the
         // 3D axis is sized from it and would follow moving objects.
         // REveViewer::SetAxesBBox() fixes the axis extent instead.
         this.glctrl.viewer.request_render(this.need_bbox_update);
         this.need_bbox_update = false;
      }

      elementAdded(el)
      {
         this.need_bbox_update = true;
         ++this.content_stamp;
         let node = this.addElementNode(el, this.parentNodeOf(el));

         // Adopt the nodes of children that were added before this element.
         if (el.childs)
            for (let c of el.childs) {
               let cn = this.id2node_map.get(c.fElementId);
               if (cn && this.nodeParent(cn) !== node) {
                  if (cn.parent) cn.parent.remove(cn);
                  this.nodeAdd(node, cn);
               }
            }
      }

      /** Rebuild the drawing of `el`. With `keepMatrix`, for a rebuild without
        * new state, the node keeps its matrix. */
      replaceElement(el, keepMatrix) {
         // The trajectory stays with the node; a new leaf node registers it
         // again, see addElementNode().
         ++this.content_stamp;

         let node = this.id2node_map.get(el.fElementId);
         if (!node) { this.elementAdded(el); return; }

         try {
            let obj3d = this.getObj3D(el.fElementId);
            this.id2obj_map.delete(el.fElementId);
            if (obj3d && obj3d === node) {
               // The element is its own node: the new drawing takes its place.
               let m = keepMatrix ? Array.from(node.matrix.elements) : null;
               let parent = node.parent;
               if (parent) parent.remove(node);
               this.id2node_map.delete(el.fElementId);
               let n = this.addElementNode(el, parent || this.glctrl.getSceneContainer(this));
               if (m) this.applyMatrix(n, m);
            } else {
               if (!keepMatrix) this.setNodeMatrix(node, el);
               if (!this.addContent(el, node)) this.nodeSetSelf(node, null);
            }
         }
         catch (e) {
            console.error("replace element", e);
         }
      }

      /** Apply one element of a "Motion" message: the same update as a
        * kCBTrans change, outside the BeginChanges/EndChanges cycle. */
      sceneElementMotion(msg)
      {
         let el = this.mgr.GetElement(msg.fElementId);
         if (!el) return;

         this.updateElementTrans(el, msg);

         // No endChanges() follows, so request the render here. Going through
         // Motion applies the viewer's render cap.
         let mo = this.glctrl.viewer.motion;
         if (mo)
            mo.requestRender();
         else
            this.glctrl.viewer.request_render();
      }

      /** Remove the drawing of `elem`, which no longer draws. A leaf gets an
        * empty group node in its place. */
      dropContent(elem)
      {
         let obj3d = this.getObj3D(elem.fElementId);
         if (!obj3d) return;
         this.id2obj_map.delete(elem.fElementId);
         this.need_bbox_update = true;
         ++this.content_stamp;

         let node = this.id2node_map.get(elem.fElementId);
         if (obj3d === node) {
            node = this.wrapInGroup(obj3d, elem);
            this.id2node_map.set(elem.fElementId, node);
            let mo = this.glctrl.viewer.motion;
            if (mo) mo.retarget(elem.fElementId, node);
         }
         this.nodeSetSelf(node, null);
      }

      /** Apply a transformation-only change to the existing renderer object by
        * setting msg.matrix on it. Requests no render; the callers do. */
      updateElementTrans(el, msg)
      {
         let node = this.id2node_map.get(msg.fElementId);
         if (!node) return;

         let mo = this.glctrl.viewer.motion;

         // Every transformation update carries the matrix; none means identity.
         this.applyMatrix(node, EveManager.decodeMatrix(msg.matrix) || kIdentity);
         ++this.trans_stamp;

         // After the matrix is applied: Motion takes the start position from it.
         // Every transformation update carries "motion"; null clears the trajectory.
         if (mo && msg.motion !== undefined)
            mo.update(msg.fElementId, node, msg.motion, this);
      }

      elementsRemoved(ids)
      {
         this.need_bbox_update = true;
         ++this.content_stamp;

         let mo = this.glctrl.viewer.motion;

         for (let i = 0; i < ids.length; i++)
         {
            let elId  = ids[i];

            // Drop the trajectory. Nothing else removes it, and a live
            // trajectory keeps Motion's render loop running.
            if (mo) mo.remove(elId);

            // Removing the node removes the subtree below it from the scene.
            let node = this.id2node_map.get(elId);
            if (node && node.parent) node.parent.remove(node);
            this.id2node_map.delete(elId);

            let obj3d = this.getObj3D(elId);
            if (!obj3d) continue;
            this.id2obj_map.delete(elId);

            if (!this.mgr.is_rcore) {
               if (typeof obj3d.dispose !== 'function')
                  console.log("EveScene.elementsRemoved no dispose function for " + this.mgr.GetElement(elId)._typename, ", rnr obj ", obj3d._typename, obj3d);
               else
                  obj3d.dispose();
            }
         }
      }

      sceneElementChange(msg)
      {
         let el = this.mgr.GetElement(msg.fElementId);

         // visibility
         if (msg.changeBit & this.mgr.EChangeBits.kCBVisibility) {
            this.applyVisibility(el);
         }

         // Update the transformation in place.
         if (msg.changeBit & this.mgr.EChangeBits.kCBTrans) {
            this.updateElementTrans(el, msg);
         }

         // A full update without render data: drop the drawing, set the matrix.
         if (!el.render_data && (msg.changeBit & this.mgr.EChangeBits.kCBObjProps)) {
            this.dropContent(el);
            let node = this.id2node_map.get(el.fElementId);
            if (node) {
               this.setNodeMatrix(node, el);
               ++this.trans_stamp;
            }
         }

         // A full update rebuilds the drawing. A core JSON change calls
         // refresh(el) of a drawing that has one, else rebuilds it from the
         // cached render data and keeps the node's matrix.
         if (el.render_data) {
            let rebuild = !!(msg.changeBit & this.mgr.EChangeBits.kCBObjProps);
            if (!rebuild && (msg.changeBit & this.mgr.EChangeBits.kCBCoreJson)) {
               let obj3d = this.getObj3D(el.fElementId);
               if (obj3d && typeof obj3d.refresh === 'function')
                  obj3d.refresh(el);
               else
                  rebuild = true;
            }
            if (rebuild) {
               this.need_bbox_update = true;
               this.replaceElement(el, !(msg.changeBit & this.mgr.EChangeBits.kCBObjProps));
            }
         }

         // A full update carries the whole transformation, motion included,
         // and has just set the node to the matrix the server sent.
         if (msg.changeBit & this.mgr.EChangeBits.kCBObjProps) {
            let node = this.id2node_map.get(el.fElementId);
            if (node) this.applyMotion(el, node);
         }
      }

      //==============================================================================
      // Selection handling
      //==============================================================================

      sanitizeIndex(index)
      {
         if (Array.isArray(index))    return index.length > 0 ? index : undefined;
         if (Number.isInteger(index)) return [ index ];
         return undefined;
      }

      sendSelectMIR(sel_id, eve_el, is_multi, index)
      {
         index = this.sanitizeIndex(index);
         let is_secsel = index !== undefined;

         let fcall = "NewElementPickedStr(" + (eve_el ? eve_el.fElementId : 0) + `, ${is_multi}, ${is_secsel}`;
         if (is_secsel)
         {
            fcall += ", \"" + index.join(",") + "\"";
         }
         fcall += ")";

         this.mgr.SendMIR(fcall, sel_id, "ROOT::Experimental::REveSelection");
      }

      /** interactive handler. Calculates selection state, apply to element and distribute to other scene */
      processElementSelected(eve_el, index, event)
      {
         // console.log("EveScene.processElementSelected", obj3d, col, index, evnt);

         let is_multi  = event && event.ctrlKey ? true : false;
         this.sendSelectMIR(this.mgr.global_selection_id, eve_el, is_multi, index);

         return true;
      }

      /** interactive handler */
      processElementHighlighted(eve_el, index, event)
      {
         // RenderCore viewer is organizing selection on stack, the last selection will set the color
         if (!this.mgr.is_rcore && this.mgr.MatchSelection(this.mgr.global_selection_id, eve_el, index))
            return true;

         // Need check for duplicates before call server, else server will un-higlight highlighted element
         if (this.mgr.MatchSelection(this.mgr.global_highlight_id, eve_el, index))
            return true;

         // when send queue below threshold, ignore highlight
         if (this.mgr.CheckSendThreshold())
            return true;

         this.sendSelectMIR(this.mgr.global_highlight_id, eve_el, false, index);

         return true;
      }

      clearHighlight()
      {
         // QQQQ This will have to change for multi client support.
         // Highlight will always be multi and we will have to track
         // which highlight is due to our connection.

         // when send queue below threshold, ignre highlight
         if (this.mgr.CheckSendThreshold())
            return true;

         let so = this.mgr.GetElement(this.mgr.global_highlight_id);

         if (so && so.prev_sel_list && so.prev_sel_list.length)
         {
            this.sendSelectMIR(this.mgr.global_highlight_id, 0, false);
         }

         return true;
      }

      applySelectionOnSceneCreate(selection_id)
      {
         let selection_obj = this.mgr.GetElement(selection_id);
         if ( ! selection_obj || ! selection_obj.prev_sel_list) return;

         let pthis = this;
         selection_obj.prev_sel_list.forEach(function(rec) {

            let prl = pthis.mgr.GetElement(rec.primary);
            if (prl && prl.fSceneId == pthis.id)
            {
               pthis.SelectElement(selection_obj, rec.primary, rec.sec_idcs, rec.extra );
            }
            else // XXXXX why else ... should we not process all of them?!!!!
            {
               for (let impId of rec.implied)
               {
                  let eli = pthis.mgr.GetElement(impId);
                  if (eli && eli.fSceneId == pthis.id)
                  {
                     // console.log("CHECK select IMPLIED", pthis);
                     pthis.SelectElement(selection_obj, impId, rec.sec_idcs, rec.extra);
                  }
               }
            }
         });
      }

      SelectElementStd(selection_obj, element_id, sec_idcs, extra)
      {
         let obj3d = this.getObj3D( element_id );
         if (!obj3d) return;

         let outline_map = this.glctrl.viewer.outline_map;
         // console.log("EveScene.SelectElement ", selection_obj.fName, element_id, selection_obj.fElementId, outline_map);

         outline_map[element_id] = outline_map[element_id] || [];

         if (outline_map[element_id][selection_obj.fElementId] !== undefined) {
            return;
         }

         const ST_Selection = 0; // Matching THREE.OutlinePassEve.selection_enum
         const ST_Highlight = 1;
         let stype  = selection_obj.fName.endsWith("Selection") ? ST_Selection : ST_Highlight;
         let oe = this.mgr.GetElement(element_id);

         let res = {
            "sel_type" : stype,
            "sec_sel"  : (oe.fSecondarySelect && sec_idcs.length > 0) ? true : false,
            "geom"     : []
         };

         if (!res.sec_sel) outline_map[element_id] = [];

         if (obj3d.get_ctrl) {
            let ctrl = obj3d.get_ctrl(obj3d);
            ctrl.DrawForSelection(sec_idcs, res, extra);
         } else {
            res.geom.push(obj3d);
         }
         outline_map[element_id][selection_obj.fElementId] = res;

         if (stype == ST_Highlight && selection_obj.sel_list) {
            this.glctrl.viewer.remoteToolTip(selection_obj.sel_list[0].tooltip);
         }
      }

      UnselectElementStd(selection_obj, element_id)
      {
         let outline_map = this.glctrl.viewer.outline_map;
         // console.log("EveScene.UnselectElement ", selection_obj.fName, element_id, selection_obj.fElementId, outline_map);

         if (outline_map[element_id] !== undefined) {
            delete outline_map[element_id][selection_obj.fElementId];
         }
      }

      SelectElementRCore(selection_obj, element_id, sec_idcs, extra)
      {
         let obj3d = this.getObj3D( element_id );
         if (!obj3d) return;

         let eve_el = this.mgr.GetElement(element_id);

         let sid = selection_obj.fElementId;
         let smap = this.glctrl.viewer.selection_map;
         if (smap[sid] === undefined) {
            smap[sid] = {};
         }

         let res = {
            "sec_sel"  : (eve_el.fSecondarySelect && sec_idcs.length > 0) ? true : false,
            "geom"     : []
         };

         if (obj3d.get_ctrl) {
            let ctrl = obj3d.get_ctrl(obj3d);
            ctrl.DrawForSelection(sec_idcs, res, extra);
         } else {
            res.geom.push(obj3d);
         }
         smap[sid][element_id] = res;
         this.glctrl.viewer.make_selection_last_in_list(sid);

         // Display tooltip.
         // XXXX Should check if highlight request came from this viewer.
         if (selection_obj.fIsHighlight && selection_obj.sel_list) {
            this.glctrl.viewer.remoteToolTip(selection_obj.sel_list[0].tooltip);
         }
      }

      UnselectElementRCore(selection_obj, element_id)
      {
         let sid = selection_obj.fElementId;
         let smap = this.glctrl.viewer.selection_map;
         if (smap[sid] !== undefined) {
            delete smap[sid][element_id];
            if (Object.keys(smap[sid]).length == 0) {
               this.glctrl.viewer.remove_selection_from_list(sid);
            }
         }
      }

   } // class EveScene

   return EveScene;
});
