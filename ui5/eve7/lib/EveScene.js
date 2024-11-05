/// @file EveScene.js

// TODO: add dependency from JSROOT components

sap.ui.define(['rootui5/eve7/lib/EveManager'], function(EveManager) {

   "use strict";

   /// constructor, handle for REveScene class

   class EveScene {

      constructor(mgr, sceneInfo, glctrl)
      {
         this.mgr     = mgr;
         this.id      = sceneInfo.fSceneId;
         this.glctrl  = glctrl;
         this.creator = glctrl.viewer.creator;
         this.id2obj_map  = new Map; // base on element id

         this.IsOverlay = mgr.GetElement(sceneInfo.fSceneId).IsOverlay;
         this.first_time = true;
         this.need_visibility_update = false;

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

               obj3d.matrixAutoUpdate = false;
               if (elem.render_data.matrix) {
                  if (this.mgr.is_rcore) {
                     obj3d.setMatrixFromArray(elem.render_data.matrix);
                  } else {
                     obj3d.matrix.fromArray(elem.render_data.matrix);
                     obj3d.updateMatrixWorld(true);
                  }
               }

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

      create3DObjects(all_ancestor_children_visible, prnt, res3d)
      {
         if (prnt === undefined) {
            prnt = this.mgr.GetElement(this.id);
            res3d = [];
         }

         if (!prnt || !prnt.childs) return res3d;

         for (let k = 0; k < prnt.childs.length; ++k)
         {
            let elem = prnt.childs[k];
            if (elem.render_data)
            {
               let fname = elem.render_data.rnr_func;
               if ( ! this.creator[fname])
               {
                  console.error("Function " + fname + " missing in creator");
               }
               else
               {
                  let obj3d = this.makeGLRepresentation(elem);
                  if (obj3d)
                  {
                     // MT - should maintain hierarchy ????
                     // Easier to remove ... but might need sub-class of
                     // Object3D to separate "graphical" children and structural children.

                     res3d.push(obj3d);

                     this.id2obj_map.set(elem.fElementId, obj3d);

                     obj3d.visible = elem.fRnrSelf && all_ancestor_children_visible;
                     obj3d.all_ancestor_children_visible = all_ancestor_children_visible;
                  }
               }
            }

            this.create3DObjects(elem.fRnrChildren && all_ancestor_children_visible, elem, res3d);
         }

         return res3d;
      }

      //==============================================================================

      /** method insert all objects into three.js container */
      redrawScene()
      {
         let eveScene = this.mgr.GetElement(this.id);
         if (!this.glctrl) return;

         let res3d = this.create3DObjects(true);
         if ( ! res3d.length && this.first_time) return;

         let cont = this.glctrl.getSceneContainer(this);
         while (cont.children.length > 0)
            cont.remove(cont.children[0]);

         for (let k = 0; k < res3d.length; ++k)
            cont.add(res3d[k]);

         this.applySelectionOnSceneCreate(this.mgr.global_selection_id);
         this.applySelectionOnSceneCreate(this.mgr.global_highlight_id);

         this.first_time = false;
      }

      removeScene()
      {
         if (!this.glctrl) return;

         let cont = this.glctrl.getSceneContainer(this);
         while (cont.children.length > 0)
            cont.remove(cont.children[0]);

         this.mgr.UnRegisterSceneReceiver(this.id, this);
         this.first_time = true;
      }

      update3DObjectsVisibility(arr, all_ancestor_children_visible)
      {
         if (!arr) return;

         for (let k = 0; k < arr.length; ++k)
         {
            let elem = arr[k];
            if (elem.render_data)
            {
               let obj3d = this.getObj3D(elem.fElementId);
               if (obj3d)
               {
                  obj3d.visible = elem.fRnrSelf && all_ancestor_children_visible;
                  obj3d.all_ancestor_children_visible = all_ancestor_children_visible;
               }
            }

            this.update3DObjectsVisibility(elem.childs, elem.fRnrChildren && all_ancestor_children_visible);
         }
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
         if (this.glctrl) {
            if (this.need_visibility_update) {
               let p = this.mgr.GetElement(this.id);
               this.update3DObjectsVisibility(p.childs, true);
               this.need_visibility_update = false;
            }
            this.glctrl.viewer.render();

         }
      }

      elementAdded(el)
      {
         if ( ! this.glctrl) return;

         let obj3d =  this.makeGLRepresentation(el);
         if ( ! obj3d) return;

         // let scene = this.mgr.GetElement(this.id);
         let container = this.glctrl.getSceneContainer(this);

         container.add(obj3d);

         this.id2obj_map.set(el.fElementId, obj3d);

         this.need_visibility_update = true;
      }

      replaceElement(el) {
         if (!this.glctrl) return;

         let container = this.glctrl.getSceneContainer(this);

         try {
            let obj3d = this.getObj3D(el.fElementId);

            if(obj3d) container.remove(obj3d);

            obj3d = this.makeGLRepresentation(el);
            if (obj3d) {
               container.add(obj3d);
               this.id2obj_map.set(el.fElementId, obj3d);
            }
         }
         catch (e) {
            console.error("replace element", e);
         }
         this.need_visibility_update = true;
      }

      elementsRemoved(ids)
      {
         for (let i = 0; i < ids.length; i++)
         {
            let elId  = ids[i];
            let obj3d = this.getObj3D(elId);
            if (!obj3d) {
               let el = this.mgr.GetElement(elId);
               if (el && el.render_data) {
                  console.warning("EveScene.elementsRemoved can't find obj3d ", this.mgr.GetElement(el));
               }
               continue;
            }

            let container = this.glctrl.getSceneContainer(this);
            container.remove(obj3d);

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
            this.need_visibility_update = true;
         }

         // other change bits
         if (el.render_data) {
            if ((el.changeBit & this.mgr.EChangeBits.kCBObjProps) || (el.changeBit & this.mgr.EChangeBits.kCBColorSelection))
            {
               this.replaceElement(el);
            }
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
