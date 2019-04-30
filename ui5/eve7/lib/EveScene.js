/// @file EveScene.js

// TODO: add dependency from JSROOT components

sap.ui.define([
    'rootui5/eve7/lib/EveManager',
    'rootui5/eve7/lib/EveElements'
], function(EveManager, EveElements) {

   "use strict";

   /// constructor, handle for REveScene class

   function EveScene(mgr, scene, viewer)
   {
      this.mgr     = mgr;
      this.scene   = scene;
      this.id      = scene.fSceneId;
      this.viewer  = viewer;
      this.creator = new EveElements();
      this.creator.useIndexAsIs = (JSROOT.GetUrlOption('useindx') !== null);
      this.id2obj_map  = {}; // base on element id
      this.mid2obj_map = {}; // base on master id

      this.first_time = true;

      this.selected = {}; // generic map of selected objects

      // register ourself for scene events
      this.mgr.RegisterSceneReceiver(scene.fSceneId, this);

      // AMT temporary solution ... resolve with callSceneReceivers in EveManager.js
      scene.eve_scene = this;
   }

   EveScene.prototype.hasRenderData = function(elem)
   {
      if (elem === undefined)
         elem = this.mgr.GetElement(this.id);

      if (!elem) return false;
      if (elem.render_data) return true;
      if (elem.childs)
         for (var k = 0; k < elem.childs.length; ++k)
            if (this.hasRenderData(elem.childs[k])) return true;
      return false;
   }

   EveScene.prototype.makeGLRepresentation = function(elem)
   {
      if ( ! elem.render_data) return null;

      var fname = elem.render_data.rnr_func;
      var obj3d = this.creator[fname](elem, elem.render_data);

      if (obj3d)
      {
         obj3d._typename = "THREE.Mesh";

         // SL: this is just identifier for highlight, required to show items on other places, set in creator
         obj3d.geo_object = elem.fMasterId || elem.fElementId;
         obj3d.geo_name   = elem.fName; // used for highlight

         obj3d.scene = this; // required for get changes when highlight/selection is changed

         //AMT: reference needed in MIR callback
         obj3d.eveId  = elem.fElementId;
         obj3d.mstrId = elem.fMasterId;

         if (elem.render_data.matrix)
         {
            obj3d.matrixAutoUpdate = false;
            obj3d.matrix.fromArray( elem.render_data.matrix );
            obj3d.updateMatrixWorld(true);
         }

         return obj3d;
      }
   }

   EveScene.prototype.create3DObjects = function(all_ancestor_children_visible, prnt, res3d)
   {
      if (prnt === undefined) {
         prnt = this.mgr.GetElement(this.id);
         res3d = [];
      }

      if (!prnt || !prnt.childs) return res3d;

      for (var k = 0; k < prnt.childs.length; ++k)
      {
         var elem = prnt.childs[k];
         if (elem.render_data)
         {
            var fname = elem.render_data.rnr_func, obj3d = null;
            if (!this.creator[fname])
            {
               console.error("Function " + fname + " missing in creator");
            }
            else
            {
               var obj3d = this.makeGLRepresentation(elem);
               if (obj3d)
               {
                  // MT - should maintain hierarchy ????
                  // Easier to remove ... but might need sub-class of
                  // Object3D to separate "graphical" children and structural children.

                  res3d.push(obj3d);

                  this.id2obj_map[elem.fElementId] = obj3d;
                  if (elem.fMasterId) this.mid2obj_map[elem.fMasterId] = obj3d;

                  obj3d.visible = elem.fRnrSelf && all_ancestor_children_visible;
                  obj3d.all_ancestor_children_visible = all_ancestor_children_visible;
               }
            }
         }

         this.create3DObjects(elem.fRnrChildren && all_ancestor_children_visible, elem, res3d);
      }

      return res3d;
   }

   /** method insert all objects into three.js container */
   EveScene.prototype.redrawScene = function()
   {
      if (!this.viewer) return;

      var res3d = this.create3DObjects(true);
      if (!res3d.length && this.first_time) return;

      var cont = this.viewer.getThreejsContainer("scene" + this.id);
      while (cont.children.length > 0)
         cont.remove(cont.children[0]);

      for (var k = 0; k < res3d.length; ++k)
         cont.add(res3d[k]);

      this.applySelectionOnSceneCreate(this.mgr.global_selection_id);
      this.applySelectionOnSceneCreate(this.mgr.global_highlight_id);
      this.viewer.render();
      this.first_time = false;
   }

   EveScene.prototype.getObj3D = function(elementId, is_master)
   {
      var map = is_master ? this.mid2obj_map : this.id2obj_map;
      return map[elementId];
   }

   EveScene.prototype.update3DObjectsVisibility = function(arr, all_ancestor_children_visible)
   {
      if (!arr) return;

      for (var k = 0; k < arr.length; ++k)
      {
         var elem = arr[k];
         if (elem.render_data)
         {
            var obj3d = this.getObj3D(elem.fElementId);
            if (obj3d)
            {
               obj3d.visible = elem.fRnrSelf && all_ancestor_children_visible;
               obj3d.all_ancestor_children_visible = all_ancestor_children_visible;
            }
         }

         this.update3DObjectsVisibility(elem.childs, elem.fRnrChildren && all_ancestor_children_visible);
      }
   }



   EveScene.prototype.colorChanged = function(el)
   {
      if (!el.render_data) return;
      console.log("color change ", el.fElementId, el.fMainColor);

      this.replaceElement(el);
   }

   EveScene.prototype.replaceElement = function(el)
   {
      if (!this.viewer) return;

      var obj3d = this.getObj3D(el.fElementId);
      var all_ancestor_children_visible = obj3d.all_ancestor_children_visible;

      var container = this.viewer.getThreejsContainer("scene" + this.id);

      container.remove(obj3d);

      obj3d = this.makeGLRepresentation(el);
      obj3d.all_ancestor_children_visible = obj3d;

      container.add(obj3d);


      this.id2obj_map[el.fElementId] = obj3d;
      if (el.fMasterId) this.mid2obj_map[el.fMasterId] = obj3d;

      this.viewer.render();
   }

   EveScene.prototype.elementAdded = function(el)
   {
      if ( ! this.viewer) return;

      var obj3d =  this.makeGLRepresentation(el);
      if ( ! obj3d) return;

      // AMT this is an overkill, temporary solution
      var scene = this.mgr.GetElement(el.fSceneId);
      this.update3DObjectsVisibility(scene.childs, true);

      var container = this.viewer.getThreejsContainer("scene" + this.id);

      container.add(obj3d);

      this.id2obj_map[el.fElementId] = obj3d;
      if (el.fMasterId) this.mid2obj_map[el.fMasterId] = obj3d;
   }

   EveScene.prototype.visibilityChanged = function(el)
   {
      var obj3d = this.getObj3D( el.fElementId );

      if (obj3d)
      {
         obj3d.visible = obj3d.all_ancestor_children_visible && el.fRnrSelf;
      }
   }

   EveScene.prototype.visibilityChildrenChanged = function(el)
   {
      console.log("visibility children changed ", this.mgr, el);

      if (el.childs)
      {
         // XXXX Overkill, but I don't have obj3d for all elements.
         // Also, can do this traversal once for the whole update package,
         // needs to be managed from EveManager.js.
         // Or marked here and then recomputed before rendering (probably better).

         var scene = this.mgr.GetElement(el.fSceneId);

         this.update3DObjectsVisibility(scene.childs, true);
      }
   }

   /** interactive handler. Calculates selection state, apply to element and distribute to other scene */
   EveScene.prototype.processElementSelected = function(obj3d, col, indx, evnt)
   {
      // MT BEGIN
      // console.log("EveScene.prototype.processElementSelected", obj3d, col, indx, evnt);

      var is_multi  = evnt && evnt.ctrlKey;
      var is_secsel = indx !== undefined;

      var fcall = "NewElementPicked(" + obj3d.eveId + `, ${is_multi}, ${is_secsel}`;
      if (is_secsel)
      {
         fcall += ", { " + (Array.isArray(indx) ? indx.join(", ") : indx) + " }";
      }
      fcall += ")";

      this.mgr.SendMIR({ "mir":        fcall,
                         "fElementId": this.mgr.global_selection_id,
                         "class":      "REX::REveSelection"
                       });

      return true;
/*
      // MT END -- Sergey's code below

      // first decide if element selected or not

      var id  = obj3d.mstrId;
      var sel = this.selected[id];

      if (indx === undefined)
      {
         if ( ! sel)
         {
            sel = this.selected[id] = { id: id, col: col };
         }
         else
         {
            // means element selected, one should toggle back
            sel.col = null;
            delete this.selected[id];
         }
      }
      else
      {
         if ( ! sel)
         {
            sel = this.selected[id] = { id: id, col: col, indx: [] };
         }
         if (evnt && evnt.ctrlKey)
         {
            var pos = sel.indx.indexOf(indx);
            if (pos < 0) sel.indx.push(indx); else
                         sel.indx.splice(pos, 1);
         }
         else if (evnt && evnt.shiftKey)
         {
            if (sel.indx.length != 1) {
               sel.indx = [ indx ];
            } else {
               var min = Math.min(sel.indx[0], indx),
                   max = Math.max(sel.indx[0], indx);
               sel.indx = [];
               if (min != max)
                  for (var i=min;i<=max;++i)
                     sel.indx.push(i);
            }
         }
         else
         {
            if ((sel.indx.length == 1) && (sel.indx[0] == indx))
               sel.indx = [];
            else
               sel.indx = [ indx ];
         }

         if ( ! sel.indx.length)
         {
            sel.col  = null;
            sel.indx = undefined;
            delete this.selected[id]; // remove selection
         }
      }

      this.setElementSelected(id, sel.col, sel.indx, true);

      this.mgr.invokeInOtherScenes(this, "setElementSelected", id, sel.col, sel.indx);

      // when true returns, controller will not try to render itself
      return true;
*/
   }

   /** interactive handler */
   EveScene.prototype.processElementHighlighted = function(obj3d, col, indx, evnt)
   {
      // Need check for duplicates before call server, else server will un-higlight highlighted element
      // console.log("EveScene.prototype.processElementHighlighted", obj3d.eveId, col, indx, evnt);
      var is_multi  = false;
      var is_secsel = indx !== undefined;

      var so = this.mgr.GetElement(this.mgr.global_highlight_id);
      var a = so.prev_sel_list;

      // AMT presume there is no multiple highlight and multiple secondary selections
      // if that is the case in the futre write data in set and comapre sets

      // console.log("EveScene.prototype.processElementHighlighted compare Reveselection ", a[0], "incoming ", obj3d.eveId,indx);
      if (a.length == 1 ) {
         var h = a[0];
         if (h.primary == obj3d.eveId || h.primary == obj3d.mstrId ) {
            if (indx) {
               if (h.sec_idcs && h.sec_idcs[0] == indx) {
                  // console.log("EveScene.prototype.processElementHighlighted processElementHighlighted same index ");
                  return true;
               }
            }
            if (!indx && !h.sec_idcs.length) {
               // console.log("processElementHighlighted primARY SElection not changed ");
               return true;
            }
         }
      }

      var fcall = "NewElementPicked(" + obj3d.eveId + `, ${is_multi}, ${is_secsel}`;
      if (is_secsel)
      {
         fcall += ", { " + (Array.isArray(indx) ? indx.join(", ") : indx) + " }";
      }
      fcall += ")";


      this.mgr.SendMIR({ "mir":        fcall,
                         "fElementId": this.mgr.global_highlight_id,
                         "class":      "REX::REveSelection"
                       });

      return true;
   }


   EveScene.prototype.clearHighlight = function()
   {
      var so = this.mgr.GetElement(this.mgr.global_highlight_id);
      if (so.prev_sel_list.length) {
         console.log("clearHighlight", this);
         this.mgr.SendMIR({ "mir":        "SelectionCleared()",
                            "fElementId": this.mgr.global_highlight_id,
                            "class":      "REX::REveSelection"
                          });
      }

      return true;
   }

   EveScene.prototype.applySelectionOnSceneCreate =  function(selection_id)
   {
      var selection_obj = this.mgr.GetElement(selection_id);
      var pthis = this;
      selection_obj.prev_sel_list.forEach(function(rec) {
         var prl = pthis.mgr.GetElement(rec.primary);
         if (prl && prl.fSceneId == pthis.id) {
            pthis.SelectElement(selection_obj, rec.primary, rec.sec_idcs);
         }
         else {
            for (var impId of rec.implied)
            {
               var eli =  pthis.mgr.GetElement(impId);
               if (eli && eli.fSceneId == pthis.id) {
               console.log("CHECK select IMPLIED", pthis);
                  pthis.SelectElement(selection_obj, impId, rec.sec_idcs);
               }
            }
         }
      });
   }

   EveScene.prototype.SelectElement = function(selection_obj, element_id, sec_idcs)
   {
      this.viewer.outlinePass.id2obj_map[element_id] = this.viewer.outlinePass.id2obj_map[element_id] || [];

      // if(this.viewer.outlinePass.id2obj_map[element_id][selection_obj.fElementId] !== undefined) return;

      var stype = selection_obj.fName.endsWith("Selection") ? "select" : "highlight";
      var estype = THREE.OutlinePass.selection_enum[stype];

      console.log("EveScene.SelectElement ", selection_obj.fName, element_id, selection_obj.fElementId, this.viewer.outlinePass.id2obj_map);

      let res = {
         "sel_type": estype,
         "sec_sel": false,
         "geom": []
      };
      var obj3d = this.getObj3D( element_id );

      if(sec_idcs === undefined || sec_idcs.length == 0)
      {
         // exit if you try to highlight an object that has already been selected
         if(estype == THREE.OutlinePass.selection_enum["highlight"] && 
            this.viewer.outlinePass.id2obj_map[element_id][this.mgr.global_selection_id] !== undefined
         ) return;

         this.viewer.outlinePass.id2obj_map[element_id] = [];
         res.geom.push(obj3d);
      }
      else
      {
         var ctrl = obj3d.get_ctrl();
         ctrl.DrawForSelection(sec_idcs, res.geom);
         res.sec_sel = true;
      }
      this.viewer.outlinePass.id2obj_map[element_id][selection_obj.fElementId] = res;
   }

   EveScene.prototype.UnselectElement = function(selection_obj, element_id)
   {
      console.log("EveScene.UnselectElement ", selection_obj.fName, element_id, selection_obj.fElementId, this.viewer.outlinePass.id2obj_map);
      if (this.viewer.outlinePass.id2obj_map[element_id] !== undefined)
      {
	      delete this.viewer.outlinePass.id2obj_map[element_id][selection_obj.fElementId];
      }
   }
   /** returns true if highlight index is differs from current */
/*   EveScene.prototype.processCheckHighlight = function(obj3d, indx)
   {
      var id = obj3d.mstrId;
      // id = obj3d.eveId;

      if ( ! this.highlight || (this.highlight.id != id)) return true;

      // TODO: make precise checks with all combinations
      return (indx !== this.highlight.indx);
   }
*/
   /** function called by changes from server or by changes from other scenes */
  /* EveScene.prototype.setElementSelected = function(mstrid, col, indx, from_interactive)
   {
      if ( ! from_interactive)
         this.selected[mstrid] = { id: mstrid, col: col, indx: indx };

      this.drawSpecial(mstrid);
   }*/

   /** Called when processing changes from server or from interactive handler */
   /*
   EveScene.prototype.setElementHighlighted = function(mstrid, col, indx, from_interactive)
   {
      // check if other element was highlighted at same time - redraw it
      if (this.highlight && (this.highlight.id != mstrid)) {
         delete this.highlight;
         this.drawSpecial(mstrid);
      }

      if ( ! col)
         delete this.highlight;
      else
         this.highlight = { id: mstrid, col: col, indx: indx };

      this.drawSpecial(mstrid, true);
   }*/
/*
   EveScene.prototype.drawSpecial = function(mstrid, prefer_highlight)
   {
      var obj3d = this.getObj3D( mstrid, true );
      if ( ! obj3d || ! obj3d.get_ctrl) obj3d = this.getObj3D( mstrid );
      if ( ! obj3d || ! obj3d.get_ctrl) return false;

      var h1 = this.highlight && (this.highlight.id == mstrid) ? this.highlight : null;
      var h2 = this.selected[mstrid];
      var ctrl = obj3d.get_ctrl();

      var did_change = false;

      if (ctrl.separateDraw)
      {
         var p2 = "s", p1 = "h";
         // swap h1-h2
         if (!prefer_highlight) { var h = h1; h1 = h2; h2 = h; p2 = "h"; p1 = "s"; }
         if (ctrl.drawSpecial(h2 ? h2.col : null, h2 ? h2.indx : undefined, p2)) did_change = true;
         if (ctrl.drawSpecial(h1 ? h1.col : null, h1 ? h1.indx : undefined, p1)) did_change = true;
      }
      else
      {
         var h = prefer_highlight ? (h1 || h2) : (h2 || h1);
         did_change = ctrl.drawSpecial(h ? h.col : null, h ? h.indx : undefined);
      }

      if (did_change && this.viewer){
         this.viewer.render();
         if(!prefer_highlight){
            if(!this.selected[mstrid]){
               // delete if its not selected anymore?
               delete this.viewer.outlinePass.id2obj_map[mstrid];
            } else {
               // is secondary selection
               let sec_sel = this.selected[mstrid].indx;

               // if its not secondary selection pass the object
               if(!sec_sel)
                  this.viewer.outlinePass.id2obj_map[mstrid] = obj3d;
               // if its secondary selection pass all the new objects returned by 'drawSpecial'
               else {
                  this.viewer.outlinePass.id2obj_map[mstrid] = []; // reset
                  if(false){
                     if(obj3d.sl_special) this.viewer.outlinePass.id2obj_map[mstrid].push(obj3d.sl_special);
                     if(obj3d.sm_special) this.viewer.outlinePass.id2obj_map[mstrid].push(obj3d.sm_special);
                  } else {
                     for(const child of obj3d.children){
                        if(child.jsroot_special)
                           this.viewer.outlinePass.id2obj_map[mstrid].push(child);
                     }
                  }
                  // print the new elements
                  console.log(this.viewer.outlinePass.id2obj_map[mstrid]);
               }
            }
         }
      }

      return did_change;
   }
*/

   EveScene.prototype.elementRemoved = function()
   {
   }

   EveScene.prototype.beginChanges = function()
   {
   }

   EveScene.prototype.endChanges = function()
   {
      if (this.viewer)
         this.viewer.render();
   }

   EveScene.prototype.onSceneCreate = function(id)
   {
      this.redrawScene();
   }

   EveScene.prototype.sceneElementChange = function(msg)
   {
      var el = this.mgr.GetElement(msg.fElementId);
      this[msg.tag](el);
   }

   EveScene.prototype.elementsRemoved = function(ids) {
      for (var  i = 0; i < ids.length; i++)
      {
         var elId = ids[i];
         var obj3d = this.getObj3D(elId);
         if (!obj3d) {
            console.log("ERROOR cant find obj3d");
         }

         var container = this.viewer.getThreejsContainer("scene" + this.id);
         container.remove(obj3d);

         delete this.id2obj_map[elId];
      }
   }

   return EveScene;

});
