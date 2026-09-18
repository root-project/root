sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/table/Column',
   'sap/m/Text',
   'sap/m/Label',
   'sap/ui/core/UIComponent'
], function (Controller, JSONModel, tableColumn, mText, mLabel, UIComponent) {

   "use strict";

   // Flat list of the geometry's overlaps and extrusions. Unlike GeoTable this is
   // not a lazily expanded hierarchy -- the whole list arrives with the element, so
   // a plain JSONModel over a flat table is all that is needed. Picking a row sends
   // a SelectOverlap MIR; the server rebuilds the 3D scene to show just that one.
   return Controller.extend("rootui5.eve7.controller.GeoOverlapTable", {

      onInit: function () {
         let data = this.getView().getViewData();
         if (data) {
            this.setupManagerAndViewType(data.eveViewerId, data.mgr);
         } else {
            UIComponent.getRouterFor(this).getRoute("GeoOverlapTable")
               .attachPatternMatched(this.onViewObjectMatched, this);
         }
      },

      onViewObjectMatched: function () {
         this.setupManagerAndViewType(EVE.$eve7tmp.eveViewerId, EVE.$eve7tmp.mgr);
         delete EVE.$eve7tmp;
      },

      setupManagerAndViewType: function (eveViewerId, mgr) {
         this.eveViewerId = eveViewerId;
         this.mgr = mgr;

         let eviewer = this.mgr.GetElement(this.eveViewerId),
             sceneInfo = eviewer.childs[0],
             scene = this.mgr.GetElement(sceneInfo.fSceneId);

         this.tableElement = scene.childs[0];

         this.configureTable();
         this.fillTable();
         // deliberately not RegisterController(): the row data isn't tied to an
         // eve-viewer/scene the generic controller protocol updates, and this
         // controller doesn't implement updateViewerAttributes().
         //
         // A precision change does rebuild the table though, so this registers for
         // sceneElementChange on the "Overlap List" scene to notice that specific
         // update -- see sceneElementChange() below.
         this.mgr.RegisterSceneReceiver(sceneInfo.fSceneId, this);
      },

      configureTable: function () {
         this.model = new JSONModel({ overlaps: [] });
         // set on the view, not just the table, so the precision Input (a sibling
         // control) can bind against it too
         this.getView().setModel(this.model);

         // matches the "Reset camera" checkbox's default state in the view
         this._resetCamera = true;

         let t = this.byId("overlapTable");
         t.setRowHeight(20);

         const col = (label, tip, width, path) => new tableColumn({
            label: new mLabel({ text: label }),
            tooltip: tip,
            width: width,
            autoResizable: true,
            visible: true,
            template: new mText({ text: path, wrapping: false })
         });

         // Fixed widths, not percentages: this view is usually docked into a narrow
         // side panel, where percentages truncate every column to three characters.
         // The volume names are the ones worth the room, so they get what is left.
         t.addColumn(col('Size [cm]', 'How far the two volumes interpenetrate', '5.5rem', '{valueStr}'));
         t.addColumn(col('Type', 'Extrusion (a daughter sticking out of its mother), or a real overlap between two daughters', '5.5rem', '{kind}'));
         t.addColumn(col('Volume A', 'First volume; the mother, for an extrusion', '13rem', '{vol1}'));
         t.addColumn(col('Volume B', 'Second volume', '13rem', '{vol2}'));
         // in per-overlap mode this counts placements of the mother; in unique mode,
         // how many entries the row stands for
         t.addColumn(col('N', 'Per-overlap mode: placements of the mother volume. Unique mode: how many instances this one row covers', '3rem', '{instStr}'));
         t.addColumn(col('Points', 'Marker points on the offending surface; in brackets, points dropped as non-finite', '6rem', '{pointsStr}'));
         t.addColumn(col('Name', 'TGeoOverlap name', '6rem', '{name}'));
      },

      fillTable: function () {
         let unique = !!this._unique,
             list = (unique ? this.tableElement?.fGroups : this.tableElement?.fOverlaps) || [];

         // sort worst first: that is the order somebody debugging a geometry wants
         let rows = list.map(o => Object.assign({}, o, {
            valueStr: (o.value !== undefined) ? o.value.toFixed(5) : '',
            instStr: unique ? `${o.ninst}` : `${o.nplaced}`,
            pointsStr: o.ndropped > 0 ? `${o.npoints} (${o.ndropped} bad)` : `${o.npoints}`
         })).sort((a, b) => b.value - a.value);

         this.model.setData({ overlaps: rows, precision: this.tableElement?.fPrecision });
         this._scanGen = this.tableElement?.fScanGen; // last list this controller has actually rendered

         let worst = rows.length ? rows[0].value.toFixed(4) : '0',
             total = this.tableElement?.fOverlaps?.length ?? 0;
         this.byId("summaryLabel").setText(
            !rows.length ? 'no overlaps found'
               : unique ? `${rows.length} distinct of ${total} overlaps, worst ${worst} cm`
                        : `${rows.length} overlaps, worst ${worst} cm`);

         // Show the worst one straight away. Only the selected overlap is drawn, so
         // without this the 3D view opens empty with nothing saying why.
         if (rows.length && !this._autoSelected) {
            this._autoSelected = true;
            let t = this.byId("overlapTable");
            t.setSelectedIndex(0);
            this.sendSelect(rows[0].idx);
         }
      },

      onRowSelect: function (oEvent) {
         let idx = oEvent.getParameter("rowIndex");
         if (idx === undefined || idx < 0) return;
         let ctx = this.byId("overlapTable").getContextByIndex(idx),
             row = ctx?.getProperty(ctx.getPath());
         if (!row) return;
         this.sendSelect(row.idx);
      },

      // The table sets the right-clicked row's binding context on the context
      // menu before opening it, so the pressed MenuItem inherits it. Falls back
      // to whatever row is currently selected if that's ever not the case.
      onPrintOverlap: function (oEvent) {
         let ctx = oEvent.getSource().getBindingContext(),
             row = ctx?.getProperty(ctx.getPath());
         if (!row) {
            let t = this.byId("overlapTable"), idx = t.getSelectedIndex();
            if (idx < 0) return;
            ctx = t.getContextByIndex(idx);
            row = ctx?.getProperty(ctx.getPath());
            if (!row) return;
         }
         let call = this._unique ? `PrintUniqueOverlap(${row.idx})` : `PrintOverlap(${row.idx})`;
         this.mgr.SendMIR(call, this.tableElement.fElementId,
                          "ROOT::Experimental::REveGeoOverlapTable");
      },

      onToggleUnique: function (oEvent) {
         this._unique = oEvent.getParameter("pressed");
         this._autoSelected = false;          // re-select the worst in the new mode
         this.byId("overlapTable").clearSelection();
         this.fillTable();
      },

      // On by default; turning it off keeps a manually-picked viewing angle across
      // rows instead of resetting the camera on every click.
      onToggleResetCamera: function (oEvent) {
         this._resetCamera = oEvent.getParameter("selected");
      },

      // Called back from EveManager.callSceneReceivers() for any element that
      // changes in the "Overlap List" scene. There's only one such element -- the
      // table itself -- but SelectOverlap()/SelectUniqueOverlap() *also* stamp it
      // on every ordinary row click (so fSelected/fSelectedGroup reach the client),
      // and re-filling + auto-selecting on THAT would immediately send another
      // SelectOverlap, stamp again, and loop forever. fScanGen only changes when
      // ScanOverlaps() actually ran (initial build or SetPrecision), so that's the
      // only case this reacts to.
      sceneElementChange: function () {
         if (this.tableElement.fScanGen === this._scanGen) return;
         this._autoSelected = false; // re-select the worst in the freshly rescanned table
         this.byId("overlapTable").clearSelection();
         this.fillTable();
      },

      // Registering as a scene receiver (above) also opts this controller into
      // EveManager.ServerEndRedrawCallback()'s generic per-frame hooks, which
      // call these on every receiver unconditionally -- unlike callSceneReceivers()
      // there's no typeof guard, so without these two no-ops every redraw logs
      // "item.endChanges is not a function". Nothing here needs the hooks
      // themselves: fillTable() already runs synchronously off sceneElementChange.
      beginChanges: function () {},
      endChanges: function () {},

      onPrecisionChange: function (oEvent) {
         let val = parseFloat(oEvent.getParameter("value"));
         if (!isFinite(val) || val <= 0) {
            // revert through the model -- the binding is two-way, so the Input
            // may already hold the bad value by the time "change" fires
            this.model.setProperty("/precision", this.tableElement?.fPrecision);
            return;
         }
         this.byId("summaryLabel").setText("rescanning...");
         this.mgr.SendMIR(`SetPrecision(${val})`, this.tableElement.fElementId,
                          "ROOT::Experimental::REveGeoOverlapTable");
         // the table refreshes itself once sceneElementChange() fires with the result
      },

      onClear: function () {
         this.byId("overlapTable").clearSelection();
         this.byId("summaryLabel").setText(this.byId("summaryLabel").getText().replace(/ — .*$/, ''));
         this.sendSelect(-1);
      },

      sendSelect: function (idx) {
         let call = this._unique ? `SelectUniqueOverlap(${idx})` : `SelectOverlap(${idx})`;
         this.mgr.SendMIR(call, this.tableElement.fElementId,
                          "ROOT::Experimental::REveGeoOverlapTable");
         if (this._resetCamera && idx >= 0)
            this.refocusWhenReady();
      },

      /** @summary Frame the newly drawn overlap.
        *
        * Without this the camera keeps whatever it was looking at, and since the
        * volumes involved can be metres across you usually end up inside one of them
        * looking at flat colour.
        *
        * The shapes arrive asynchronously and there is no client-side hook for "that
        * scene finished updating", so this simply re-frames a few times over the next
        * second and a half. Resetting the camera is idempotent, so the extra calls
        * cost nothing and one of them is guaranteed to land after the shapes do. */
      refocusWhenReady: function () {
         clearInterval(this._focusTimer);
         let ticks = 0;
         this._focusTimer = setInterval(() => {
            for (let ctrl of this.mgr.gl_controllers || []) {
               let v = ctrl.viewer;
               // scene_bbox only exists once the viewer has finished bootstrapping;
               // calling resetCamera() before that throws inside recalcSceneBBox and
               // leaves the viewer unable to render at all.
               if (!v || !v.scene || !v.scene_bbox || typeof v.resetCamera !== "function")
                  continue;
               try {
                  v.resetCamera();
               } catch (e) {
                  console.warn("GeoOverlapTable: camera reset skipped", e);
               }
            }
            if (++ticks >= 6)
               clearInterval(this._focusTimer);
         }, 250);
      },

      onExit: function () {
         clearInterval(this._focusTimer);
         if (this.mgr && this.tableElement)
            this.mgr.UnRegisterSceneReceiver(this.tableElement.fSceneId, this);
      },

      switchSingle: function () {
         EVE.$eve7tmp = { mgr: this.mgr, eveViewerId: this.eveViewerId };
         UIComponent.getRouterFor(this).navTo("GeoOverlapTable",
            { viewName: this.mgr.GetElement(this.eveViewerId).fName });
      },

      swap: function () {
         this.mgr.controllers[0].switchViewSides(this.mgr.GetElement(this.eveViewerId));
      },

      detachViewer: function () {
         this.mgr.controllers[0].removeView(this.mgr.GetElement(this.eveViewerId));
         this.destroy();
      }
   });
});
