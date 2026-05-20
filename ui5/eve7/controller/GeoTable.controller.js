sap.ui.define([
    'rootui5/geom/controller/GeomHierarchy.controller',
    'rootui5/geom/model/GeomBrowserModel',
    'sap/ui/table/Column',
    'sap/ui/Device',
    'sap/ui/unified/Menu',
    'sap/ui/unified/MenuItem',
    'sap/ui/core/Popup',
    'sap/ui/layout/HorizontalLayout',
    'rootui5/geom/lib/ColorBox',
    'sap/m/CheckBox',
    'sap/m/Text',
    'sap/ui/core/Icon',
    'sap/ui/core/UIComponent',
], function (GeomHierarchy, GeomBrowserModel, tableColumn, Device, Menu, MenuItem, Popup, HorizontalLayout, GeomColorBox, mCheckBox, mText, Icon, UIComponent) {

    "use strict";

    return GeomHierarchy.extend("rootui5.eve7.controller.GeoTable", {

        onInit: function () {
            // disable narrowing axis range
            EVE.JSR.settings.Zooming = false;

            let data = this.getView().getViewData();
            if (data) {
                this.setupManagerAndViewType(data.eveViewerId, data.mgr);
            }
            else {
                UIComponent.getRouterFor(this).getRoute("GeoTable").attachPatternMatched(this.onViewObjectMatched, this);
            }
        },

        onViewObjectMatched: function (oEvent) {
            let args = oEvent.getParameter("arguments");
            this.setupManagerAndViewType(EVE.$eve7tmp.eveViewerId, EVE.$eve7tmp.mgr);
            delete EVE.$eve7tmp;
        },

        setupManagerAndViewType: function (eveViewerId, mgr) {
            this.eveViewerId = eveViewerId;
            this.mgr = mgr;

            let eviewer = this.mgr.GetElement(this.eveViewerId);
            let sceneInfo = eviewer.childs[0];
            let scene = this.mgr.GetElement(sceneInfo.fSceneId);
            let topNodeEve = scene.childs[0];

            //let h = this.byId('geomHierarchyPanel');

            let websocket = this.mgr.handle.createChannel();

            // h.getController().configure({
            this.configure({
                websocket,
                show_columns: true,
                jsroot: EVE.JSR
            });

            this.model.addNodeAttributes = function (node, item) {
                node._node = this.provideLogicalNode(item);

                if (item.pvis !== undefined) {
                    node.pvisible = item.pvis != 0;
                    node.avisible = item.vis != 0;
                    node.top = item.top;
                } else {
                    console.error("add node attributes not handled");
                }
            };

            console.log('channel id is', websocket.getChannelId());
            this.mgr.handle.send("SETCHANNEL:" + topNodeEve.fElementId + "," + websocket.getChannelId());
            topNodeEve.websocket = websocket;
        },


        configureTable(show_columns) {

            // create model only for browser - no need for anybody else
            this.model = new GeomBrowserModel();

            this.model.useIndexSuffix = false;

            let t = this.byId("treeTable");

            t.setModel(this.model);

            t.setRowHeight(20);

            this.model.assignTreeTable(t);

            t.addColumn(new tableColumn({
                label: 'Description',
                tooltip: 'Name of geometry nodes',
                autoResizable: true,
                width: show_columns ? '50%' : '100%',
                visible: true,
                tooltip: "{name}",
                template: new HorizontalLayout({
                    content: [
                        new Icon({ visible: '{top}', src: 'sap-icon://badge', tooltip: '{name} selected as top node' }).addStyleClass('sapUiTinyMarginEnd'),
                        new mText({ text: '{name}', tooltip: '{name}', wrapping: false })
                    ]
                })
            }));
            t.addColumn(new tableColumn({
                label: 'Visibility',
                tooltip: 'Visibility flags',
                autoResizable: true,
                visible: true,
                width: '20%',
                template: new HorizontalLayout({
                    content: [
                        new mCheckBox({ enabled: true, visible: true, selected: "{pvisible}", select: evnt => this.changeVisibility(evnt, true), tooltip: 'Visibility Self' }),
                        new mCheckBox({ enabled: true, visible: true, selected: "{avisible}", select: evnt => this.changeVisibility(evnt), tooltip: 'Visibility Children' })
                    ]
                })
            }));
            t.addColumn(new tableColumn({
                label: 'Color',
                tooltip: 'Color of geometry volumes',
                width: '10%',
                autoResizable: true,
                visible: true,
                template: new GeomColorBox({ color: "{_node/color}", visible: "{= !!${_node/color}}" })
            }));
            t.addColumn(new tableColumn({
                label: 'Material',
                tooltip: 'Material of the volumes',
                width: '20%',
                autoResizable: true,
                visible: true,
                template: new mText({ text: "{_node/material}", wrapping: false })
            }));

            this._columnResized = 0;

            // catch re-rendering of the table to assign handlers
            t.addEventDelegate({
                onAfterRendering() {
                    this.assignRowHandlers();
                    if (this._columnResized < 1) return;
                    this._columnResized = 0;
                    let fullsz = 4;

                    t.getColumns().forEach(col => {
                        if (col.getVisible()) fullsz += 4 + col.$().width();
                    });

                    this.viewer?.byId('geomViewerApp').getAggregation('_navMaster').setWidth(fullsz + 'px');
                }
            }, this);

            t.attachEvent("columnResize", {}, evnt => {
                this._columnResized++;
            }, this);

        },

        switchSingle: function () {
            let oRouter = UIComponent.getRouterFor(this);
            EVE.$eve7tmp = { mgr: this.mgr, eveViewerId: this.eveViewerId };

            oRouter.navTo("GeoTable", { viewName: this.mgr.GetElement(this.eveViewerId).fName });
        },

        swap: function () {
            this.mgr.controllers[0].switchViewSides(this.mgr.GetElement(this.eveViewerId));
        },

        detachViewer: function () {
            this.mgr.controllers[0].removeView(this.mgr.GetElement(this.eveViewerId));
            this.destroy();
        },
        cdTop() {
            this.websocket.send('CDTOP:');

            let len = this.model?.getLength() ?? 0;
            for (let n = 0; n < len; ++n)
                this.model?.setProperty(`/nodes/${n}/top`, false);
            // this.model?.setProperty(ctxt.getPath() + '/top', true);
            // this.doReload();
        },
        cdUp() {
            this.websocket.send('CDUP:');

            let len = this.model?.getLength() ?? 0;
            for (let n = 0; n < len; ++n)
                this.model?.setProperty(`/nodes/${n}/top`, false);
            // this.model?.setProperty(ctxt.getPath() + '/top', true);
            // this.doReload();
        },

        /** @summary Get entry with physical node visibility */
        getPhysVisibilityEntry(path, force) {
            console.error("AMT get PHY entry unused ...");

            let stack = this.getStackByPath(this.fullModel, path);
            if (stack === null)
                return;

            let len = stack.length;

            for (let i = 0; i < this.physVisibility?.length; ++i) {
                let item = this.physVisibility[i], match = true;
                if (len != item.stack?.length) continue;
                for (let k = 0; match && (k < len); ++k)
                    if (stack[k] != item.stack[k])
                        match = false;
                if (match)
                    return item;
            }
        },

        /** @summary invoked when visibility checkbox clicked */
        changeVisibility(oEvent, physical) {
            let row = oEvent.getSource(),
                flag = oEvent.getParameter('selected'),
                ctxt = row?.getBindingContext(),
                ttt = ctxt?.getProperty(ctxt?.getPath());

            if (!ttt?.path)
                return console.error('Fail to get path');


            let msg = '';

            if (physical) {
                ttt.pvisible = flag;
                ttt._elem.pvis = flag;
                msg = flag ? 'SHOW' : 'HIDE';
            } else {
                ttt.avisible = flag;
                ttt._elem.vis = flag;
                msg = "SETVI" + (flag ? '1' : '0');
            }

            msg += ':' + JSON.stringify(ttt.path);

            this.websocket.send(msg);
        },

        onCellContextMenu(oEvent) {
            if (Device.support.touch)
                return; //Do not use context menus on touch devices

            let ctxt = oEvent.getParameter('rowBindingContext'),
                colid = oEvent.getParameter('columnId'),
                prop = ctxt?.getProperty(ctxt.getPath());

            oEvent.preventDefault();

            if (!prop?._elem) return;

            if (!this._oIdContextMenu) {
                this._oIdContextMenu = new Menu();
                this.getView().addDependent(this._oIdContextMenu);
            }

            this._oIdContextMenu.destroyItems();

            this._oIdContextMenu.addItem(new MenuItem({
                text: 'Set as top',
                select: () => {
                    this.setPhysTopNode(prop.path);
                    this.websocket.send('SETAPEX:' + JSON.stringify(prop.path));

                    let len = this.model?.getLength() ?? 0;
                    for (let n = 0; n < len; ++n)
                        this.model?.setProperty(`/nodes/${n}/top`, false);
                    this.model?.setProperty(ctxt.getPath() + '/top', true);
                }
            }));

            //Open the menu on the cell
            let oCellDomRef = oEvent.getParameter("cellDomRef");
            this._oIdContextMenu.open(false, oCellDomRef, Popup.Dock.BeginTop, Popup.Dock.BeginBottom, oCellDomRef, "none none");
        }
    });
});
