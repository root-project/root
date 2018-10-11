/**
 * This submodule is responsible for creating visualization for simple neural networks and deep neural networks.
 * Simple neural networks are created with d3js, and HTML5 canvas element is used for visualization of deep neural networks.
 * Interactions supported (non deep networks):
 *          - Mouseover (node, weight): focusing
 *          - Zooming and grab and move supported
 *          - Reset: double click
 * Interactions supported (deep networks):
 *          - Zooming and grab and move supported
 * Author: Attila Bagoly <battila93@gmail.com>
 * Created:  6/9/16
 */

(function(factory){
    define(['d3'], function(d3){
        return factory({}, d3);
    });
}(function(NeuralNetwork, d3) {

    // https://github.com/wbkd/d3-extended
    d3.selection.prototype.moveToFront = function() {
        return this.each(function(){
            this.parentNode.appendChild(this);
        });
    };
    d3.selection.prototype.moveToBack = function() {
        return this.each(function() {
            var firstChild = this.parentNode.firstChild;
            if (firstChild) {
                this.parentNode.insertBefore(this, firstChild);
            }
        });
    };

    var style = {
        "neuron": {
            "colors": {
                "input": "#00A000",
                "hidden": "#0000C7",
                "output": "#F6BD00",
                "bias": "#8F686F"
            },
            "mouseon": {
                "change_radius": 2,
                "alpha": 0.2
            }
        },
        "synapse": {
            "colors": {
                "negative": "#00005E",
                "positive": "#FF4B00"
            },
            "deepNet_colors":{
                "negative": "rgba(0,0,94, 0.4)",
                "positive": "rgba(94,0,0, 0.4)"//"#FF4B00"
            },
            "default_width_range": [0.5, 5],
            "width_range": [0.5, 5],
            "deepNet_default_width_range": [0.5, 2],
            "deepNet_width_range": [0.5, 2],
            "default_alpha": 0.7,
            "alpha": 0.7,
            "mouseon": {
                "width_range": [0.5, 10],
                "alpha": 0.1
            }
        },
        "variables":{
            "labels_layer0_padding": 0.03
        },
        "legend": {
            "pos": {"x": 150, "y": 10},
            "rect": {"width": 10, "height":10},
            "dy": 20,
            "padding": 10,
            "deepNet_colors": {
                "negative": "rgb(0,0,94)",
                "positive": "rgb(94,0,0)"//"#FF4B00"
            }
        }
    };

    var canvas;

    var getNeuronNumber = function(net, layer_index){
        return Number(Object.keys(net["layout"]["layer_"+layer_index]).length-1);
    };

    var getNeuronsAttr = function (net, num_layers, layer_index) {
        var numn = getNeuronNumber(net, layer_index);
        var neuronsAttr = Array(numn);
        for(var i=0;i<numn;i++){
            neuronsAttr[i] = {
                "position": {
                    "x": (layer_index + 0.5) * canvas.width / (num_layers),
                    "y": (i + 0.5) * canvas.height / numn
                },
                "radius": canvas.height/(numn+(numn>5?0:5))/4,
                "type":  (i==(numn-1) ? "bias" : (layer_index==0 ? "input" : "hidden")),
                "neuron": i,
                "layer": layer_index,
            };
            if (layer_index==(num_layers-1)){
                neuronsAttr[i]["type"] = "output";
            }
        }
        return neuronsAttr;
    };

    var getWeights = function(net, layer, neuron){
        var neuron = net["layout"]["layer_"+layer]["neuron_"+neuron];
        if (neuron["nsynapses"]!=0) return neuron["weights"];
        return [];
    };

    var getMinMaxWeight = function(net, num_layers){
        var max = -1e30;
        var min =  1e30;
        var tmp;
        for(var i=0;i<num_layers;i++){
            for(var j=0;j<getNeuronNumber(net, i);j++){
                tmp = d3.max(getWeights(net, i, j));
                if (max < tmp) max = tmp;
                tmp = d3.min(getWeights(net, i, j));
                if (min > tmp) min = tmp;
            }
        }
        return {"min": min, "max": max};
    };

    var getSynapses = function(net, layer_index, neuron, pos, layer2){
        var weights  = getWeights(net, layer_index, neuron);
        var synapses = Array(weights.length);
        for(var i in weights){
            synapses[i] = {
                "layer": layer_index,
                "neuron": neuron,
                "nextlayer_neuron": i,
                "pos": [pos, layer2[i].position],
                "weight": weights[i],
                "type":   (weights[i]<0 ? "negative" : "positive")
            };
        }
        return synapses;
    };

    var getInputLabels = function(net, layer0){
        var labels = net["variables"];
        labels.push("Bias node");
        var variables = Array(labels.length);
        for(var i in layer0){
            variables[i] = {
                "x": layer0[i].position.x-style["variables"]["labels_layer0_padding"]*canvas.width,
                "y": layer0[i].position.y,
                "text": labels[i] + ":"
            };
        }
        return variables;
    };

    var drawInputLabels = function(group){
        group.append("text")
            .text(function(d){return d[1].text;})
            .attr("x", function(d){return d[1].x-this.getComputedTextLength();})
            .attr("y", function(d){return d[1].y+0.25*this.getBBox().height;});
    };

    var drawNeurons = function (svg, net, neuronsattr, layer_num, input_variable_labels) {
        if (input_variable_labels!==undefined){
            var dat = d3.zip(neuronsattr, getInputLabels(net, neuronsattr));
        } else {
            var dat = d3.zip(neuronsattr, Array(neuronsattr.length));
        }
        var group = svg.append("g").attr("id", "layer_"+layer_num).attr("class", "layer").selectAll("g")
            .data(dat)
            .enter()
            .append("g").attr("id", function(d){return "neuron_"+layer_num+""+d[0].neuron;});
        group.append("circle")
            .attr('r',     function(d){return d[0].radius})
            .attr('cx',    function(d){return d[0].position.x;})
            .attr('cy',    function(d){return d[0].position.y;})
            .style("fill", function(d){return style["neuron"]["colors"][d[0].type];});
        if (input_variable_labels!==undefined){
            drawInputLabels(group)
        }
        animate(svg, group);
    };

    var scaleSynapsisPos = d3.scale.linear();
    var scaleSynapsisNeg = d3.scale.linear();

    var synapse = d3.svg.line()
        .x(function(d){return d.x;})
        .y(function(d){return d.y;})
        .interpolate("linear");

    var drawSynapses = function(svg, net, layer1, layer1_index, layer2){
        for(var idx in layer1){
            var synapses = getSynapses(net, layer1_index, idx, layer1[idx].position, layer2);
            svg.select("g#neuron_"+layer1_index+""+idx)
                .selectAll("path")
                .data(synapses)
                .enter()
                .append("path").moveToBack()
                .attr("d", function(d){return synapse(d.pos);})
                .attr("stroke", function(d){return style["synapse"]["colors"][d.type]})
                .attr("stroke-width", function(d){
                    return d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                })
                .attr("stroke-opacity", style["synapse"]["alpha"]);
        }
    };

    var animate = function(svg, group){
        style.synapse.width_range = Object.assign({}, style.synapse.default_width_range);
        style.synapse.alpha = Object.assign({}, style.synapse.default_alpha);
        group.on('mouseover', function(d) {
            scaleSynapsisPos.range(style["synapse"]["mouseon"]["width_range"]);
            scaleSynapsisNeg.range(style["synapse"]["mouseon"]["width_range"]);
            var self = d3.select(this).moveToFront().transition();
            self.selectAll("path")
                .style("stroke-opacity", 1)
                .attr("stroke-width", function(d){
                    return d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                });
            self.selectAll("circle")
                .style("fill-opacity", 1)
                .attr("r", function(d){return d[0].radius*style["neuron"]["mouseon"]["change_radius"]});
            self.selectAll("text")
                .attr("x", function(d){return d[1].x-d[0].radius-this.getComputedTextLength();});

            var allbutnotthis = svg.selectAll("g.layer").selectAll("g")
                .filter(function(x){return !(d[0].neuron==x[0].neuron&&d[0].layer==x[0].layer);}).transition();
            allbutnotthis.selectAll("circle").filter(function(x){return (d[0].layer+1)!=x[0].layer})
                .style("fill-opacity", style["neuron"]["mouseon"]["alpha"])
                .attr("r", function(d){return d[0].radius});
            allbutnotthis.selectAll("path")
                .style("stroke-opacity", style["synapse"]["mouseon"]["alpha"]);
        });
        group.on('mouseout', function(d){
            scaleSynapsisPos.range(style["synapse"]["width_range"]);
            scaleSynapsisNeg.range(style["synapse"]["width_range"]);
            var gg = svg.selectAll("g.layer").selectAll("g").transition();
            gg.selectAll("circle")
                .style("fill-opacity", 1)
                .attr("r", function(d){return d[0].radius;});
            gg.selectAll("path")
                .style("stroke-opacity", 1)
                .attr("stroke-width", function(d){
                    return d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                });
            gg.selectAll("text")
                .attr("x", function(d){return d[1].x-this.getComputedTextLength();});
        });
    };

    var drawLegend = function(svg){
        var labels = [
            {"c": style["synapse"]["colors"]["positive"], "txt": "Positive weight"},
            {"c": style["synapse"]["colors"]["negative"], "txt": "Negative weight"}
        ];
        var attr = style["legend"];

        var container = svg.append("g").attr("id", "legend");
        container.selectAll("g")
            .data(labels)
            .enter()
            .append("g")
            .each(function(d, i){
                var g = d3.select(this);
                g.append("rect")
                    .attr("x", canvas.width-attr.pos.x)
                    .attr("y", attr.pos.y+i*attr.dy)
                    .attr("width", attr.rect.width)
                    .attr("height", attr.rect.height)
                    .style("fill", function(d){return d.c;});
                g.append("text")
                    .attr("x", canvas.width-attr.pos.x+attr.rect.width+attr.padding)
                    .attr("y", attr.pos.y+i*attr.dy+attr.rect.height)
                    .text(function(d){return d.txt;})
                    .style("fill", function(d){return d.c;});
            });
    };

    NeuralNetwork.draw = function (divid, netobj) {
        if ("layers" in netobj && "synapses" in netobj) return NeuralNetwork.drawDeepNetwork(divid, netobj, true);
        if ("layers" in netobj && "Biases" in netobj["layers"][0]) return NeuralNetwork.drawDeepNetwork(divid, netobj);

        var svg, net;

        var div = d3.select("#"+divid);
        canvas = {
            width:  div.property("style")["width"],
            height: div.property("style")["height"]
        };

        net = netobj;
        style.synapse.width_range = Object.assign({}, style.synapse.default_width_range);
        style.synapse.alpha = Object.assign({}, style.synapse.default_alpha);
        scaleSynapsisPos.range(style["synapse"]["width_range"]);
        scaleSynapsisNeg.range(style["synapse"]["width_range"]);

        svg = div.append("svg")
            .attr("id", "svg_"+divid)
            .attr("width", canvas.width)
            .attr("height", canvas.height);
        Object.keys(canvas).forEach(function (key) {
            canvas[key] = Number(canvas[key].replace("px",""))
        });

        var num_layers = Number(net["layout"]["nlayers"]);

        scaleSynapsisPos.domain([0,getMinMaxWeight(net, num_layers).max]);
        scaleSynapsisNeg.domain([0, Math.abs(getMinMaxWeight(net, num_layers).min)]);
        var zoom = d3.behavior.zoom()
            .scaleExtent([1, 20])
            .on("zoom", function(){
                svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
            });
        svg = svg
            .on("dblclick", function(){
                zoom.scale(1);
                zoom.translate([0, 0]);
                svg.transition().attr("transform", "translate(0,0)scale(1)");
            })
            .append("g").call(zoom).append("g");

        var layers = Array(num_layers);
        for(var i=0;i<num_layers;i++){
            layers[i] = getNeuronsAttr(net, num_layers, i);

        }
        for(i=0;i<num_layers;i++) {
            drawNeurons(svg, net, layers[i], i, i==0 ? true : undefined);
            drawSynapses(svg, net, layers[i], i, layers[i + 1]);
        }
        drawLegend(svg);
    };


    var transformDeepNetObject = function(deepnet){
        vars = deepnet["variables"];
        var layers = deepnet["layers"];
        var layout = {
            layer_0: {
                nneurons: vars.length+1
            }
        };
        var nodes = Number(layers[0]["Weights"]["row"]);
        for(var j=0;j<vars.length;j++){
            layout["layer_0"]["neuron_"+j]={
                nsynapses: nodes,
                weights: layers[0]["Weights"]["data"].slice(j*nodes, (j+1)*nodes)
            }
        }
        layout["layer_0"]["neuron_"+vars.length] = {
            nsynapses: Number(layers[0]["Weights"]["row"]),
            weights: layers[0]["Biases"]["data"]
        };
        vars.push("Bias node");
        for(var i=0; i<(layers.length-1);i++){
            layout["layer_"+(i+1)] = {
                nneurons: Number(layers[i]["Weights"]["row"])+1
            };
            var cnodes = Number(layers[i]["Weights"]["row"]);
            nodes = Number(layers[i+1]["Weights"]["row"]);
            for(var j=0; j<cnodes; j++){
                layout["layer_"+(i+1)]["neuron_"+j] = {
                    nsynapses: nodes,
                    weights: layers[i+1]["Weights"]["data"].slice(j*nodes, (j+1)*nodes)
                }
            }
            layout["layer_"+(i+1)]["neuron_"+cnodes] = {
                nsynapses: Number(layers[i]["Weights"]["row"]),
                weights: layers[i+1]["Biases"]["data"]
            }
        }
        layout["layer_"+(i+1)] = {
            nneurons: Number(layers[i]["Weights"]["row"])
        };
        for(var j=0; j<Number(layers[i]["Weights"]["row"]);j++){
            layout["layer_"+(i+1)]["neuron_"+j] = {
                nsynapses: 0
            }
        }
        layout["nlayers"] = i+2;
        var net = {
            variables: vars,
            layout: layout
        };
        return net;
    };

    var transformDeepNetObjectOld = function(deepnet){
        vars = deepnet["variables"];
        vars.push("Bias node");
        var layers = deepnet["layers"];
        var synapses = deepnet["synapses"]["synapses"];
        var layout = {
            layer_0: {
                nneurons: vars.length
            }
        };
        var nodes;
        for(var j=0;j<vars.length;j++){
            nodes = Number(layers[0]["Nodes"]);
            layout["layer_0"]["neuron_"+j]={
                nsynapses: nodes,
                weights: synapses.slice(j*nodes, (j+1)*nodes)
            }
        }
        for(var i=0; i<(layers.length-1);i++){
            layout["layer_"+(i+1)] = {
                nneurons: Number(layers[i]["Nodes"])
            };
            nodes = Number(layers[i+1]["Nodes"]);
            for(var j=0; j<=Number(layers[i]["Nodes"]);j++){
                layout["layer_"+(i+1)]["neuron_"+j] = {
                    nsynapses: nodes,
                    weights: synapses.slice(j*nodes, (j+1)*nodes)
                }
            }
        }
        layout["layer_"+(i+1)] = {
            nneurons: Number(layers[i]["Nodes"])
        };
        for(var j=0; j<Number(layers[i]["Nodes"]);j++){
            layout["layer_"+(i+1)]["neuron_"+j] = {
                nsynapses: 0
            }
        }
        layout["nlayers"] = i+2;
        var net = {
            variables: vars,
            layout: layout
        };
        return net;
    };

    var drawDeepNetNeurons = function (context,  neuronsattr, vars) {
        for(var i=0;i<neuronsattr.length;i++){
            context.beginPath();
            context.arc(neuronsattr[i].position.x+30, neuronsattr[i].position.y, neuronsattr[i].radius, 0, 2*Math.PI);
            context.fillStyle = style["neuron"]["colors"][neuronsattr[i].type];
            context.fill();
            context.closePath();
        }
        if (vars!==undefined){
            context.font = "16px bold Comic Sans MS";
            context.fillStyle = "#000";
            var text;
            for(var k=0;k<vars.length;k++){
                text = vars[k] + ":";
                context.fillText(text, neuronsattr[k].position.x+10-context.measureText(text).width, neuronsattr[k].position.y+5);
            }
        }
    };

    var drawDeepNetSynapses = function(ctx, net, layer1, layer1_index, layer2){
        var idx, si, d;
        for(idx in layer1) {
            var synapses = getSynapses(net, layer1_index, idx, layer1[idx].position, layer2);
            for(si in synapses){
                d = synapses[si];
                ctx.beginPath();
                ctx.moveTo(d.pos[0].x+30, d.pos[0].y);
                ctx.lineTo(d.pos[1].x+30, d.pos[1].y);
                ctx.lineWidth = d.type=="positive" ? scaleSynapsisPos(d.weight) : scaleSynapsisNeg(Math.abs(d.weight));
                ctx.strokeStyle = style["synapse"]["deepNet_colors"][d.type];
                ctx.stroke();
                ctx.closePath();
            }
        }
    };

    var drawDeepNetwork = function(context, net){
        var num_layers = Number(net["layout"]["nlayers"]);

        var layers = Array(num_layers);
        for(var i=0;i<num_layers;i++){
            layers[i] = getNeuronsAttr(net, num_layers, i);
        }
        for(i=0;i<num_layers;i++) {
            drawDeepNetSynapses(context, net, layers[i], i, layers[i + 1]);
            drawDeepNetNeurons(context, layers[i], i==0 ? net["variables"] : undefined);
        }
    };

    var drawDNNLabels = function(context){
        context.beginPath();
        context.fillStyle = style["legend"]["deepNet_colors"]["positive"];
        context.rect(canvas.width-170, 10, style["legend"]["rect"]["width"], style["legend"]["rect"]["height"]);
        context.fill();
        context.closePath();
        context.beginPath();
        context.fillStyle = style["legend"]["deepNet_colors"]["negative"];
        context.rect(canvas.width-170, 30, style["legend"]["rect"]["width"], style["legend"]["rect"]["height"]);
        context.fill();
        context.closePath();

        context.font = "16px bold Comic Sans MS";
        context.fillStyle = style["legend"]["deepNet_colors"]["positive"];
        context.fillText("Positive weight", canvas.width-150, 20);
        context.fillStyle = style["legend"]["deepNet_colors"]["negative"];
        context.fillText("Negative weight", canvas.width-150, 40);
    };


    NeuralNetwork.drawDeepNetwork = function (divid, netobj, oldStructure) {
        if (oldStructure===undefined) {
            oldStructure = false;
        }
        var div = d3.select("#"+divid);
        canvas = {
            width:  Number(div.property("style")["width"].replace("px","")),
            height: Number(div.property("style")["height"].replace("px",""))
        };

        if (oldStructure){
            net = transformDeepNetObjectOld(netobj);
        } else {
            net = transformDeepNetObject(netobj);
        }

        scaleSynapsisPos.range(style["synapse"]["deepNet_width_range"]);
        scaleSynapsisNeg.range(style["synapse"]["deepNet_width_range"]);

        var context = div.append("canvas")
            .attr("width", canvas.width+"px")
            .attr("height", canvas.height+"px")
            .call(d3.behavior.zoom().scaleExtent([1, 20]).on("zoom", function(){
                context.save();
                context.clearRect(0, 0, canvas.width, canvas.height);
                context.translate(d3.event.translate[0], d3.event.translate[1]);
                context.scale(d3.event.scale, d3.event.scale);
                drawDeepNetwork(context, net);
                drawDNNLabels(context);
                context.restore();
            }))
            .node().getContext("2d");

        drawDeepNetwork(context, net);
        drawDNNLabels(context);
    };

    Object.seal(NeuralNetwork);
    return NeuralNetwork;
}));