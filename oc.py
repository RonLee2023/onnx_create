# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Compile onnx graph.
"""
import argparse
import json
import os
import onnx
import onnx.numpy_helper
from onnx import helper 
from onnx import TensorProto 
import numpy as np
import onnx.numpy_helper
from onnxnode import LayerNode
from onnxnode_manager import OnnxNodeManager
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        default="samples/sparseconv_modify.json",
        help="",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="output",
        help="Define path to place output production",
    )
    args = parser.parse_args()
    args.config = os.path.abspath(args.config)
    args.output = os.path.abspath(args.output)
    return args


def main():
    args = get_args()
    model = None
    with open(args.config) as fp:
        all_nodes = json.load(fp)
        model_nodes = []
        model_weights = []
        for node in all_nodes:
            if "net_inputs" in node:
                for oneinput in node["net_inputs"]:
                    OnnxNodeManager.create_io_node(oneinput["node_name"], "net_inputs", oneinput["dtype"], oneinput["shape"])
            elif "net_outputs" in node:
                for oneinput in node["net_outputs"]:
                    OnnxNodeManager.create_io_node(oneinput["node_name"], "net_outputs", oneinput["dtype"], oneinput["shape"])
            else:
                #create tensor value info between 2 nodes
                OnnxNodeManager.create_node(node["node_name"], node["op_type"], 
                                            node["input_nodes"], node["input_infos"],
                                            node["output_nodes"], node["output_infos"])

        for config_data in all_nodes:    
            node_name = config_data["node_name"] if "node_name" in config_data else None
            if node_name == None:
                if "net_inputs" in config_data or "net_outputs" in config_data:
                    continue
                else:
                    print("Find node has no node name!")

            layer_node = OnnxNodeManager.get_layer_node(node_name)
            OnnxNodeManager.add_input_output(layer_node, 
                                             config_data["input_nodes"] if "input_nodes" in config_data else None, 
                                             config_data["output_nodes"] if "output_nodes" in config_data else None)

            if "weights" in config_data:
                weights = layer_node.add_weights(config_data["weights"])
                model_weights.extend(weights)

            layer_node.set_attribute(config_data["attributes"])
            model_nodes.append(layer_node.onnx_node)

    graph_new = helper.make_graph(model_nodes, 'linear_func', OnnxNodeManager.inputs, OnnxNodeManager.outputs) 
    for w in model_weights:
        graph_new.initializer.append(w)
    model_new = helper.make_model(graph_new) 

    #print(model_new)
    #for t in model_new.graph.initializer:
    #    print(t)

    onnx.save(model_new, args.output) 
    print("onnx file has save to ", args.output)

if __name__ == "__main__":
    main()
