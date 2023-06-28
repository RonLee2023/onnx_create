import json
import os
import onnx
import onnx.numpy_helper
from onnx import helper 
from onnx import TensorProto 
import numpy as np
import onnx.numpy_helper
from onnxnode import LayerNode
from onnxnode import LayerNodeInfo

class _OnnxNodeManager:
    def __init__(self) -> None:
        self._nodes = {}
        self._inputs = []
        self._outputs = []
        self.tensor_value_infos = {}
    
    def create_node(self, node_name, op_type, input_nodes:json, input_infos:json, output_nodes:json, output_infos:json) -> None:
        layer_node = LayerNode(node_name, op_type, None, None)
        if node_name in self._nodes.keys():
            print("ERROR: The name is repeated",node_name)

        for idx, input in enumerate(input_infos):
            layer_node.input_infos.append(LayerNodeInfo(input_nodes[idx], input))
            layer_node.inputs.append(input_nodes[idx])
        for idx, output in enumerate(output_infos):
            layer_node.output_infos.append(LayerNodeInfo(output_nodes[idx], output))
            layer_node.outputs.append(output_nodes[idx])

        self._nodes[node_name] = layer_node


    def create_io_node(self, node_name, op_type, dtype, shape) -> None:
        layer_node = LayerNode(node_name, op_type, dtype, shape)
        if node_name in self._nodes.keys():
            print("ERROR: The name is repeated",node_name)  
        self._nodes[node_name] = layer_node
        if op_type == "net_inputs":
            self._inputs.append(layer_node.onnx_node)
        elif op_type == "net_outputs":
            self._outputs.append(layer_node.onnx_node)
        else:
            print("Unkonwn io op type")

    def get_layer_node(self, node_name):
        return self._nodes[node_name]
    def get_onnx_node(self, node_name) :
        return self._nodes[node_name].onnx_node
    
    @property
    def inputs(self):
        return self._inputs
    @property
    def outputs(self):
        return self._outputs
    
    def get_tensor_value_info(self, pre_layer_node:LayerNode, pre_idx, cur_layer_node:LayerNode, cur_idx) :
        if not pre_layer_node or not cur_layer_node:
            return None

        tvi_name = pre_layer_node.name + "_" + str(pre_idx) + "_"+cur_layer_node.name + "_" + str(cur_idx)
        if tvi_name in self.tensor_value_infos:
            return self.tensor_value_infos[tvi_name]
        else:
            assert pre_layer_node.output_infos[pre_idx].dtype == cur_layer_node.input_infos[cur_idx].dtype
            assert pre_layer_node.output_infos[pre_idx].shape == cur_layer_node.input_infos[cur_idx].shape

            tvi = helper.make_tensor_value_info(tvi_name, 
                                                pre_layer_node.str2tensorprototype(pre_layer_node.output_infos[pre_idx].dtype), 
                                                pre_layer_node.output_infos[pre_idx].shape) 
            self.tensor_value_infos[tvi_name] = tvi
            return self.tensor_value_infos[tvi_name]


    def add_input_output(self, layer_node:LayerNode, inputs_json, outputs_json) ->LayerNode:

        if inputs_json != None and len(inputs_json) != 0:
            for idx, input_name in enumerate(inputs_json):
                pre_layer_node = self.get_layer_node(input_name)
                if pre_layer_node.is_ionode:
                    layer_node.onnx_node.input.append(input_name)
                else:
                    pre_idx = pre_layer_node.get_output_idx_by_name(layer_node.name)
                    tvi = self.get_tensor_value_info(pre_layer_node, pre_idx, layer_node, idx)
                    layer_node.onnx_node.input.append(tvi.name)
            
        if outputs_json != None and len(outputs_json) != 0:
            for idx, output_name in enumerate(outputs_json):
                next_layer_node = self.get_layer_node(output_name)
                if next_layer_node.is_ionode:
                    layer_node.onnx_node.output.append(output_name)
                else:
                    next_idx = next_layer_node.get_input_idx_by_name(layer_node.name)
                    tvi = self.get_tensor_value_info(layer_node, idx, next_layer_node, next_idx)
                    layer_node.onnx_node.output.append(tvi.name)


OnnxNodeManager = _OnnxNodeManager()