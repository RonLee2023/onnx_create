# onnx_create
概述
作用
本工具作用是根据配置生成onnx的网络（算子可以是自定义算子)，并保存至Onnx文件 中。 

便于调试程序，生成测试case等

可以生成单个算子，或一个网络片段，如下图所示。

                                                     

开发原因
制作测例时，需要手写Python，来配置网络。这样需要有onnx学习成本，并且手写容易出错。

使用方法
1 切换分支
代码地址 ：  compiler · main · Zelos_Chip / ka_toolchains · GitLab

使用分支br_v3.0_onnx_tool，代码位于tool目录下。



2.修改配置文件 
在tool/sample下，有两个文件 sparseconv_create.json，sparseconv_q.json 分别对应用单个算子和网络片段。

配置文件如图，其中attributes可以修改kernel,stride等属性，权重可以配置最大最小值 ，如果最大最小值相同则固定权值





3.配置说明 
配置字段说明如下，

强烈建议将例子中的配置复制修改



3.1 net_inputs,net_outputs
代表网络的输入/输出



node_name

输入/输出的node名
dtype	输入/输出的数据类型(TensorProto字符串）
shape	输入/输出的shape


3.2 node 




node_name	node名字
op_type	op类型，与onnx定义相同 ，支持自定义Op类型
input_nodes	输入的node名字的list，按照输入顺序
output_nodes	输出的node名字的list，按照输出顺序
input_infos	
输入数据的信息，包含数据类型(dtype字段)，形状（shape字段)

多个输入成list类型，需要与input_nodes对应

output_infos	
输出数据的信息，包含数据类型(dtype字段)，形状（shape字段)

多个输出成list类型，需要与output_nodes对应

weights	
权重信息。 可以配置随机的最大最小值，如果最大最小相同则为固定权重。

map类型，包含 name,dtype,shape,rand_min,rand_max字段

attributes	node的各个属性， 与onnx中的定义相同


联系方式： 441356547@qq.com
