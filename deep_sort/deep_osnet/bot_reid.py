# !/usr/local/bin/python3
import os
import numpy as np
from PIL import Image
import tensorrt as trt

from .common import do_inference, allocate_buffers


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class BotReid():
    def __init__(self, model_path, size, gpu_id=0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(TRT_LOGGER)
        self.network = self.builder.create_network()
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)
        self.size = size

        self.engine = load_engine(model_path)

        '''
        with open(model_path, 'rb') as model:
            print('Beginning ONNX file parsing:%s'%(model_path))
            self.parser.parse(model.read())

        for i in range(0,self.parser.num_errors):
            print(self.parser.get_error(i))
        
        for i in range(0,self.network.num_layers):
            layer=self.network.get_layer(i)
            print(layer.name,layer.type,layer.num_inputs,layer.num_outputs)

        self.engine = self.builder.build_cuda_engine(self.network)
        outEngineName=model_path[:-4]+'trt'
        with open(outEngineName, "wb") as f:
                f.write(self.engine.serialize())
        '''

        '''
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                runtime.deserialize_cuda_engine(f.read())
        else:
            build_engine()
        '''

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        # [0.485, 0.456, 0.406]
        self.mean = np.ones((3, self.size[0], self.size[1]), dtype=np.float32)
        self.mean[0] = self.mean[0] * 0.485
        self.mean[1] = self.mean[1] * 0.456
        self.mean[2] = self.mean[2] * 0.406

        # [0.229, 0.224, 0.225]
        self.std = np.ones((3, self.size[0], self.size[1]), dtype=np.float32)
        self.std[0] = self.std[0] * 0.229
        self.std[1] = self.std[1] * 0.224
        self.std[2] = self.std[2] * 0.225

    def infer(self, image):
        img = image.transpose(2, 0, 1) / 255.0
        img = img - self.mean
        img = img / self.std
        np.copyto(self.inputs[0].host, img.ravel())
        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                   stream=self.stream)

        return trt_outputs
