/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma message "hello_world_test.cc"
extern "C"
{
#include "fastest_det_test.h"
#include <math.h>
#include <stdio.h>
}

#include "models/fastest_det.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace
{
constexpr int kTensorArenaSize = 0x70000;
uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model *model = nullptr;

using FastestDetOpResolver = tflite::MicroMutableOpResolver<11>;

TfLiteStatus RegisterOps(FastestDetOpResolver &op_resolver)
{
    TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
    TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
    TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
    TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddGather());
    TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    return kTfLiteOk;
}
} // namespace

TfLiteStatus load_model()
{
    if (model != nullptr) { return kTfLiteOk; }
    model = ::tflite::GetModel(model_data);
    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteOk;
}

TfLiteStatus Infer(const char *data, size_t len, int8_t **out, size_t *out_len)
{
    if (model == nullptr) { return kTfLiteError; }
    FastestDetOpResolver op_resolver;
    TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

    tflite::MicroInterpreter interpreter(
        model, op_resolver, tensor_arena, kTensorArenaSize);
    TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
    TfLiteTensor *input = interpreter.input(0);
    TFLITE_CHECK_NE(input, nullptr);
    TfLiteTensor *output = interpreter.output(0);
    TFLITE_CHECK_NE(output, nullptr);
    memcpy(input->data.int8, data, len);
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    *out = output->data.int8;
    *out_len = output->bytes;

    return kTfLiteOk;
}

extern "C" int init_tflite()
{
    tflite::InitializeTarget();
    TF_LITE_ENSURE_STATUS(load_model());
    return kTfLiteOk;
}

extern "C" int
infer(const char *data, size_t len, int8_t **out, size_t *out_len)
{
    return Infer(data, len, out, out_len);
}
