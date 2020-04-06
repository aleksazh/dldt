#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_blob.h>
#include <ie_layers.h>

#include <details/ie_exception.hpp>
#include <ie_parameter.hpp>
#include <string>
#include <tuple>
#include <vector>

using namespace InferenceEngine;

//
// details/ie_exception.hpp
//
details::InferenceEngineException::~InferenceEngineException() noexcept {}
//
// ie_layers.h
//
CNNLayer::~CNNLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  CNNLayer::~CNNLayer() {" << std::endl;}
WeightableLayer::~WeightableLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  WeightableLayer::~WeightableLayer() {" << std::endl;}
ConvolutionLayer::~ConvolutionLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ConvolutionLayer::~ConvolutionLayer() {" << std::endl;}
DeconvolutionLayer::~DeconvolutionLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  DeconvolutionLayer::~DeconvolutionLayer() {" << std::endl;}
DeformableConvolutionLayer::~DeformableConvolutionLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  DeformableConvolutionLayer::~DeformableConvolutionLayer() {" << std::endl;}
PoolingLayer::~PoolingLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  PoolingLayer::~PoolingLayer() {" << std::endl;}
BinaryConvolutionLayer::~BinaryConvolutionLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  BinaryConvolutionLayer::~BinaryConvolutionLayer() {" << std::endl;}
FullyConnectedLayer::~FullyConnectedLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  FullyConnectedLayer::~FullyConnectedLayer() {" << std::endl;}
ConcatLayer::~ConcatLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ConcatLayer::~ConcatLayer() {" << std::endl;}
SplitLayer::~SplitLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  SplitLayer::~SplitLayer() {" << std::endl;}
NormLayer::~NormLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  NormLayer::~NormLayer() {" << std::endl;}
SoftMaxLayer::~SoftMaxLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  SoftMaxLayer::~SoftMaxLayer() {" << std::endl;}
GRNLayer::~GRNLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  GRNLayer::~GRNLayer() {" << std::endl;}
MVNLayer::~MVNLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  MVNLayer::~MVNLayer() {" << std::endl;}
ReLULayer::~ReLULayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ReLULayer::~ReLULayer() {" << std::endl;}
ClampLayer::~ClampLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ClampLayer::~ClampLayer() {" << std::endl;}
ReLU6Layer::~ReLU6Layer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ReLU6Layer::~ReLU6Layer() {" << std::endl;}
EltwiseLayer::~EltwiseLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  EltwiseLayer::~EltwiseLayer() {" << std::endl;}
CropLayer::~CropLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  CropLayer::~CropLayer() {" << std::endl;}
ReshapeLayer::~ReshapeLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ReshapeLayer::~ReshapeLayer() {" << std::endl;}
TileLayer::~TileLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  TileLayer::~TileLayer() {" << std::endl;}
ScaleShiftLayer::~ScaleShiftLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ScaleShiftLayer::~ScaleShiftLayer() {" << std::endl;}
TensorIterator::~TensorIterator() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  TensorIterator::~TensorIterator() {" << std::endl;}
RNNCellBase::~RNNCellBase() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  RNNCellBase::~RNNCellBase() {" << std::endl;}
LSTMCell::~LSTMCell() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  LSTMCell::~LSTMCell() {" << std::endl;}
GRUCell::~GRUCell() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  GRUCell::~GRUCell() {" << std::endl;}
RNNCell::~RNNCell() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  RNNCell::~RNNCell() {" << std::endl;}
RNNSequenceLayer::~RNNSequenceLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  RNNSequenceLayer::~RNNSequenceLayer() {" << std::endl;}
PReLULayer::~PReLULayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  PReLULayer::~PReLULayer() {" << std::endl;}
PowerLayer::~PowerLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  PowerLayer::~PowerLayer() {" << std::endl;}
BatchNormalizationLayer::~BatchNormalizationLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  BatchNormalizationLayer::~BatchNormalizationLayer() {" << std::endl;}
GemmLayer::~GemmLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  GemmLayer::~GemmLayer() {" << std::endl;}
PadLayer::~PadLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  PadLayer::~PadLayer() {" << std::endl;}
GatherLayer::~GatherLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  GatherLayer::~GatherLayer() {" << std::endl;}
StridedSliceLayer::~StridedSliceLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  StridedSliceLayer::~StridedSliceLayer() {" << std::endl;}
ShuffleChannelsLayer::~ShuffleChannelsLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ShuffleChannelsLayer::~ShuffleChannelsLayer() {" << std::endl;}
DepthToSpaceLayer::~DepthToSpaceLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  DepthToSpaceLayer::~DepthToSpaceLayer() {" << std::endl;}
SpaceToDepthLayer::~SpaceToDepthLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  SpaceToDepthLayer::~SpaceToDepthLayer() {" << std::endl;}
SparseFillEmptyRowsLayer::~SparseFillEmptyRowsLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  SparseFillEmptyRowsLayer::~SparseFillEmptyRowsLayer() {" << std::endl;}
SparseSegmentReduceLayer::~SparseSegmentReduceLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  SparseSegmentReduceLayer::~SparseSegmentReduceLayer() {" << std::endl;}
ExperimentalSparseWeightedReduceLayer::~ExperimentalSparseWeightedReduceLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ExperimentalSparseWeightedReduceLayer::~ExperimentalSparseWeightedReduceLayer() {" << std::endl;}
SparseToDenseLayer::~SparseToDenseLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  SparseToDenseLayer::~SparseToDenseLayer() {" << std::endl;}
BucketizeLayer::~BucketizeLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  BucketizeLayer::~BucketizeLayer() {" << std::endl;}
ReverseSequenceLayer::~ReverseSequenceLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ReverseSequenceLayer::~ReverseSequenceLayer() {" << std::endl;}
OneHotLayer::~OneHotLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  OneHotLayer::~OneHotLayer() {" << std::endl;}
RangeLayer::~RangeLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  RangeLayer::~RangeLayer() {" << std::endl;}
FillLayer::~FillLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  FillLayer::~FillLayer() {" << std::endl;}
SelectLayer::~SelectLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  SelectLayer::~SelectLayer() {" << std::endl;}
BroadcastLayer::~BroadcastLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  BroadcastLayer::~BroadcastLayer() {" << std::endl;}
QuantizeLayer::~QuantizeLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  QuantizeLayer::~QuantizeLayer() {" << std::endl;}
MathLayer::~MathLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  MathLayer::~MathLayer() {" << std::endl;}
ReduceLayer::~ReduceLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ReduceLayer::~ReduceLayer() {" << std::endl;}
TopKLayer::~TopKLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  TopKLayer::~TopKLayer() {" << std::endl;}
UniqueLayer::~UniqueLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  UniqueLayer::~UniqueLayer() {" << std::endl;}
NonMaxSuppressionLayer::~NonMaxSuppressionLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  NonMaxSuppressionLayer::~NonMaxSuppressionLayer() {" << std::endl;}
ScatterLayer::~ScatterLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  ScatterLayer::~ScatterLayer() {" << std::endl;}
//
// ie_parameter.hpp
//
Parameter::~Parameter() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  Parameter::~Parameter() {" << std::endl;
    clear();
}

#ifdef __clang__
Parameter::Any::~Any() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  Parameter::Any::~Any() {" << std::endl;}

template struct InferenceEngine::Parameter::RealData<int>;
template struct InferenceEngine::Parameter::RealData<bool>;
template struct InferenceEngine::Parameter::RealData<float>;
template struct InferenceEngine::Parameter::RealData<uint32_t>;
template struct InferenceEngine::Parameter::RealData<std::string>;
template struct InferenceEngine::Parameter::RealData<unsigned long>;
template struct InferenceEngine::Parameter::RealData<std::vector<int>>;
template struct InferenceEngine::Parameter::RealData<std::vector<std::string>>;
template struct InferenceEngine::Parameter::RealData<std::vector<unsigned long>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int, unsigned int>>;
#endif  // __clang__
//
// ie_blob.h
//
Blob::~Blob() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  Blob::~Blob() {" << std::endl;}

MemoryBlob::~MemoryBlob() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  MemoryBlob::~MemoryBlob() {" << std::endl;}

#ifdef __clang__
template <typename T, typename U>
TBlob<T, U>::~TBlob() {
    std::cerr << "./inference-engine/src/inference_engine/ie_rtti.cpp:  TBlob<T, U>::~TBlob() {" << std::endl;
    free();
}

template class InferenceEngine::TBlob<float>;
template class InferenceEngine::TBlob<double>;
template class InferenceEngine::TBlob<int16_t>;
template class InferenceEngine::TBlob<uint16_t>;
template class InferenceEngine::TBlob<int8_t>;
template class InferenceEngine::TBlob<uint8_t>;
template class InferenceEngine::TBlob<int>;
template class InferenceEngine::TBlob<long>;
template class InferenceEngine::TBlob<long long>;
#endif  // __clang__
