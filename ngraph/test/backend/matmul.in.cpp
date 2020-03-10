//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x0_0x2)
{
    Shape shape_a{2, 0};
    Shape shape_b{0, 2};
    Shape shape_r{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 0, 0}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_0x2_2x0)
{
    Shape shape_a{0, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{0, 0};
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3x2_2x0)
{
    Shape shape_a{3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 0};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{3, 0};
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x2_2x2)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::MatMul>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{19, 22, 43, 50}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x3_3x3)
{
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    copy_data(b, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  vector<float>{30.f, 36.f, 42.f, 66.f, 81.f, 96.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_2x3_3x3_int64)
{
    Shape shape_in1{2, 3};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::i64, shape_in1);
    auto B = make_shared<op::Parameter>(element::i64, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, false, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::i64, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::i64, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::i64, shape_out);

    copy_data(a, vector<int64_t>{1, 2, 3, 4, 5, 6});
    copy_data(b, vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(
        test::all_close(read_vector<int64_t>(result), vector<int64_t>{30, 36, 42, 66, 81, 96}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3x2_3x3_transpose)
{
    Shape shape_in1{3, 2};
    Shape shape_in2{3, 3};
    Shape shape_out{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, true, false);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    copy_data(b, vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  vector<float>{30.f, 36.f, 42.f, 66.f, 81.f, 96.f}));
}

NGRAPH_TEST(${BACKEND_NAME}, matmul_3x2_2x3_transpose)
{
    Shape shape_in1{3, 2};
    Shape shape_in2{2, 3};
    Shape shape_out{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_in1);
    auto B = make_shared<op::Parameter>(element::f32, shape_in2);
    auto matmul = make_shared<op::MatMul>(A, B, true, true);
    auto f = make_shared<Function>(matmul, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape_in1);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape_in2);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape_out);

    copy_data(a, vector<float>{1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    copy_data(b, vector<float>{1.f, 3.f, 5.f, 2.f, 4.f, 6.f});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});

    EXPECT_TRUE(
        test::all_close_f(read_vector<float>(result), vector<float>{22.f, 28.f, 49.f, 64.f}));
}
