/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "get_handle.hpp"

#include <miopen/miopen.h>
#include <miopen/datatype.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/tensorOp/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensorOp/solvers.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/visit_float.hpp>
#include "random.hpp"
#include <tensor_util.hpp>
#include <verify.hpp>

#include "tensorOp/tensor_op_helpers.hpp"

#include "perf_helper.hpp"
#include <miopen/float_equal.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string_view>

namespace miopen {
namespace unit_tests {

namespace
{

const std::string fakeId = {"testOCL"};

std::vector<std::vector<size_t>> tensorALensArr = {{32, 16, 8, 4, 4}, // tensor A
                                                   {16, 20, 16, 8},
                                                   {20, 16, 8},
                                                   {1, 16, 8},
                                                   {16, 8},
                                                   {8}};

std::vector<std::vector<size_t>> tensorBLensArr = {{32, 16, 8, 4, 4}, // tensor B
                                                   {32, 16, 1, 1, 1},
                                                   {1, 16, 8, 1, 1},
                                                   {1, 1, 8, 4, 1},
                                                   {16, 20, 16, 8},
                                                   {16, 20, 16, 1},
                                                   {16, 20, 1, 1},
                                                   {16, 1, 1, 1},
                                                   {1, 20, 16, 8},
                                                   {1, 20, 16, 1},
                                                   {1, 20, 1, 1},
                                                   {1, 1, 16, 8},
                                                   {1, 1, 1, 8},
                                                   {20, 16, 8},
                                                   {20, 16, 1},
                                                   {1, 16, 8},
                                                   {1, 16, 1},
                                                   {20, 1, 1},
                                                   {16, 8},
                                                   {16, 1},
                                                   {1, 8},
                                                   {8},
                                                   {1}};

std::vector<std::vector<int64_t>> offsetsArr = {
    {0, 0, 0}, {64, 32, 16}, {32, 16, 32}, {32, 16, 32}};

std::vector<std::vector<float>> alphabetaArr = {{1, 1, 0}, {-1, 1, 1}, {1.0, 0.5, 0.3}};

std::vector<std::vector<size_t>> stridesArr = {{8 * 16 * 20 * 16, 8 * 16 * 20, 8 * 16, 8, 1}};

std::vector<bool> packedArr = {true, false};

std::vector<miopenTensorOp_t> operationArr = {
    miopenTensorOpAdd, miopenTensorOpMul, miopenTensorOpMin, miopenTensorOpMax};

struct PerfTestData
{
    std::vector<size_t> lens;
    std::vector<size_t> strides;
};

std::vector<PerfTestData> GenPerfTensorDesc()
{
    // const size_t maxTotalSize = maxTotalSize * 1024ull * 1024ull / sizeof(T);
    const size_t maxTotalSize = 256; // NOLINT
    std::vector<PerfTestData> result;
    result.reserve(maxTotalSize);
    for(size_t N = 1; N <= maxTotalSize; N *= 2)
    {
        for(size_t C = 1; C <= maxTotalSize / N; C *= 2)
        {
            for(size_t H = 1; H <= maxTotalSize / (N * C); H *= 2)
            {
                for(size_t W = 1; W <= maxTotalSize / (N * C * H); W *= 2)
                {
                    size_t totalSize = N * C * H * W;
                    // Ensure the total size does not exceed the maximum limit
                    if(totalSize <= maxTotalSize)
                    {
                        result.push_back({std::vector<size_t>{N, C, H, W}, std::vector<size_t>{C * H * W, H * W, W, 1}});
                    }
                }
            }
        }
    }
    return result;
}

/// Custom OCL implementation for the solver, copy of the original one
struct TestOp4dTensorLiteOCL final : solver::tensorOp::TensorOpSolver
{
    const std::string& SolverDbId() const override { return fakeId; }

    bool IsApplicable([[maybe_unused]] const ExecutionContext& context,
                      const miopen::tensorOp::ProblemDescription& problem) const override
    {
        const auto& aTensorDesc = problem.GetATensorDesc();
        const auto& bTensorDesc = problem.GetBTensorDesc();
        const auto& cTensorDesc = problem.GetCTensorDesc();

        const auto& alens = aTensorDesc.GetLengths();
        const auto& blens = bTensorDesc.GetLengths();
        const auto& clens = cTensorDesc.GetLengths();

        auto asize = alens.size();

        if(asize == 4)
        {
            auto&& [num_wg, work_per_wg, bitmap] = solver::tensorOp::GetBitmapAndWgInfo(blens, clens);

            // quick fix for btensor = <1, 1, 1, 1>
            if(bTensorDesc.GetElementSize() == 1)
                bitmap = 4;

            bool fwd_conv_bias = (bitmap == (1 << 2));

            bool packed_tensor = true;
            packed_tensor &= aTensorDesc.IsPacked();
            packed_tensor &= bTensorDesc.IsPacked();
            packed_tensor &= cTensorDesc.IsPacked();

            bool packed_equal_tensor =
                packed_tensor && (bTensorDesc.GetElementSize() == cTensorDesc.GetElementSize());

            if(!fwd_conv_bias && packed_equal_tensor)
                return true;
        }

        return false;
    }

    solver::ConvSolution GetSolution([[maybe_unused]] const ExecutionContext& context,
                             const miopen::tensorOp::ProblemDescription& problem) const override
    {
        auto result = solver::ConvSolution{miopenStatusSuccess};

        const auto& bTensorDesc = problem.GetBTensorDesc();
        const auto& cTensorDesc = problem.GetCTensorDesc();

        miopenDataType_t data_type = bTensorDesc.GetType();

        auto&& [num_wg_orig, work_per_wg, incr_wg, bitmap, local_threads, global_threads] =
            solver::tensorOp::Get4dParams(problem, true);

        auto&& [RD_BLCK, READ_TYPE] =
            solver::tensorOp::GetRDBLCKandREADTYPE(cTensorDesc.GetElementSize(), bTensorDesc.GetType());

        size_t total_work = std::max(cTensorDesc.GetElementSize() / RD_BLCK, size_t(1));

        const std::array<size_t, 3> vld{local_threads, 1, 1};
        const std::array<size_t, 3> vgd{global_threads, 1, 1};

        KernelBuildParameters build_params = KernelBuildParameters{};

        solver::tensorOp::GetCommonParams(build_params, problem, false);

        build_params.Define("USE_4D_TENSOR_LITE");
        build_params.Define("RD_BLCK", std::to_string(RD_BLCK));
        build_params.Define("READ_TYPE", READ_TYPE);

        auto kernel = miopen::solver::KernelInfo{};

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});
        kernel.kernel_file  = "MIOpenTensorKernels.cl";
        kernel.kernel_name  = "Op4dTensorLite";

        using std::begin, std::end;

        kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
        kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

        result.invoker_factory = [data_type, total_work](const std::vector<Kernel> kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

                visit_float(data_type, [&](auto as_float) {
                    auto miopen_alpha0 = as_float(*(static_cast<const float*>(params.alpha0)));
                    auto miopen_alpha1 = as_float(*(static_cast<const float*>(params.alpha1)));
                    auto miopen_beta   = as_float(*(static_cast<const float*>(params.beta)));

                    kernel(params.ATensor,
                        params.BTensor,
                        params.CTensor,
                        miopen_alpha0,
                        miopen_alpha1,
                        miopen_beta,
                        static_cast<int64_t>(params.Aoffset),
                        static_cast<int64_t>(params.Boffset),
                        static_cast<int64_t>(params.Coffset),
                        static_cast<int64_t>(total_work),
                        static_cast<int>(!float_equal(miopen_beta, 0.0)));
                });
            };
        };
        result.construction_params.push_back(kernel);

        return result;
    }

    std::size_t
    GetWorkspaceSize([[maybe_unused]] const ExecutionContext& context,
                     [[maybe_unused]] const miopen::tensorOp::ProblemDescription& problem) const override
    {
        return 0;
    }

    bool MayNeedWorkspace() const override { return false; }
};

struct TestCase
{
    std::vector<size_t> tensorlens_ac;
    std::vector<size_t> tensorlens_b;
    std::vector<int64_t> offsets;
    std::vector<size_t> stride_a;
    std::vector<size_t> stride_b;
    std::vector<size_t> stride_c;
    std::vector<float> alphabeta;
    bool packed;
    miopenTensorOp_t operation;
};

bool checkTensorsCompatibility(const std::vector<size_t>& tensorALens,
                               const std::vector<size_t>& tensorBLens)
{
    if(tensorALens.size() != tensorBLens.size())
    {
        return false;
    }

    for(size_t idx = 0; idx < tensorBLens.size(); ++idx)
    {
        if((tensorBLens[idx] != 1) && (tensorALens[idx] != tensorBLens[idx]))
        {
            return false;
        }
    }

    return true;
}

void AddTestCases(std::vector<TestCase>& testCases,
                  const std::vector<size_t>& tensorALens,
                  const std::vector<size_t>& tensorBLens,
                  std::vector<size_t> const& stride_a,
                  std::vector<size_t> const& stride_b,
                  std::vector<size_t> const& stride_c)
{
    for(bool packed : packedArr)
        for(const auto& offsets : offsetsArr)
        {
            std::vector<int64_t> final_offsets{0, 0, 0};
            if(!packed)
            {
                if(std::any_of(offsets.begin(), offsets.end(), [](int64_t o) { return o < 0; }))
                    continue;

                final_offsets = offsets;
            }

            auto checkStride = [p = packed](const std::vector<size_t>& lens,
                                            const std::vector<size_t>& strides) {
                if(p)
                    return true;

                if(lens.size() > strides.size())
                    return false;

                // only sparsed case allowed, since all the kernels do not support the last
                // dimension strides
                if(strides.back() == 1)
                {
                    // we use float here for all types because strides are independent to type
                    auto packedStrides =
                        miopen::TensorDescriptor(miopen_type<float>{}, lens).GetStrides();

                    return std::equal(packedStrides.rbegin(),
                                      packedStrides.rend(),
                                      strides.rbegin(),
                                      [](size_t ps, size_t s) { return s >= ps; });
                }

                // currently tensor operations do not support non-one stride in the last dimention.
                return false;
            };

            if(!checkStride(tensorALens, stride_a))
                continue;
            if(!checkStride(tensorBLens, stride_b))
                continue;
            if(!checkStride(tensorALens, stride_c))
                continue;

            for(const auto& alphabeta : alphabetaArr)
                for(const auto& operation : operationArr)
                {
                    TestCase& testCase = testCases.emplace_back();

                    testCase.tensorlens_ac = tensorALens;
                    testCase.tensorlens_b  = tensorBLens;
                    testCase.alphabeta     = alphabeta;
                    testCase.offsets       = final_offsets;
                    testCase.packed        = packed;
                    testCase.operation     = operation;
                    testCase.stride_a      = stride_a;
                    testCase.stride_b      = stride_b;
                    testCase.stride_c      = stride_c;
                }
        }
}

std::vector<TestCase> GenCases()
{
    std::vector<TestCase> testCases;

    for(const auto& tensorALens : tensorALensArr)
        for(const auto& tensorBLens : tensorBLensArr)
        {
            if(!checkTensorsCompatibility(tensorALens, tensorBLens))
            {
                continue;
            }

            AddTestCases(testCases, tensorALens, tensorBLens, stridesArr[0], stridesArr[0], stridesArr[0]);
        }

    return testCases;
}

std::vector<TestCase> GenPerfCases()
{
    std::vector<TestCase> testCases{};
    for(auto& tensorADesc : GenPerfTensorDesc())
    {
        for(auto& tensorBDesc : GenPerfTensorDesc())
        {
            if(!checkTensorsCompatibility(tensorADesc.lens, tensorBDesc.lens))
            {
                continue;
            }

            AddTestCases(testCases, tensorADesc.lens, tensorBDesc.lens, tensorADesc.strides, tensorBDesc.strides, tensorADesc.strides);
        }
    }
    return testCases;
}

inline auto GetCases()
{
    static const auto cases = testing::ValuesIn(GenCases());
    return cases;
}

inline auto GetPerfCases()
{
    static const auto cases = testing::ValuesIn(GenPerfCases());
    return cases;
}
} // anonymous namespace

template<typename T>
struct Op4DTensorLiteTest
    : public ::testing::TestWithParam<TestCase>
{
protected:
    void SetUp() override
    {
        prng::reset_seed();
    }

    void CreateTensors()
    {
        const TestCase& testCase = GetParam();

        tensorA = CreateTensor(
            testCase.tensorlens_ac, testCase.stride_a, testCase.offsets[0], testCase.packed);
        tensorB = CreateTensor(
            testCase.tensorlens_b, testCase.stride_b, testCase.offsets[1], testCase.packed);
        tensorC = CreateTensor(
            testCase.tensorlens_ac, testCase.stride_c, testCase.offsets[2], testCase.packed);
    }

    tensor<T> CreateTensor(const std::vector<size_t>& lens,
                           const std::vector<size_t>& strides,
                           int64_t offset,
                           bool isPacked)
    {
        uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

        if(!isPacked)
        {
            std::vector<size_t> real_strides(strides.begin() + (strides.size() - lens.size()),
                                             strides.end());
            auto r = tensor<T>{lens, real_strides}.generate(tensor_elem_gen_integer{max_value});
            r.data.resize(r.data.size() + offset);
            return r;
        }
        else
        {
            return tensor<T>{lens}.generate(tensor_elem_gen_integer{max_value});
        }
    }

    void Run()
    {
        CreateTensors();

        std::vector<T> tensorGPUOCLData = runOCL();
        std::vector<T> tensorGPUHIPData = runHIP();

        CompareResults(tensorGPUOCLData, tensorGPUHIPData);
    }

    void PerfRun()
    {
        CreateTensors();

        std::vector<T> tensorGPUOCLData = runOCL(true); 
        std::vector<T> tensorGPUHIPData = runHIP(true);
    }

    std::vector<T> runOCL(bool is_perf = false)
    {
        // preparing test solver as in OpTensor function
        const TestCase& testCase = GetParam();

        auto&& handle = get_handle();

        auto a_dev = handle.Write(tensorA.data);
        auto b_dev = handle.Write(tensorB.data);
        auto c_dev = handle.Write(tensorC.data);

        auto* alpha0 = &testCase.alphabeta[0];
        auto* alpha1 = &testCase.alphabeta[1];
        auto* beta = &testCase.alphabeta[2];

        const auto Aoffset = testCase.offsets[0];
        const auto Boffset = testCase.offsets[1];
        const auto Coffset = testCase.offsets[2];

        const auto problem = tensorOp::ProblemDescription{
        testCase.operation, beta, tensorA.desc, tensorB.desc, tensorC.desc, false};

        if(!TestOp4dTensorLiteOCL{}.IsApplicable(&handle, problem))
            return {};

        const auto invoke_params = tensorOp::InvokeParams{
            alpha0, a_dev.get(), alpha1, b_dev.get(), beta, c_dev.get(), Aoffset, Boffset, Coffset};

        const auto algo    = AlgorithmName{"TestTensorOpSolver"};
        const auto solvers = solver::SolverContainer<TestOp4dTensorLiteOCL>{};

        if(is_perf)
        {
            auto callback = [&algo, &solvers, &invoke_params, &handle, &problem](std::vector<T>& elapsedTime_ms) -> void
            {
                handle.EnableProfiling();
                handle.ResetKernelTime();
                for(auto i = 0; i < NUM_PERF_RUNS + NUM_WARMUP_RUNS; i++)
                {
                    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
                    if(i >= NUM_WARMUP_RUNS)
                        elapsedTime_ms.push_back(static_cast<T>(handle.GetKernelTime()));
                    handle.ResetKernelTime();
                }
            };

            ph.perfTest(callback, "Tensor4DOpsLiteSolver");
        }
        else
            solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

        auto result = handle.Read<T>(c_dev, tensorC.data.size());

        return result;
    }

    std::vector<T> runHIP(bool is_perf = false)
    {
        const TestCase& testCase = GetParam();

        auto&& handle = get_handle();

        auto a_dev = handle.Write(tensorA.data);
        auto b_dev = handle.Write(tensorB.data);
        auto c_dev = handle.Write(tensorC.data);

        const auto problem = tensorOp::ProblemDescription{
        testCase.operation, &testCase.alphabeta[2], tensorA.desc, tensorB.desc, tensorC.desc, false};

        if(!TestOp4dTensorLiteOCL{}.IsApplicable(&handle, problem))
            return {};

        if(is_perf)
        {
            auto callback = [this, &handle, &testCase, &a_dev, &b_dev, &c_dev](std::vector<T>& elapsedTime_ms) -> void
            {
                handle.EnableProfiling();
                handle.ResetKernelTime();
                for(auto i = 0; i < NUM_PERF_RUNS + NUM_WARMUP_RUNS; i++)
                {
                    miopen::OpTensor(handle,
                            testCase.operation,
                            &testCase.alphabeta[0],
                            tensorA.desc,
                            a_dev.get(),
                            &testCase.alphabeta[1],
                            tensorB.desc,
                            b_dev.get(),
                            &testCase.alphabeta[2],
                            tensorC.desc,
                            c_dev.get(),
                            testCase.offsets[0],
                            testCase.offsets[1],
                            testCase.offsets[2],
                            false);
                    if(i >= NUM_WARMUP_RUNS)
                        elapsedTime_ms.push_back(static_cast<T>(handle.GetKernelTime()));
                    handle.ResetKernelTime();
                }
            };

            ph.perfTest(callback, "Tensor4DOpsLiteSolver");
        }
        else
            miopen::OpTensor(handle,
                            testCase.operation,
                            &testCase.alphabeta[0],
                            tensorA.desc,
                            a_dev.get(),
                            &testCase.alphabeta[1],
                            tensorB.desc,
                            b_dev.get(),
                            &testCase.alphabeta[2],
                            tensorC.desc,
                            c_dev.get(),
                            testCase.offsets[0],
                            testCase.offsets[1],
                            testCase.offsets[2],
                            false); // it does not verify non-standard behaviour
        auto result = handle.Read<T>(c_dev, tensorC.data.size());

        return result;
    }

    void WritePerfResults(std::string_view filename)
    {
        const TestCase& testCase = GetParam();
        std::stringstream stats{};
        miopenDataType_t data_type = miopen_type<T>{};
        stats << "_aclens_" << std::to_string(testCase.tensorlens_ac[0]) << "_" <<
                    std::to_string(testCase.tensorlens_ac[1]) << "_" <<
                    std::to_string(testCase.tensorlens_ac[2]) << "_" <<
                    std::to_string(testCase.tensorlens_ac[3]) << "_acstrides_" <<
                    std::to_string(testCase.stride_a[0]) << "_" <<
                    std::to_string(testCase.stride_a[1]) << "_" <<
                    std::to_string(testCase.stride_a[2]) << "_" <<
                    std::to_string(testCase.stride_a[3]);
        stats << "_blens_" + std::to_string(testCase.tensorlens_b[0]) << "_" <<
                    std::to_string(testCase.tensorlens_b[1]) << "_" <<
                    std::to_string(testCase.tensorlens_b[2]) << "_" <<
                    std::to_string(testCase.tensorlens_b[3]) << "_bstrides_" <<
                    std::to_string(testCase.stride_b[0]) << "_" <<
                    std::to_string(testCase.stride_b[1]) << "_" <<
                    std::to_string(testCase.stride_b[2]) << "_" <<
                    std::to_string(testCase.stride_b[3]);
        stats << "_alpha0_" << std::to_string(testCase.alphabeta[0]) << "_alpha1_" << std::to_string(testCase.alphabeta[1]) <<
                    "_beta_" << std::to_string(testCase.alphabeta[2]) << "_" << miopen::GetDataType(data_type);

        ph.writeStatsToCSV(std::string{filename}, stats.str());
    }

    void TearDown() override
    {
        WritePerfResults("test_4d_lite.csv");
    }

    void CompareResults(const std::vector<T>& tensorLHSData, const std::vector<T>& tensorRHSData)
    {
        const TestCase& testCase = GetParam();

        double tolerance = 1;

        if(std::is_same_v<T, half_float::half>)
        {
            // taken from original c-test
            tolerance = 80;
        }

        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        double error     = miopen::rms_range(tensorRHSData, tensorLHSData);

        ASSERT_LE(error, threshold)
            << "TensorOp: " << testCase.operation << std::endl
            << "A tensor: " << tensorA.desc.ToString() << std::endl
            << "B tensor: " << tensorB.desc.ToString() << std::endl
            << "IsPacked: " << testCase.packed << std::endl
            << "Offsets: " << testCase.offsets[0] << "," << testCase.offsets[1] << ","
            << testCase.offsets[2] << std::endl;
    }

private:
    tensor<T> tensorA;
    tensor<T> tensorB;
    tensor<T> tensorC;

    PerfHelper<T> ph;
};

template<typename T>
struct Op4DTensorLitePerfTest : Op4DTensorLiteTest<T>
{
};

using GPU_Op4dTensorLiteTest_FP32 = Op4DTensorLiteTest<float>;
using GPU_Op4dTensorLiteTest_FP16 = Op4DTensorLiteTest<half_float::half>;
using GPU_Op4dTensorLiteTest_FP64 = Op4DTensorLiteTest<double>;

using GPU_Op4dTensorLitePerfTest_FP32 = Op4DTensorLitePerfTest<float>;
using GPU_Op4dTensorLitePerfTest_FP16 = Op4DTensorLitePerfTest<half_float::half>;
using GPU_Op4dTensorLitePerfTest_FP64 = Op4DTensorLitePerfTest<double>;

TEST_P(GPU_Op4dTensorLiteTest_FP32, PortTest32)
{
    Run();
}

TEST_P(GPU_Op4dTensorLiteTest_FP16, PortTest16)
{
    Run();
}

TEST_P(GPU_Op4dTensorLiteTest_FP64, PortTest64)
{
    Run();
}

TEST_P(GPU_Op4dTensorLitePerfTest_FP32, PortPerfTest32)
{
    PerfRun();
}

TEST_P(GPU_Op4dTensorLitePerfTest_FP16, PortPerfTest16)
{
    PerfRun();
}

TEST_P(GPU_Op4dTensorLitePerfTest_FP64, PortPerfTest64)
{
    PerfRun();
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Op4dTensorLiteTest_FP32, GetCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Op4dTensorLiteTest_FP16, GetCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Op4dTensorLiteTest_FP64, GetCases());

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Op4dTensorLitePerfTest_FP32, GetPerfCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Op4dTensorLitePerfTest_FP16, GetPerfCases());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Op4dTensorLitePerfTest_FP64, GetPerfCases());

} // namespace unit_tests
} // namespace miopen
