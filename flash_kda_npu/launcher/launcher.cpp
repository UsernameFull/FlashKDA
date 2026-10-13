// Host-side launcher: loads the extracted device binary and launches kernels on
// the current torch_npu stream via aclrtLaunchKernel.
#include <acl/acl_rt.h>
#include <acl/acl_rt_compile.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

struct KernelLib {
    aclrtBinHandle bin = nullptr;
    std::unordered_map<std::string, aclrtFuncHandle> funcs;
};

KernelLib g_lib;
aclrtContext g_ctx = nullptr;
std::mutex g_mu;
void *g_argsBuf = nullptr;
size_t g_argsCap = 0;

void Check(aclError err, const std::string &what) {
    if (err != ACL_SUCCESS) {
        throw std::runtime_error(what + " failed, aclError=" + std::to_string(err));
    }
}

void EnsureRt() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_ctx != nullptr) {
        return;
    }
    aclrtContext cur = nullptr;
    aclError e = aclrtGetCurrentContext(&cur);
    if (e != ACL_SUCCESS || cur == nullptr) {
        e = aclInit(nullptr);
        Check(e, "aclInit");
        e = aclrtSetDevice(0);
        Check(e, "aclrtSetDevice");
        e = aclrtCreateContext(&g_ctx, 0);
        Check(e, "aclrtCreateContext");
        e = aclrtSetCurrentContext(g_ctx);
        Check(e, "aclrtSetCurrentContext");
    } else {
        g_ctx = cur;
    }
}

void *AllocArgs(size_t size) {
    if (g_argsCap < size) {
        if (g_argsBuf != nullptr) {
            aclrtFree(g_argsBuf);
        }
        Check(aclrtMalloc(&g_argsBuf, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc args");
        g_argsCap = size;
    }
    return g_argsBuf;
}

void Launch(const std::string &name, uint32_t numBlocks, const void *args, size_t argsSize,
            uint64_t stream) {
    EnsureRt();
    auto it = g_lib.funcs.find(name);
    if (it == g_lib.funcs.end()) {
        throw std::runtime_error("kernel not loaded: " + name);
    }
    void *devArgs = AllocArgs(argsSize);
    // `args` points at host memory (stack structs / std::vector data), so this is H2D.
    Check(aclrtMemcpy(devArgs, argsSize, args, argsSize, ACL_MEMCPY_HOST_TO_DEVICE),
          "aclrtMemcpy args");
    aclError e = aclrtLaunchKernel(it->second, numBlocks, devArgs, argsSize,
                                   reinterpret_cast<aclrtStream>(stream));
    Check(e, "aclrtLaunchKernel " + name);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "load_kernel_binary",
        [](const std::string &path) {
            EnsureRt();
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) {
                throw std::runtime_error("cannot open " + path);
            }
            std::streamsize n = f.tellg();
            f.seekg(0);
            std::vector<char> buf(n);
            f.read(buf.data(), n);
            aclrtBinaryLoadOption opt;
            opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
            opt.value.magic = 1;
            aclrtBinaryLoadOptions opts;
            opts.options = &opt;
            opts.numOpt = 1;
            aclError e = aclrtBinaryLoadFromData(buf.data(), n, &opts, &g_lib.bin);
            Check(e, "aclrtBinaryLoad");
            if (g_lib.bin == nullptr) {
                throw std::runtime_error("aclrtBinaryLoad returned null handle for " + path);
            }
        },
        py::arg("path"));
    m.def(
        "try_load_file",
        [](const std::string &path) {
            aclrtBinHandle h = nullptr;
            Check(aclrtBinaryLoadFromFile(path.c_str(), nullptr, &h), "aclrtBinaryLoadFromFile");
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
        },
        py::arg("path"));
    m.def(
        "try_load",
        [](const std::string &path, uint32_t magic) {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) {
                throw std::runtime_error("cannot open " + path);
            }
            std::streamsize n = f.tellg();
            f.seekg(0);
            std::vector<char> buf(n);
            f.read(buf.data(), n);
            aclrtBinaryLoadOption opt;
            opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
            opt.value.magic = magic;
            aclrtBinaryLoadOptions opts;
            opts.options = &opt;
            opts.numOpt = 1;
            aclrtBinHandle h = nullptr;
            Check(aclrtBinaryLoadFromData(buf.data(), n, &opts, &h), "aclrtBinaryLoadFromData");
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
        },
        py::arg("path"), py::arg("magic"));
    m.def(
        "try_get_function",
        [](uint64_t handle, const std::string &name) {
            aclrtFuncHandle fn = nullptr;
            Check(aclrtBinaryGetFunction(reinterpret_cast<aclrtBinHandle>(handle), name.c_str(), &fn),
                  "aclrtBinaryGetFunction");
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fn));
        },
        py::arg("handle"), py::arg("name"));
    m.def(
        "get_function",
        [](const std::string &name) {
            aclrtFuncHandle fn = nullptr;
            Check(aclrtBinaryGetFunction(g_lib.bin, name.c_str(), &fn), "aclrtBinaryGetFunction");
            g_lib.funcs[name] = fn;
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fn));
        },
        py::arg("name"));
    m.def(
        "rtc_compile",
        [](const std::string &src, const std::string &funcName, const std::string &arch) {
            EnsureRt();
            aclrtcProg prog = nullptr;
            Check(aclrtcCreateProg(&prog, src.c_str(), funcName.c_str(), 0, nullptr, nullptr),
                  "aclrtcCreateProg");
            // The CANN layout is <ASCEND_HOME_PATH>/<arch>-linux; derive it at
            // runtime so RTC keeps working when the toolkit is installed
            // elsewhere or on a different host architecture.
            const char *cannHomeEnv = std::getenv("ASCEND_HOME_PATH");
            const std::string cannHome =
                (cannHomeEnv != nullptr && cannHomeEnv[0] != '\0')
                    ? cannHomeEnv
                    : "/usr/local/Ascend/ascend-toolkit/latest";
            const std::string ascBase = cannHome +
#if defined(__aarch64__)
                "/aarch64-linux";
#else
                "/x86_64-linux";
#endif
            std::vector<std::string> optStrs = {
                arch.empty() ? "--npu-soc=Ascend910B3" : arch,
                "-O3",
                "-std=c++17",
                "-I" + ascBase + "/tikcpp/tikcfw",
                "-I" + ascBase + "/tikcpp/tikcfw/interface",
                "-I" + ascBase + "/tikcpp/tikcfw/impl",
                "-I" + ascBase + "/asc",
                "-I" + ascBase + "/asc/include",
                "-I" + ascBase + "/asc/include/basic_api",
                "-I" + ascBase + "/asc/include/adv_api",
                "-I" + ascBase + "/asc/include/c_api",
                "-I" + ascBase + "/asc/impl/basic_api",
                "-I" + ascBase + "/asc/impl/adv_api",
                "-I" + ascBase + "/asc/impl/c_api",
                "-I" + ascBase + "/asc/impl/utils",
                "-include" + cannHome + "/include/version/asc_devkit_version.h",
            };
            // Extra compiler options for experiments, whitespace separated:
            //   KDA_RTC_OPTS="--cce-disable-asc-reserved-ubuf -O2"
            std::vector<std::string> extraOpts;
            if (const char *extra = std::getenv("KDA_RTC_OPTS")) {
                std::stringstream ss(extra);
                std::string tok;
                while (ss >> tok) {
                    extraOpts.push_back(tok);
                }
                for (auto &o : extraOpts) {
                    optStrs.push_back(o);
                }
            }
            std::vector<const char *> options;
            for (auto &o : optStrs) {
                options.push_back(o.c_str());
            }
            aclError e = aclrtcCompileProg(prog, static_cast<int>(options.size()), options.data());
            if (e != ACL_SUCCESS) {
                size_t logSize = 0;
                aclrtcGetCompileLogSize(prog, &logSize);
                std::string log(logSize, '\0');
                if (logSize > 0) {
                    aclrtcGetCompileLog(prog, &log[0]);
                }
                aclrtcDestroyProg(&prog);
                throw std::runtime_error("aclrtcCompileProg failed, log:\n" + log);
            }
            size_t binSize = 0;
            Check(aclrtcGetBinDataSize(prog, &binSize), "aclrtcGetBinDataSize");
            std::vector<char> bin(binSize);
            Check(aclrtcGetBinData(prog, bin.data()), "aclrtcGetBinData");
            aclrtcDestroyProg(&prog);
            aclrtBinaryLoadOption opt;
            opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
            opt.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
            aclrtBinaryLoadOptions opts;
            opts.options = &opt;
            opts.numOpt = 1;
            aclrtBinHandle bh = nullptr;
            Check(aclrtBinaryLoadFromData(bin.data(), binSize, &opts, &bh), "aclrtBinaryLoadFromData");
            aclrtFuncHandle fn = nullptr;
            Check(aclrtBinaryGetFunction(bh, funcName.c_str(), &fn), "aclrtBinaryGetFunction");
            g_lib.funcs[funcName] = fn;
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fn));
        },
        py::arg("src"), py::arg("func_name"), py::arg("arch") = "");
    m.def(
        "launch_argsarray",
        [](const std::string &name, uint32_t numBlocks, uint64_t stream,
           const std::vector<std::string> &argBlobs) {
            EnsureRt();
            auto it = g_lib.funcs.find(name);
            if (it == g_lib.funcs.end()) {
                throw std::runtime_error("kernel not loaded: " + name);
            }
            std::vector<void *> args;
            for (auto &b : argBlobs) {
                args.push_back(const_cast<char *>(b.data()));
            }
            Check(aclrtLaunchKernelWithArgsArray(it->second, numBlocks,
                                                 reinterpret_cast<aclrtStream>(stream), nullptr,
                                                 args.data()),
                  "aclrtLaunchKernelWithArgsArray " + name);
        },
        py::arg("name"), py::arg("num_blocks"), py::arg("stream"), py::arg("arg_bytes"));
    m.def(
        "launch_argsarray_engine",
        [](const std::string &name, uint32_t numBlocks, uint64_t stream,
           const std::vector<std::string> &argBlobs, int32_t engine) {
            EnsureRt();
            auto it = g_lib.funcs.find(name);
            if (it == g_lib.funcs.end()) {
                throw std::runtime_error("kernel not loaded: " + name);
            }
            std::vector<void *> args;
            for (auto &b : argBlobs) {
                args.push_back(const_cast<char *>(b.data()));
            }
            aclrtLaunchKernelAttr attr;
            attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
            attr.value.engineType = static_cast<aclrtEngineType>(engine);
            aclrtLaunchKernelCfg cfg;
            cfg.attrs = &attr;
            cfg.numAttrs = 1;
            Check(aclrtLaunchKernelWithArgsArray(it->second, numBlocks,
                                                 reinterpret_cast<aclrtStream>(stream), &cfg,
                                                 args.data()),
                  "aclrtLaunchKernelWithArgsArray engine " + name);
        },
        py::arg("name"), py::arg("num_blocks"), py::arg("stream"), py::arg("arg_bytes"), py::arg("engine"));
    m.def(
        "launch_smoke",
        [](uint64_t srcA, uint64_t srcB, uint64_t srcH, uint64_t srcB128, uint64_t outC1,
           uint64_t outC2, uint64_t outC3, uint64_t outC5, uint64_t outC6, uint64_t stream,
           const std::vector<int32_t> &params) {
            struct {
                uint64_t a[9];
                int32_t p[18];
            } args;
            args.a[0] = srcA;
            args.a[1] = srcB;
            args.a[2] = srcH;
            args.a[3] = srcB128;
            args.a[4] = outC1;
            args.a[5] = outC2;
            args.a[6] = outC3;
            args.a[7] = outC5;
            args.a[8] = outC6;
            if (params.size() != 18) {
                throw std::runtime_error("params must have 18 ints");
            }
            for (size_t i = 0; i < 18; ++i) {
                args.p[i] = params[i];
            }
            Launch("kda_bt16_smoke_kernel", 1, &args, sizeof(args), stream);
        });
    m.def(
        "launch_hoststub",
        [](const std::string &libPath, const std::string &kernelName, uint32_t numBlocks,
           uint64_t stream, const std::vector<std::string> &argBlobs) {
            static std::unordered_map<std::string, void *> g_dl;
            static std::unordered_map<std::string, uint64_t> g_fn;
            auto getFn = [&]() -> uint64_t {
                auto it = g_fn.find(kernelName);
                if (it != g_fn.end()) {
                    return it->second;
                }
                void *dl = nullptr;
                auto dlit = g_dl.find(libPath);
                if (dlit == g_dl.end()) {
                    dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                    if (!dl) {
                        throw std::runtime_error("dlopen failed: " + std::string(dlerror()));
                    }
                    g_dl[libPath] = dl;
                } else {
                    dl = dlit->second;
                }
                // C++ mangled name: _Z<len>launch_and_profiling_<name>mjPvPS_j
                // "launch_and_profiling_" is 21 chars; full name len = 21 + kernelName.len().
                std::string mangled = "_Z" + std::to_string(21 + kernelName.size()) +
                                      "launch_and_profiling_" + kernelName + "mjPvPS_j";
                void *fn = dlsym(dl, mangled.c_str());
                if (!fn) {
                    throw std::runtime_error("dlsym failed: " + mangled + " : " +
                                             std::string(dlerror()));
                }
                g_fn[kernelName] = reinterpret_cast<uint64_t>(fn);
                return g_fn[kernelName];
            };
            auto fn = getFn();
            using LaunchFn = uint32_t (*)(uint64_t, uint32_t, void *, void **, uint32_t);
            // The host-stub launch_and_profiling_<name> expects `args` to be a pointer to a
            // device-resident blob (like aclrtlaunch_* packs the args) and `size` its byte size.
            // Copy the arg pointer array into device memory.
            size_t blobSize = argBlobs.size() * 8;
            void *devArgs = nullptr;
            Check(aclrtMalloc(&devArgs, blobSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc args");
            std::vector<void *> hostArgs;
            for (auto &b : argBlobs) {
                hostArgs.push_back(const_cast<char *>(b.data()));
            }
            Check(aclrtMemcpy(devArgs, blobSize, hostArgs.data(), blobSize, ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy args");
            uint32_t ret = reinterpret_cast<LaunchFn>(fn)(
                0, numBlocks, reinterpret_cast<void *>(stream), &devArgs,
                static_cast<uint32_t>(blobSize));
            aclrtFree(devArgs);
            if (ret != 0) {
                throw std::runtime_error("launch_and_profiling_ failed, ret=" +
                                         std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("kernel_name"), py::arg("num_blocks"),
        py::arg("stream"), py::arg("arg_bytes"));

    // Direct binding to host-stub extern "C" aclrtlaunch_kda_vec_test_kernel
    m.def(
        "launch_vec_test",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           uint64_t pG, uint64_t pOut, int32_t mode) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_vec_test_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym aclrtlaunch_kda_vec_test_kernel failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(pG), reinterpret_cast<void *>(pOut), mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_kda_vec_test_kernel failed, ret=" +
                                         std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("p_g"), py::arg("p_out"), py::arg("mode"));

    // Direct binding to host-stub extern "C" aclrtlaunch_kda_bt16_smoke_kernel
    m.def(
        "launch_smoke_offline",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           const std::vector<uint64_t> &ptrs, int32_t nProbe) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_bt16_smoke_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym aclrtlaunch_kda_bt16_smoke_kernel failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            if (ptrs.size() != 8) {
                throw std::runtime_error("smoke needs 8 input/output ptrs");
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, void *,
                                    void *, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(ptrs[0]), reinterpret_cast<void *>(ptrs[1]),
                reinterpret_cast<void *>(ptrs[2]), reinterpret_cast<void *>(ptrs[3]),
                reinterpret_cast<void *>(ptrs[4]), reinterpret_cast<void *>(ptrs[5]),
                reinterpret_cast<void *>(ptrs[6]), reinterpret_cast<void *>(ptrs[7]),
                nProbe);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_kda_bt16_smoke_kernel failed, ret=" +
                                         std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("ptrs"), py::arg("n_probe"));

    // Load an offline-compiled device ELF by entry id, launch with engine attr.
    m.def(
        "load_offline_entry",
        [](const std::string &path, uint64_t entry) {
            EnsureRt();
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) {
                throw std::runtime_error("cannot open " + path);
            }
            std::streamsize n = f.tellg();
            f.seekg(0);
            std::vector<char> buf(n);
            f.read(buf.data(), n);
            aclrtBinaryLoadOption opt;
            opt.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
            opt.value.magic = 0x41494343UL;  // CUBE_CORE
            aclrtBinaryLoadOptions opts;
            opts.options = &opt;
            opts.numOpt = 1;
            aclrtBinHandle h = nullptr;
            Check(aclrtBinaryLoadFromData(buf.data(), static_cast<size_t>(n), &opts, &h),
                  "aclrtBinaryLoadFromData");
            aclrtFuncHandle fn = nullptr;
            Check(aclrtBinaryGetFunctionByEntry(h, entry, &fn),
                  "aclrtBinaryGetFunctionByEntry");
            g_lib.funcs["offline_entry_" + std::to_string(entry)] = fn;
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fn));
        },
        py::arg("path"), py::arg("entry"));
    m.def(
        "launch_offline_entry",
        [](uint64_t entry, uint32_t numBlocks, uint64_t stream,
           const std::vector<std::string> &argBlobs, int32_t engine) {
            EnsureRt();
            std::string key = "offline_entry_" + std::to_string(entry);
            auto it = g_lib.funcs.find(key);
            if (it == g_lib.funcs.end()) {
                throw std::runtime_error("offline kernel not loaded: " + key);
            }
            std::vector<void *> args;
            for (auto &b : argBlobs) {
                args.push_back(const_cast<char *>(b.data()));
            }
            aclrtLaunchKernelAttr attr;
            attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
            attr.value.engineType = static_cast<aclrtEngineType>(engine);
            aclrtLaunchKernelCfg cfg;
            cfg.attrs = &attr;
            cfg.numAttrs = 1;
            Check(aclrtLaunchKernelWithArgsArray(it->second, numBlocks,
                                                 reinterpret_cast<aclrtStream>(stream), &cfg,
                                                 args.data()),
                  "aclrtLaunchKernelWithArgsArray offline " + key);
        },
        py::arg("entry"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("arg_bytes"), py::arg("engine"));

    // Launch host-stub vec_test with a CANN-native stream (not torch's).
    m.def(
        "launch_vec_test_ns",
        [](const std::string &libPath, uint32_t numBlocks,
           uint64_t pG, uint64_t pOut, int32_t mode) {
            EnsureRt();
            aclrtStream s = nullptr;
            Check(aclrtCreateStream(&s), "aclrtCreateStream");
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_vec_test_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym aclrtlaunch_kda_vec_test_kernel failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, s, reinterpret_cast<void *>(pG),
                reinterpret_cast<void *>(pOut), mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_kda_vec_test_kernel failed, ret=" +
                                         std::to_string(ret));
            }
            aclrtSynchronizeStream(s);
            aclrtDestroyStream(s);
        },
        py::arg("lib_path"), py::arg("num_blocks"),
        py::arg("p_g"), py::arg("p_out"), py::arg("mode"));

    // Launch host-stub vec_test2 (Duplicate->UB->GM) with torch stream.
    m.def(
        "launch_vec_test2",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           uint64_t pG, uint64_t pOut, int32_t mode) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_vec_test2_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym aclrtlaunch_kda_vec_test2_kernel failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(pG), reinterpret_cast<void *>(pOut), mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_kda_vec_test2_kernel failed, ret=" +
                                         std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("p_g"), py::arg("p_out"), py::arg("mode"));

    // Generic launcher for host-stub kernels with signature (numBlocks, stream, p0, p1, mode)
    m.def(
        "launch_vt",
        [](const std::string &libPath, const std::string &kernelName, uint32_t numBlocks,
           uint64_t stream, uint64_t p0, uint64_t p1, int32_t mode) {
            EnsureRt();
            static std::unordered_map<std::string, void *> g_dls;
            static std::unordered_map<std::string, uint64_t> g_fns;
            auto key = libPath + "#" + kernelName;
            if (g_fns.find(key) == g_fns.end()) {
                void *dl = nullptr;
                auto it = g_dls.find(libPath);
                if (it == g_dls.end()) {
                    dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                    if (!dl) {
                        throw std::runtime_error("dlopen failed");
                    }
                    g_dls[libPath] = dl;
                } else {
                    dl = it->second;
                }
                std::string sym = "aclrtlaunch_" + kernelName;
                void *fn = dlsym(dl, sym.c_str());
                if (!fn) {
                    throw std::runtime_error("dlsym " + sym + " failed");
                }
                g_fns[key] = reinterpret_cast<uint64_t>(fn);
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fns[key])(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(p0), reinterpret_cast<void *>(p1), mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_ failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("kernel_name"), py::arg("num_blocks"),
        py::arg("stream"), py::arg("p0"), py::arg("p1"), py::arg("mode"));

    // Generic host-stub launcher: passes an array of device pointers.
    // The generated aclrtlaunch_<name> has signature (numBlocks, stream, p0, p1, ..., mode)
    // We resolve the mangled launch_and_profiling_<name> and pack ptrs as the args blob.
    m.def(
        "launch_hs",
        [](const std::string &libPath, const std::string &kernelName, uint32_t numBlocks,
           uint64_t stream, const std::vector<uint64_t> &ptrs, int32_t mode) {
            EnsureRt();
            static std::unordered_map<std::string, void *> g_dls;
            static std::unordered_map<std::string, uint64_t> g_fns;
            auto key = libPath + "#" + kernelName;
            if (g_fns.find(key) == g_fns.end()) {
                void *dl = nullptr;
                auto it = g_dls.find(libPath);
                if (it == g_dls.end()) {
                    dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                    if (!dl) {
                        throw std::runtime_error("dlopen failed");
                    }
                    g_dls[libPath] = dl;
                } else {
                    dl = it->second;
                }
                std::string mangled = "_Z" + std::to_string(21 + kernelName.size()) +
                                      "launch_and_profiling_" + kernelName + "mjPvPS_j";
                void *fn = dlsym(dl, mangled.c_str());
                if (!fn) {
                    throw std::runtime_error("dlsym " + mangled + " failed");
                }
                g_fns[key] = reinterpret_cast<uint64_t>(fn);
            }
            using LaunchFn = uint32_t (*)(uint64_t, uint32_t, void *, void **, uint32_t);
            // Build a device-resident args blob: [ptrs..., mode]
            std::vector<uint64_t> blob;
            for (auto p : ptrs) {
                blob.push_back(p);
            }
            blob.push_back(static_cast<uint64_t>(mode));
            size_t blobBytes = blob.size() * sizeof(uint64_t);
            void *devArgs = nullptr;
            Check(aclrtMalloc(&devArgs, blobBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
            Check(aclrtMemcpy(devArgs, blobBytes, blob.data(), blobBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy");
            uint32_t ret = reinterpret_cast<LaunchFn>(g_fns[key])(
                0, numBlocks, reinterpret_cast<void *>(stream), &devArgs,
                static_cast<uint32_t>(blobBytes));
            aclrtFree(devArgs);
            if (ret != 0) {
                throw std::runtime_error("launch_and_profiling_ failed, ret=" +
                                         std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("kernel_name"), py::arg("num_blocks"),
        py::arg("stream"), py::arg("ptrs"), py::arg("mode"));

    // Direct binding for the K1 vector kernel (5 device ptrs + mode).
    m.def(
        "launch_k1vec",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           const std::vector<uint64_t> &ptrs, int32_t mode) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_k1vec_on_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym kda_k1vec_on failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            if (ptrs.size() != 5) {
                throw std::runtime_error("k1vec needs 5 ptrs");
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, void *,
                                    void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(ptrs[0]), reinterpret_cast<void *>(ptrs[1]),
                reinterpret_cast<void *>(ptrs[2]), reinterpret_cast<void *>(ptrs[3]),
                reinterpret_cast<void *>(ptrs[4]), mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_kda_k1vec_on_kernel failed, ret=" +
                                         std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("ptrs"), py::arg("mode"));

    // Generic launcher for host-stub kernels with 9 device ptrs + mode.
    m.def(
        "launch_hs9",
        [](const std::string &libPath, const std::string &kernelName, uint32_t numBlocks,
           uint64_t stream, const std::vector<uint64_t> &ptrs, int32_t mode) {
            EnsureRt();
            static std::unordered_map<std::string, void *> g_dls;
            static std::unordered_map<std::string, uint64_t> g_fns;
            auto key = libPath + "#" + kernelName;
            if (g_fns.find(key) == g_fns.end()) {
                void *dl = nullptr;
                auto it = g_dls.find(libPath);
                if (it == g_dls.end()) {
                    dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                    if (!dl) {
                        throw std::runtime_error("dlopen failed");
                    }
                    g_dls[libPath] = dl;
                } else {
                    dl = it->second;
                }
                std::string sym = "aclrtlaunch_" + kernelName;
                void *fn = dlsym(dl, sym.c_str());
                if (!fn) {
                    throw std::runtime_error("dlsym " + sym + " failed");
                }
                g_fns[key] = reinterpret_cast<uint64_t>(fn);
            }
            if (ptrs.size() != 9) {
                throw std::runtime_error("launch_hs9 needs 9 ptrs");
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, void *,
                                    void *, void *, void *, void *, void *, int32_t);
            // numBlocks, stream, 9 ptrs, mode = 12 args; Fn has 1 uint32 + 10 void* + 1 int32 = 12
            uint32_t ret = reinterpret_cast<Fn>(g_fns[key])(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(ptrs[0]), reinterpret_cast<void *>(ptrs[1]),
                reinterpret_cast<void *>(ptrs[2]), reinterpret_cast<void *>(ptrs[3]),
                reinterpret_cast<void *>(ptrs[4]), reinterpret_cast<void *>(ptrs[5]),
                reinterpret_cast<void *>(ptrs[6]), reinterpret_cast<void *>(ptrs[7]),
                reinterpret_cast<void *>(ptrs[8]), mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_ failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("kernel_name"), py::arg("num_blocks"),
        py::arg("stream"), py::arg("ptrs"), py::arg("mode"));

    // Generic launcher for host-stub kernels with 3 device ptrs + mode.
    m.def(
        "launch_vt3",
        [](const std::string &libPath, const std::string &kernelName, uint32_t numBlocks,
           uint64_t stream, uint64_t p0, uint64_t p1, uint64_t p2, int32_t mode) {
            EnsureRt();
            static std::unordered_map<std::string, void *> g_dls;
            static std::unordered_map<std::string, uint64_t> g_fns;
            auto key = libPath + "#" + kernelName;
            if (g_fns.find(key) == g_fns.end()) {
                void *dl = nullptr;
                auto it = g_dls.find(libPath);
                if (it == g_dls.end()) {
                    dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                    if (!dl) {
                        throw std::runtime_error("dlopen failed");
                    }
                    g_dls[libPath] = dl;
                } else {
                    dl = it->second;
                }
                std::string sym = "aclrtlaunch_" + kernelName;
                void *fn = dlsym(dl, sym.c_str());
                if (!fn) {
                    throw std::runtime_error("dlsym " + sym + " failed");
                }
                g_fns[key] = reinterpret_cast<uint64_t>(fn);
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fns[key])(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(p0), reinterpret_cast<void *>(p1),
                reinterpret_cast<void *>(p2), mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_ failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("kernel_name"), py::arg("num_blocks"),
        py::arg("stream"), py::arg("p0"), py::arg("p1"), py::arg("p2"), py::arg("mode"));

    // Dedicated launcher for kda_k1_vecop (9 ptrs + chunks).
    m.def(
        "launch_vecop",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           const std::vector<uint64_t> &ptrs, int32_t chunks) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_k1_vecop_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym kda_k1_vecop failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            if (ptrs.size() != 9) {
                throw std::runtime_error("vecop needs 9 ptrs");
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, void *,
                                    void *, void *, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(ptrs[0]), reinterpret_cast<void *>(ptrs[1]),
                reinterpret_cast<void *>(ptrs[2]), reinterpret_cast<void *>(ptrs[3]),
                reinterpret_cast<void *>(ptrs[4]), reinterpret_cast<void *>(ptrs[5]),
                reinterpret_cast<void *>(ptrs[6]), reinterpret_cast<void *>(ptrs[7]),
                reinterpret_cast<void *>(ptrs[8]), chunks);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_vecop failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("ptrs"), py::arg("chunks"));

    // Multi-block vecop: 9 ptrs + chunks/hvNum/hNum/gqa.
    m.def(
        "launch_vecop_mb",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           const std::vector<uint64_t> &ptrs, int32_t chunks, int32_t hvNum, int32_t hNum, int32_t gqa) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_k1_vecop_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym vecop_mb failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            if (ptrs.size() != 9) {
                throw std::runtime_error("vecop_mb needs 9 ptrs");
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, void *,
                                    void *, void *, void *, void *, void *, int32_t, int32_t, int32_t, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(ptrs[0]), reinterpret_cast<void *>(ptrs[1]),
                reinterpret_cast<void *>(ptrs[2]), reinterpret_cast<void *>(ptrs[3]),
                reinterpret_cast<void *>(ptrs[4]), reinterpret_cast<void *>(ptrs[5]),
                reinterpret_cast<void *>(ptrs[6]), reinterpret_cast<void *>(ptrs[7]),
                reinterpret_cast<void *>(ptrs[8]), chunks, hvNum, hNum, gqa);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_vecop_mb failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("ptrs"), py::arg("chunks"), py::arg("hv_num"), py::arg("h_num"), py::arg("gqa"));

    // Dedicated launcher for K2 fused kernel: 10 ptrs + nt + mode.
    m.def(
        "launch_k2f",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           const std::vector<uint64_t> &ptrs, int32_t nt, int32_t mode) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) { throw std::runtime_error("dlopen failed"); }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_k2_fused_kernel");
                if (!fn) { throw std::runtime_error("dlsym k2f failed"); }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            if (ptrs.size() != 10) { throw std::runtime_error("k2f needs 10 ptrs"); }
            using Fn = uint32_t (*)(uint32_t, void*, void*, void*, void*, void*, void*,
                                    void*, void*, void*, void*, void*, int32_t, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(ptrs[0]), reinterpret_cast<void *>(ptrs[1]),
                reinterpret_cast<void *>(ptrs[2]), reinterpret_cast<void *>(ptrs[3]),
                reinterpret_cast<void *>(ptrs[4]), reinterpret_cast<void *>(ptrs[5]),
                reinterpret_cast<void *>(ptrs[6]), reinterpret_cast<void *>(ptrs[7]),
                reinterpret_cast<void *>(ptrs[8]), reinterpret_cast<void *>(ptrs[9]),
                nt, mode);
            if (ret != 0) { throw std::runtime_error("k2f launch failed ret=" + std::to_string(ret)); }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("ptrs"), py::arg("nt"), py::arg("mode"));

    // Generic host-stub launcher for kernels with N ptrs + mode/chunks (per-kernel arity).
    // Used for cube1 (6 ptrs), cube2w (3 ptrs), cube2u (3 ptrs).
    m.def(
        "launch_hs_kernel",
        [](const std::string &libPath, const std::string &kernelName, uint32_t numBlocks,
           uint64_t stream, const std::vector<uint64_t> &ptrs, int32_t extra) {
            EnsureRt();
            static std::unordered_map<std::string, void *> g_dls;
            static std::unordered_map<std::string, uint64_t> g_fns;
            auto key = libPath + "#" + kernelName;
            if (g_fns.find(key) == g_fns.end()) {
                void *dl = nullptr;
                auto it = g_dls.find(libPath);
                if (it == g_dls.end()) {
                    dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                    if (!dl) {
                        throw std::runtime_error("dlopen failed");
                    }
                    g_dls[libPath] = dl;
                } else {
                    dl = it->second;
                }
                std::string sym = "aclrtlaunch_" + kernelName;
                void *fn = dlsym(dl, sym.c_str());
                if (!fn) {
                    throw std::runtime_error("dlsym " + sym + " failed");
                }
                g_fns[key] = reinterpret_cast<uint64_t>(fn);
            }
            // Build a generic call by casting to a max-arity function pointer.
            // Signatures are (numBlocks, stream, p0..pN, extra). We use a raw ABI call via
            // uint64 registers; this works for <= 8 args on this ABI.
            using Fn = uint32_t (*)(uint32_t, void *, uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, int32_t);
            auto fn = reinterpret_cast<Fn>(g_fns[key]);
            uint64_t a[6] = {0};
            for (size_t i = 0; i < ptrs.size() && i < 6; ++i) {
                a[i] = ptrs[i];
            }
            uint32_t ret = fn(numBlocks, reinterpret_cast<void *>(stream), a[0], a[1], a[2],
                              a[3], a[4], a[5], extra);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_ failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("kernel_name"), py::arg("num_blocks"),
        py::arg("stream"), py::arg("ptrs"), py::arg("extra"));

    // Dedicated launcher for cube2w (3 ptrs + chunks + mode).
    m.def(
        "launch_cube2w",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           uint64_t pA, uint64_t pWop, uint64_t pW, int32_t chunks, int32_t mode) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_k1_cube2w_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym cube2w failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, int32_t, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(pA), reinterpret_cast<void *>(pWop),
                reinterpret_cast<void *>(pW), chunks, mode);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_cube2w failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("p_a"), py::arg("p_wop"), py::arg("p_w"), py::arg("chunks"), py::arg("mode"));

    // Dedicated launcher for cube2u (3 ptrs + chunks).
    m.def(
        "launch_cube2u",
        [](const std::string &libPath, uint32_t numBlocks, uint64_t stream,
           uint64_t pA, uint64_t pUop, uint64_t pU, int32_t chunks) {
            EnsureRt();
            static void *g_dl = nullptr;
            static uint64_t g_fn = 0;
            if (!g_fn) {
                g_dl = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (!g_dl) {
                    throw std::runtime_error("dlopen failed");
                }
                void *fn = dlsym(g_dl, "aclrtlaunch_kda_k1_cube2u_kernel");
                if (!fn) {
                    throw std::runtime_error("dlsym cube2u failed");
                }
                g_fn = reinterpret_cast<uint64_t>(fn);
            }
            using Fn = uint32_t (*)(uint32_t, void *, void *, void *, void *, int32_t);
            uint32_t ret = reinterpret_cast<Fn>(g_fn)(
                numBlocks, reinterpret_cast<void *>(stream),
                reinterpret_cast<void *>(pA), reinterpret_cast<void *>(pUop),
                reinterpret_cast<void *>(pU), chunks);
            if (ret != 0) {
                throw std::runtime_error("aclrtlaunch_cube2u failed, ret=" + std::to_string(ret));
            }
        },
        py::arg("lib_path"), py::arg("num_blocks"), py::arg("stream"),
        py::arg("p_a"), py::arg("p_uop"), py::arg("p_u"), py::arg("chunks"));
}
