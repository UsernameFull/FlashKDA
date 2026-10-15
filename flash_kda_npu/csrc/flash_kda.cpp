// Host-side launcher: RTC-compiles AscendC sources and launches kernels on
// the current torch_npu stream via aclrtLaunchKernelWithArgsArray.
#include <acl/acl_rt.h>
#include <acl/acl_rt_compile.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

std::unordered_map<std::string, aclrtFuncHandle> g_funcs;
aclrtContext g_ctx = nullptr;
std::mutex g_mu;

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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rtc_compile",
        [](const std::string &src, const std::string &funcName) {
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
                "--npu-soc=Ascend910B3", "-O3", "-std=c++17",
                "-include" + cannHome + "/include/version/asc_devkit_version.h",
            };
            for (const char *sub : {"tikcpp/tikcfw", "tikcpp/tikcfw/interface",
                                     "tikcpp/tikcfw/impl", "asc", "asc/include",
                                     "asc/include/basic_api", "asc/include/adv_api",
                                     "asc/include/c_api", "asc/impl/basic_api",
                                     "asc/impl/adv_api", "asc/impl/c_api",
                                     "asc/impl/utils"}) {
                optStrs.push_back("-I" + ascBase + "/" + sub);
            }
            if (const char *extra = std::getenv("KDA_RTC_OPTS")) {
                std::stringstream ss(extra);
                std::string tok;
                while (ss >> tok) {
                    optStrs.push_back(tok);
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
            g_funcs[funcName] = fn;
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fn));
        },
        py::arg("src"), py::arg("func_name"));
    m.def(
        "launch_argsarray_engine",
        [](const std::string &name, uint32_t numBlocks, uint64_t stream,
           const std::vector<std::string> &argBlobs, int32_t engine) {
            EnsureRt();
            auto it = g_funcs.find(name);
            if (it == g_funcs.end()) {
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
}
