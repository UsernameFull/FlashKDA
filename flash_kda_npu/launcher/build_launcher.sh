#!/bin/bash
set -e

# Build launcher.so for AscendC RTC kernel compilation and execution
# Usage: ./build_launcher.sh

cd "$(dirname "$0")"

python -c "
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

home = Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.1.0'))
arch = os.environ.get('ASCEND_ARCH', os.uname().machine) + '-linux'
inc = str(home / arch / 'include')
lib = str(home / arch / 'lib64')

name = os.environ.get('LAUNCHER_NAME', 'kda_bt16_launcher')
kwargs = {}
if name == 'kda_ascendc_v1_launcher':
    build_dir = Path.cwd().parents[1] / 'build/S02_clean/torch_extensions' / name
    build_dir.mkdir(parents=True, exist_ok=True)
    kwargs['build_directory'] = str(build_dir)

launcher = load(
    name=name,
    sources=['launcher.cpp'],
    extra_cflags=['-O3', '-std=c++17', '-I' + inc],
    extra_ldflags=['-L' + lib, '-lascendcl', '-lacl_rtc'],
    verbose=True,
    **kwargs
)

print('Build successful. Module exports:')
print(dir(launcher))
"

echo "Done. Launcher built and cached in torch_extensions."
