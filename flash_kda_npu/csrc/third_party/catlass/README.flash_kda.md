# CATLASS headers

Unmodified `include/` snapshot of https://gitcode.com/cann/catlass.git,
commit `41bf90da655bba3c66d0acd7e00abe33960ecfd6`.
Source: the CATLASS submodule checked out by vllm-ascend on this host.
License: CANN Open Software License Agreement Version 2.0, see LICENSE.

Vendored to make RTC compilation and wheel installation independent of network
access and other workspace checkouts. FlashKDA composes upstream TileMmad,
copy, arithmetic, cast and broadcast tiles with its existing chunk/head pipeline
and uses CATLASS cross-core flags. Custom operations remain where upstream
interfaces do not match the required layout, rounding or batched transfers.
The complete include tree retains upstream dependencies without modifying
third-party headers. See ../../catlass/scheduled_tiles.hpp and the current
benchmark report at ../../../benchmarks/results/catlass_extra_tiles_910b3.md.
