// Pre-included into gsplat's CUDA build on Windows (gsplat_trainer._import_gsplat).
// rpcndr.h, pulled in by windows.h, has `#define small char`, which breaks torch's
// c10/cuda/CUDACachingAllocator.h (a parameter named `small`). Include it once here, then undo
// the macro; its include guard keeps later includes from defining it again.
#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef small
