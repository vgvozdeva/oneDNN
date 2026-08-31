#===============================================================================
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================



if(itt_cmake_included)
    return()
endif()
set(itt_cmake_included true)

# Canonical location of the ITT API copy bundled with oneDNN.
set(ITTNOTIFY_ROOT "${PROJECT_SOURCE_DIR}/third_party/ittnotify"
    CACHE INTERNAL "Location of the ITT API copy bundled with oneDNN")

# DNNL_ITTAPI_INCLUDE_DIR is the user-facing knob and is never modified here:
# empty means "use the bundled copy", non-empty means "use an external ITT API".
# That emptiness is the single source of truth, checked directly wherever the
# distinction matters (see src/common/CMakeLists.txt).
#
# ITT_INCLUDE_DIR is the resolved include path derived from it. It is set before
# the early returns below so that it is available on every architecture.
if(DNNL_ITTAPI_INCLUDE_DIR)
    file(TO_CMAKE_PATH "${DNNL_ITTAPI_INCLUDE_DIR}" ITT_INCLUDE_DIR)
else()
    set(ITT_INCLUDE_DIR "${ITTNOTIFY_ROOT}")
endif()

# ITT API is only used by these features.
if(NOT (DNNL_ENABLE_ITT_TASKS OR DNNL_ENABLE_JIT_PROFILING))
    return()
endif()

# The bundled ITT sources are only built for these architectures
# (mirrors the gate in src/common/CMakeLists.txt).
if(NOT (DNNL_TARGET_ARCH STREQUAL "X64" OR DNNL_TARGET_ARCH STREQUAL "AARCH64"))
    return()
endif()

if(NOT EXISTS "${ITT_INCLUDE_DIR}/ittnotify/ittnotify.h")
    message(FATAL_ERROR
        "ITT API headers not found: "
        "'${ITT_INCLUDE_DIR}/ittnotify/ittnotify.h' does not exist. "
        "Set DNNL_ITTAPI_INCLUDE_DIR to a directory that contains the "
        "'ittnotify/' subdirectory with 'ittnotify.h' and 'jitprofiling.h'.")
endif()

if(DNNL_ITTAPI_INCLUDE_DIR)
    message(STATUS "ITT API: using external headers (${ITT_INCLUDE_DIR}); "
        "the surrounding project must provide the ITT implementation")
else()
    message(STATUS "ITT API: using bundled copy (${ITT_INCLUDE_DIR})")
endif()

# With an external ITT API the bundled implementation is not compiled (see
# src/common/CMakeLists.txt), so the __itt_*/iJIT_* symbols must come from
# elsewhere. DNNL_ITTAPI_LIBRARY, when set, is linked into libdnnl; it may be a
# path to a library or the name of a CMake target.
#
# Leaving it empty is legitimate: a STATIC libdnnl is meant to be absorbed into
# a larger binary that provides ITT itself, and even a SHARED libdnnl works when
# the loading executable exports the symbols (e.g. ITT linked with -rdynamic).
# For a SHARED libdnnl without an explicit library, though, the unresolved
# references propagate to every consumer at link time, so warn about it.
if(DNNL_ITTAPI_INCLUDE_DIR AND NOT DNNL_ITTAPI_LIBRARY
        AND DNNL_LIBRARY_TYPE STREQUAL "SHARED")
    message(WARNING
        "ITT API: external headers are used for a SHARED libdnnl, but "
        "DNNL_ITTAPI_LIBRARY is not set. The resulting library will have "
        "undefined ITT symbols and linking against it will fail unless the "
        "consumer provides an ITT implementation and exports those symbols. "
        "Set DNNL_ITTAPI_LIBRARY to link an ITT implementation into libdnnl.")
endif()
