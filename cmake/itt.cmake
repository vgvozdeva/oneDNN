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

# ITT API is only used by these features.
if(NOT (DNNL_ENABLE_ITT_TASKS OR DNNL_ENABLE_JIT_PROFILING))
    return()
endif()

# The bundled ITT sources are only built for these architectures
# (mirrors the gate in src/common/CMakeLists.txt).
if(NOT (DNNL_TARGET_ARCH STREQUAL "X64" OR DNNL_TARGET_ARCH STREQUAL "AARCH64"))
    return()
endif()

file(TO_CMAKE_PATH "${DNNL_ITTAPI_INCLUDE_DIR}" DNNL_ITTAPI_INCLUDE_DIR)

if(NOT EXISTS "${DNNL_ITTAPI_INCLUDE_DIR}/ittnotify/ittnotify.h")
    message(FATAL_ERROR
        "ITT API headers not found: "
        "'${DNNL_ITTAPI_INCLUDE_DIR}/ittnotify/ittnotify.h' does not exist. "
        "Set DNNL_ITTAPI_INCLUDE_DIR to a directory that contains the "
        "'ittnotify/' subdirectory with 'ittnotify.h' and 'jitprofiling.h'.")
endif()

# When DNNL_ITTAPI_INCLUDE_DIR points to the copy shipped with oneDNN, the
# bundled sources are compiled into the library (see src/common/CMakeLists.txt).
# Otherwise an external ITT API is used (headers only) and its symbols are
# expected to be provided by the enclosing project.
get_filename_component(_itt_bundled
    "${PROJECT_SOURCE_DIR}/third_party/ittnotify" ABSOLUTE)
get_filename_component(_itt_given "${DNNL_ITTAPI_INCLUDE_DIR}" ABSOLUTE)

if(_itt_given STREQUAL _itt_bundled)
    message(STATUS "ITT API: using bundled copy (${DNNL_ITTAPI_INCLUDE_DIR})")
else()
    message(STATUS "ITT API: using external headers (${DNNL_ITTAPI_INCLUDE_DIR})")
endif()
