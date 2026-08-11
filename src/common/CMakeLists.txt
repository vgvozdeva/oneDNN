#===============================================================================
# Copyright 2019 Intel Corporation
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

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
if(DNNL_ENABLE_JIT_PROFILING OR DNNL_ENABLE_ITT_TASKS)
    # Compile the bundled ITT sources only when DNNL_ITTAPI_INCLUDE_DIR points to
    # the copy shipped with oneDNN. When it points elsewhere, an external ITT API
    # is used (headers only) and its symbols are resolved by the enclosing
    # project, so the bundled sources must not be built to avoid duplicate
    # definitions.
    get_filename_component(_itt_given "${DNNL_ITTAPI_INCLUDE_DIR}" ABSOLUTE)
    get_filename_component(_itt_bundled "${ITTNOTIFY_ROOT}" ABSOLUTE)
    if((DNNL_TARGET_ARCH STREQUAL "AARCH64" OR DNNL_TARGET_ARCH STREQUAL "X64")
       AND _itt_given STREQUAL _itt_bundled)
        file(GLOB ITT_SOURCES
            ${ITTNOTIFY_ROOT}/ittnotify/*.c
            )
        list(APPEND SOURCES ${ITT_SOURCES})

        # Add processor trace sources
        if(DNNL_TARGET_ARCH STREQUAL "X64")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITT_API_IPT_SUPPORT")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DITT_API_IPT_SUPPORT")
            if(UNIX OR MINGW)
                enable_language(ASM)
                set(ITT_PT ${ITTNOTIFY_ROOT}/ittnotify/ittptmark64.S)
            else()
                enable_language(ASM_MASM)
                set(ITT_PT ${ITTNOTIFY_ROOT}/ittnotify/ittptmark64.asm)
            endif()
            list(APPEND SOURCES ${ITT_PT})
        endif()
    endif()
endif()

if(NOT DNNL_CPU_RUNTIME STREQUAL "THREADPOOL")
    # avoid building and linking empty objects
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/stream_threadpool.cpp")
endif()

if(NOT DNNL_EXPERIMENTAL_LOGGING)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/logging.cpp")
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_common)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
