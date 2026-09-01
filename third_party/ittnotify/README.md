This code is from [Intel(R) Instrumentation and Tracing Technology (ITT) and
Just-In-Time (JIT) API](https://github.com/intel/ittapi)

tag: 3.26.3

The public headers are separated from the implementation the same way they are
in the upstream repository. Only `include/` is added to the include path, so
oneDNN sources include the ITT API headers exactly the way a user-provided copy
of the ITT API would be included:

    include/                 - public headers (added to the include path)
        ittnotify.h          - ITT task API
        jitprofiling.h       - JIT profiling API
        legacy/ittnotify.h   - legacy ITT API
    src/                     - implementation (compiled into the library)

Local modifications: none.
