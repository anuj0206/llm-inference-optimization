# Third-party source notices

The build generates exact excerpts of `ggml/src/ggml-cuda/mmvq.cu` and `quantize.cu` from the pinned llama.cpp checkout, and includes its CUDA helpers. It also links the existing ggml-base host library. The original checkout is not modified. Generated excerpts and a copy of the license remain in the local build directory.

llama.cpp / ggml:

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The JSON reader/writer is the existing vendored nlohmann/json 3.12.0 header: copyright 2013-2025 Niels Lohmann, MIT license, with the same permission and warranty terms reproduced above. Its source and embedded license attribution remain in `llama.cpp/vendor/nlohmann/json.hpp`. CUDA Toolkit components remain subject to NVIDIA's retained component licenses.
