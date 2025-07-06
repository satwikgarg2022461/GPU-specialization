# Capstone GPU edge detection

**Capstone Project of the Coursera
[GPU Programming Specialization](https://www.coursera.org/specializations/gpu-programming)**

This project is based on my submission for the [coursera cuda at scale](https://github.com/alex-n-braun/coursera_cuda_at_scale) project.

| edges                              | generated from                   |
| ---------------------------------- | -------------------------------- |
| ![edges image](data/Lena_edge.png) | ![original image](data/Lena.png) |
|                                    |                                  |


## Project Overview and Build Instructions

### Code Organization

`bin/`
After building using make, this folder will hold the executable `edgeDetection` together with a number of object files.

`data/`
This folder holds example data, an image [`Lena.png`](./data/Lena.png). By default, the output is stored in the same folder.

`include/`
Holds header files.

`src/`
Holds the source code of the project

`.clang-format`, `.clang-tidy`
Config files for setting up clang tools that can be used for formatting and
cleaning the code.

`README.md`
This file describes the project. It holds human-readable instructions for building and executing the code.

`Makefile`
This file contains the instructions for building the project using the make utility. It specifies the dependencies, compilation flags, and the target executable to be generated.

### Supported OSes

The project was testet on Ubuntu 24.04.

### Supported CPU Architecture

The project was tested on x86_64.

### CUDA APIs involved

NVIDIA CUDA Deep Neural Network library (cuDNN)

### Dependencies needed to build/run

- [FreeImage](https://freeimage.sourceforge.io/) On Ubuntu, `apt install libfreeimage-dev`
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/) Need to [install NVidia cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)
- [opencv](https://opencv.org/) `apt install libopencv-dev`

For using tools like clang-format and clang-tidy, you need to install separate
packages

### Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform. The project was tested with CUDA 12.5.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

### Tools

- `bear` for creating a `compile_commands.json` file. This can be used for clangd integration in vscode.
- `clang-tidy`, `clang-format` as linter and for code formatting

### Build and Run

The project has been tested on Ubuntu 24.04. There is a [`Makefile`](./Makefile), therefore the project can be built using

```
$ make all
```

### Running the Program

You can run the program using the following command:

```bash
make run
```

This command will execute the compiled binary, applying the edge filter on the example input image (Lena.png), and save the result as Lena_edge.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
./bin/edgeDetection --input data/Lena.png --output data/Lena_edges.png
```

You can also process a video file. The i/o is implemented using opencv:

```bash
./bin/edgeDetection --input some_input.mp4 --output edges_video.mp4
```

Cleaning Up: To clean up the compiled binaries and other generated files, run:

```bash
make clean
```

This will remove all files in the bin/ directory.

## Final Words

This project has deepened my understanding of GPU programming, CUDA, and performance optimization. The Coursera GPU Programming Specialization provided a solid foundation, covering:

- Concurrent programming with GPUs
- Parallel programming using CUDA
- Scaling CUDA applications
- Utilizing advanced CUDA libraries

However, several key areas were not covered, particularly:

- Modern CUDA Features: Advancements like CUDA Graphs for optimizing execution.
- Tool Landscape:
   - Profiling: NVIDIA Nsight Systems & Compute
   - Benchmarking: Performance assessment tools
   - Debugging: Advanced GPU debugging utilities

While this project and coursework offered a strong start, ongoing learning and hands-on experimentation will be essential for mastering modern GPU development.
