# Lab 1. Gribanov Danil - 6133
## Hardware
- NVIDIA GeForce GTX 1050Ti;
- Intel(R) i7-8750H;

## What compare

- Numpy dot (OpenMP);
- Cuda via Cython call (GPU);
- Python (1 thread);
- Cython (1 thread);
- C++ via Cython call (1 thread);

# How to...
Build docker via
```
docker-compose up --build -d
```

Attach to docker shell and go to `/home/lab1` folder.

Inside folder run:
```
./init.sh
```

To install Python and other libraries for current work.
In the end Jupyter Notebook will start (in console will be url for it), where work can be done.

# Results

All implementations:

![all_imp](images/AllImplementations.svg)

Without python (1 thread):

![all_imp_wo_python](images/WithoutPythonloops.svg)


Gpu acceleration:

![gpu_acc](images/GPUAcceleration.svg)


# Conclusion
When dealing with large vectors (shape more than 2^24), its better to run it on with high computer library Numpy (faster/efficient compare to GPU). In smaller one, its no difference where to run it (except python-loops).

- Cython:
    
    \+ C in Python;
    
    \+ Can call native code in C/C++ (as in this work we call CUDA and native C++ code);
    
    \+ Cython code can be faster as native C/C++;
    
    \- Boilerplate code;
    
    \- C++/C like syntax, link libraries, code compliation and etc...
