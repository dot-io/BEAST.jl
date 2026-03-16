# Tradeoffs for scattering within the momintegrals kernel vs in a separate kernel:

+Separate kernel:
. Could maybe use shared memory tiling?
. scheduling flexibility (?adv)

# Using CUDA.jl's handy launch_configuration function

to auto-select block size/count based on detected hardware

# Optimization: while loops instead of steprange in both kernels

Reduces register usage per thread running the kernel, directly increases occupancy
