
# Configure and launch a GPU kernel, given the kernel function and its arguments.
function kernel_config!(gpu_kernel,args...)
    kernel = @cuda launch=false gpu_kernel(args...)
    config = launch_configuration(kernel.fun)
    
    threads = config.threads
    blocks  = config.blocks
    @show config
    CUDA.@sync begin
         kernel(args...; threads, blocks)
    end

    return 
end

# Configure and launch a GPU kernel with shared memory, given the kernel function and its arguments.
# The `compute_shmem` argument is a function that takes the number of threads and returns the amount of shared memory to allocate.
function kernel_config_shared!(gpu_kernel,compute_shmem,args...)
    kernel = @cuda launch=false gpu_kernel(args...)
    config = launch_configuration(kernel.fun, shmem=compute_shmem)
    
    threads = config.threads
    blocks  = config.blocks
    @show config

    CUDA.@sync begin
         kernel(args...; threads, blocks,shmem=compute_shmem(threads))
    end

    return 
end

function launch_gpu_kernel!(gpu_kernel, args...;gpu_blocksize=(32,32),problem_size)
    
    @assert all(gpu_blocksize .> 0)  "GPU block size must be positive integers."

    threadsPerBlock = prod(gpu_blocksize)
    @assert threadsPerBlock <= 1024 "GPU block size exceeds maximum threads per block."


    blocks = ceil.(Int, problem_size ./ gpu_blocksize)
    println("GPU kernel: $(gpu_kernel)")
    println("Problem size: $problem_size")
    println("Blocks: $blocks, Threads per block: $gpu_blocksize")

    CUDA.@sync begin
        @cuda blocks=blocks threads=gpu_blocksize gpu_kernel(args...)
    end

    return
end