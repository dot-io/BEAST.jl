using Test
using LinearAlgebra
using CompScienceMeshes
using BEAST
using CUDA

if !CUDA.functional()
    @info "CUDA not functional — skipping GPU block-assembly tests"
else
    # Explicitly load cuda extension for testing purposes
    const BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)
    @assert BEASTCUDAExt !== nothing "BEASTCUDAExt failed to load."

    using .BEASTCUDAExt: assembleblock_gpu!, CuMatrixStore


    # Build CPU-side primer data and bundle the device-resident inputs that
    # `assembleblock_body_gpu!` expects. Returns a NamedTuple holding everything
    # both the GPU call and the post-hoc reference comparison need.

    """
    run_gpu_block(biop::IntegralOperator, tfs::Space, test_ids::Vector{Int},
                bfs::Space, trial_ids::Vector{Int}, ctx::NamedTuple) -> Z::Matrix

    Executes `assembleblock_body_gpu!` for the given dof subsets and context,
    accumulating results into a host matrix `Z` sized to the subset dimensions.
    Argument `ctx` is return value of assembleblock_primer_gpu(...), which holds
    all the device-resident data that `assembleblock_body_gpu!` needs, as well
    as the GPU quadrature strategy to use.
"""
    function test_host_block(biop, tfs, test_ids, bfs, trial_ids)
        ZT = BEAST.scalartype(biop, tfs, bfs)
        Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
        store = CuMatrixStore(Z_dev)

        assembleblock_gpu!(biop, tfs, bfs, store; test_ids, trial_ids)

        return Array(Z_dev)
    end

    @testset "GPU block assembly — Maxwell3D singlelayer × RT(sphere5)" begin
        sphere = readmesh(joinpath(@__DIR__, "..", "assets", "sphere5.in"), T=Float64)
        sphere_trans = translate(sphere, [0.0, 0.0, 4.0]) # ensure far-field geometry for testing
        k = 2π / 200.0
        op = Maxwell3D.singlelayer(wavenumber=k)
        X = raviartthomas(sphere)
        X_trans = raviartthomas(sphere_trans)
        A_ref = assemble(op, X, X_trans)

        @testset "full block matches assemble(op, X, X)" begin
            ndofs = numfunctions(X)
            Z = test_host_block(op, X, collect(1:ndofs), X_trans, collect(1:ndofs))
            @test Z ≈ A_ref atol = sqrt(eps(Float64))
        end

        @testset "subset block matches A_ref[I, J]" begin
            I = [3, 2, 7]
            J = [11, 5, 9, 1]
            Z = test_host_block(op, X, I, X_trans, J)
            @test Z ≈ A_ref[I, J] atol = sqrt(eps(Float64))
        end

        @testset "disjoint subset — far-field block" begin
            # Non-overlapping dof subsets ensure all interactions take the
            # DoubleQuadRule path the GPU kernel is specialised for.
            I = collect(1:5)
            J = collect((numfunctions(X)-4):numfunctions(X))
            Z = test_host_block(op, X, I, X_trans, J)
            @test Z ≈ A_ref[I, J] atol = sqrt(eps(Float64))
        end
    end

    @testset "GPU block assembly — Maxwell3D doublelayer × RT(sphere5)" begin
        sphere = readmesh(joinpath(@__DIR__, "..", "assets", "sphere5.in"), T=Float64)
        sphere_trans = translate(sphere, [0.0, 0.0, 4.0]) # ensure far-field geometry for testing
        k = 2π / 200.0
        op = Maxwell3D.doublelayer(wavenumber=k)
        X = raviartthomas(sphere)
        X_trans = raviartthomas(sphere_trans)

        A_ref = assemble(op, X, X_trans)

        ndofs = numfunctions(X)
        Z = test_host_block(op, X, collect(1:ndofs), X_trans, collect(1:ndofs))
        @test Z ≈ A_ref atol = sqrt(eps(Float64))
    end
end
