@info "Testing GPU block assembly..."

using Test
using LinearAlgebra
using CompScienceMeshes
using BEAST
using CUDA

if !CUDA.functional()
    @info "CUDA not functional — skipping GPU block-assembly tests"
else
    const BEASTCUDAExt = Base.get_extension(BEAST, :BEASTCUDAExt)
    @assert BEASTCUDAExt !== nothing "BEASTCUDAExt failed to load."

    using .BEASTCUDAExt: assembleblock_gpu, assembleblock_primer_gpu,
        assembleblock_body_gpu!, CuMatrixStore

    """
        run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
    Run a gpu block assembly. primer and body deliberately kept separate because
    ACA code may want to call primer once and body on multiple indices.
       """
    function run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel=:gather_tile_coop)
        ZT = BEAST.scalartype(biop, tfs, bfs)
        Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
        store = CuMatrixStore(Z_dev)
        ctx = assembleblock_primer_gpu(biop, tfs, bfs; kernel)
        assembleblock_body_gpu!(biop, tfs, test_ids, bfs, trial_ids, ctx, store; kernel)
        return Array(Z_dev)
    end

    """
        test_all_kernels(biop, tfs, test_ids, bfs, trial_ids, A_ref; atol)
    Run all four kernel versions for a dof subset and @test each against
    a CPU-assembled matrix.
    """
    function test_all_kernels(biop, tfs, test_ids, bfs, trial_ids, A_ref;
        atol=sqrt(eps(Float64)))
        for kernel in (:scatter, :gather_entry, :gather_tile, :gather_tile_coop)
            @testset "kernel=$kernel" begin
                Z = run_gpu_block(biop, tfs, test_ids, bfs, trial_ids; kernel)
                @test Z ≈ A_ref atol = atol
            end
        end
    end

    # The meshes used are two far-apart spheres. Reason for this is that
    # the DoubleQuadRule is the main one implemented as only far-field matrix
    # blocks are handled by ACA.
    const SPHERE = joinpath(@__DIR__, "..", "assets", "sphere5.in")
    const K = 2π / 200.0   # long wavelength keeps spheres well in far-field

    @testset "GPU assembly: Maxwell3D singlelayer operator & Raviart-Thomas basis" begin
        sphere = readmesh(SPHERE, T=Float64)
        sphere2 = translate(sphere, [0.0, 0.0, 4.0])
        op = Maxwell3D.singlelayer(wavenumber=K)
        X = raviartthomas(sphere)
        X2 = raviartthomas(sphere2)
        A = assemble(op, X, X2)
        n = numfunctions(X)

        @info "System matrix size: $(n^2)"

        @testset "full block" begin
            test_all_kernels(op, X, collect(1:n), X2, collect(1:n), A)
        end

        @testset "row subset I, all trial dofs" begin
            I = [3, 2, 7]
            test_all_kernels(op, X, I, X2, collect(1:n), A[I, :])
        end

        @testset "row+col subset I,J" begin
            I = [3, 2, 7]
            J = [11, 5, 9, 1]
            test_all_kernels(op, X, I, X2, J, A[I, J])
        end

        @testset "disjoint small subsets" begin
            I = collect(1:5)
            J = collect((n-4):n)
            test_all_kernels(op, X, I, X2, J, A[I, J])
        end
    end

    @testset "GPU assembly: Maxwell3D doublelayer operator & Raviart-Thomas basis" begin
        sphere = readmesh(SPHERE, T=Float64)
        sphere2 = translate(sphere, [0.0, 0.0, 4.0])
        op = Maxwell3D.doublelayer(wavenumber=K)
        X = raviartthomas(sphere)
        X2 = raviartthomas(sphere2)
        A = assemble(op, X, X2)
        n = numfunctions(X)

        @testset "full block" begin
            test_all_kernels(op, X, collect(1:n), X2, collect(1:n), A)
        end

        @testset "row+col subset I,J" begin
            I = [1, 4, 9]
            J = [2, 6, 8]
            test_all_kernels(op, X, I, X2, J, A[I, J])
        end
    end

    # =========================================================================
    # Helmholtz3D — piecewise-constant Lagrange (P0) basis
    # BEAST's Helmholtz operators use gamma=ik (imaginary wavenumber convention).
    # =========================================================================

    @testset "GPU — Helmholtz3D singlelayer × P0 Lagrange" begin
        sphere = readmesh(SPHERE, T=Float64)
        sphere2 = translate(sphere, [0.0, 0.0, 4.0])
        op = Helmholtz3D.singlelayer(gamma=im * K)
        X = lagrangecxd0(sphere)
        X2 = lagrangecxd0(sphere2)
        A = assemble(op, X, X2)
        n = numfunctions(X)

        @testset "full block" begin
            test_all_kernels(op, X, collect(1:n), X2, collect(1:n), A)
        end

        @testset "row+col subset I,J" begin
            I = [1, 3, 5]
            J = [2, 4, 6]
            test_all_kernels(op, X, I, X2, J, A[I, J])
        end

        @testset "disjoint small subsets" begin
            I = collect(1:4)
            J = collect((n-3):n)
            test_all_kernels(op, X, I, X2, J, A[I, J])
        end
    end

    @testset "GPU — Helmholtz3D doublelayer × P0 Lagrange" begin
        sphere = readmesh(SPHERE, T=Float64)
        sphere2 = translate(sphere, [0.0, 0.0, 4.0])
        op = Helmholtz3D.doublelayer(gamma=im * K)
        X = lagrangecxd0(sphere)
        X2 = lagrangecxd0(sphere2)
        A = assemble(op, X, X2)
        n = numfunctions(X)

        @testset "full block" begin
            test_all_kernels(op, X, collect(1:n), X2, collect(1:n), A)
        end

        @testset "row+col subset I,J" begin
            I = [2, 5, 8]
            J = [1, 3, 7]
            test_all_kernels(op, X, I, X2, J, A[I, J])
        end
    end
end
