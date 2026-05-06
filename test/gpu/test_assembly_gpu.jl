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

    using .BEASTCUDAExt: assembleblock_gpu, assembleblock_primer_gpu,
        assembleblock_body_gpu!, CuMatrixStore

    const ALL_KERNELS = (:scatter, :gather_entry, :gather_tile, :gather_tile_coop)

    """
        test_gpu_block(op, tfs, bfs, test_ids, trial_ids; kernel)

    Run a single GPU assembly for the given operator, spaces, and DOF subsets
    using the requested kernel, returning the result as a host `Matrix`.
    """
    function test_gpu_block(op, tfs, bfs, test_ids, trial_ids; kernel)
        ZT = BEAST.scalartype(op, tfs, bfs)
        Z_dev = CUDA.zeros(ZT, length(test_ids), length(trial_ids))
        store = CuMatrixStore(Z_dev)

        ctx = assembleblock_primer_gpu(op, tfs, bfs; kernel)
        assembleblock_body_gpu!(op, tfs, test_ids, bfs, trial_ids, ctx, store; kernel)

        return Array(Z_dev)
    end

    # ── Shared far-field geometry ────────────────────────────────────────
    # Two separated spheres guarantee that every (test, trial) element pair
    # is well-separated, so the GPU kernel's DoubleQuadRule-only path is
    # sufficient (no Sauter-Schwab or WiltonSE needed).

    sphere_a = readmesh(joinpath(@__DIR__, "..", "assets", "sphere5.in"), T=Float64)
    sphere_b = translate(sphere_a, [0.0, 0.0, 4.0])

    # ── Maxwell3D tests ──────────────────────────────────────────────────

    @testset "Maxwell3D" begin
        k = 2π / 200.0
        X_a = raviartthomas(sphere_a)
        X_b = raviartthomas(sphere_b)
        ndofs = numfunctions(X_a)

        @testset "singlelayer" begin
            op = Maxwell3D.singlelayer(wavenumber=k)
            A_ref = assemble(op, X_a, X_b)

            @testset "full block — kernel=$ker" for ker in ALL_KERNELS
                Z = test_gpu_block(op, X_a, X_b,
                    collect(1:ndofs), collect(1:ndofs); kernel=ker)
                @test Z ≈ A_ref atol=sqrt(eps(Float64))
            end

            @testset "subset block — kernel=$ker" for ker in ALL_KERNELS
                I = [3, 2, 7]
                J = [11, 5, 9, 1]
                Z = test_gpu_block(op, X_a, X_b, I, J; kernel=ker)
                @test Z ≈ A_ref[I, J] atol=sqrt(eps(Float64))
            end
        end

        @testset "doublelayer" begin
            op = Maxwell3D.doublelayer(wavenumber=k)
            A_ref = assemble(op, X_a, X_b)

            @testset "full block — kernel=$ker" for ker in ALL_KERNELS
                Z = test_gpu_block(op, X_a, X_b,
                    collect(1:ndofs), collect(1:ndofs); kernel=ker)
                @test Z ≈ A_ref atol=sqrt(eps(Float64))
            end

            @testset "subset block — kernel=$ker" for ker in ALL_KERNELS
                I = [3, 2, 7]
                J = [11, 5, 9, 1]
                Z = test_gpu_block(op, X_a, X_b, I, J; kernel=ker)
                @test Z ≈ A_ref[I, J] atol=sqrt(eps(Float64))
            end
        end
    end

    # ── Helmholtz3D tests ────────────────────────────────────────────────

    @testset "Helmholtz3D" begin
        k = 1.0
        Y_a = lagrangec0(sphere_a)
        Y_b = lagrangec0(sphere_b)
        ndofs = numfunctions(Y_a)

        @testset "singlelayer" begin
            op = Helmholtz3D.singlelayer(wavenumber=k)
            A_ref = assemble(op, Y_a, Y_b)

            @testset "full block — kernel=$ker" for ker in ALL_KERNELS
                Z = test_gpu_block(op, Y_a, Y_b,
                    collect(1:ndofs), collect(1:ndofs); kernel=ker)
                @test Z ≈ A_ref atol=sqrt(eps(Float64))
            end

            @testset "subset block — kernel=$ker" for ker in ALL_KERNELS
                I = [3, 2, 7]
                J = [11, 5, 9, 1]
                Z = test_gpu_block(op, Y_a, Y_b, I, J; kernel=ker)
                @test Z ≈ A_ref[I, J] atol=sqrt(eps(Float64))
            end
        end

        @testset "doublelayer" begin
            op = Helmholtz3D.doublelayer(wavenumber=k)
            A_ref = assemble(op, Y_a, Y_b)

            @testset "full block — kernel=$ker" for ker in ALL_KERNELS
                Z = test_gpu_block(op, Y_a, Y_b,
                    collect(1:ndofs), collect(1:ndofs); kernel=ker)
                @test Z ≈ A_ref atol=sqrt(eps(Float64))
            end

            @testset "subset block — kernel=$ker" for ker in ALL_KERNELS
                I = [3, 2, 7]
                J = [11, 5, 9, 1]
                Z = test_gpu_block(op, Y_a, Y_b, I, J; kernel=ker)
                @test Z ≈ A_ref[I, J] atol=sqrt(eps(Float64))
            end
        end
    end
end
