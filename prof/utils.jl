using BEAST: BEAST, scalartype, Maxwell3D, numfunctions, scalartype, raviartthomas
using CUDA: zeros
using CompScienceMeshes: translate, meshsphere

ext = Base.get_extension(BEAST, :BEASTCUDA)

function rowcol_scenarios(n_test::Int, n_trial::Int)
    row_w  = min(320, n_trial)
    col_h  = min(320, n_test)
    multi_w = min(80, n_trial)
    multi_h = min(80, n_test)

    discontig_test  = [clamp(round(Int, i * n_test  / 5), 1, n_test)  for i in 1:4]
    discontig_trial = [clamp(round(Int, i * n_trial / 5), 1, n_trial) for i in 1:4]

    return [
        ("1×$(row_w) (wide row)",        [n_test ÷ 2],    collect(1:row_w)),
        ("$(col_h)×1 (tall col)",        collect(1:col_h), [n_trial ÷ 2]),
        ("4×$(multi_w) (discontig rows)", discontig_test,  collect(1:multi_w)),
        ("$(multi_h)×4 (discontig cols)", collect(1:multi_h), discontig_trial),
    ]
end

function build_and_run(kernel::Symbol;h=.2, k=4.0)
    sphere = meshsphere(radius=1.0, h=h)
    sphere2 = translate(sphere, [0.0, 0.0, 3.0])
    X1 = raviartthomas(sphere)
    X2 = raviartthomas(sphere2)

    biop = Maxwell3D.singlelayer(;wavenumber=k)

    ntest = numfunctions(X1)
    ntrial = numfunctions(X2)

    scenarios = rowcol_scenarios(ntest, ntrial)

    ZT = scalartype(biop, X1, X2)

    for (i, tup) in enumerate(scenarios)
        println("Scenario $i: ", tup[1])
        test_ids = tup[2]
        trial_ids = tup[3]
        Z_dev = zeros(ZT, length(test_ids), length(trial_ids))
        store = ext.CuMatrixStore(Z_dev)
        ctx = ext.assembleblock_primer_gpu(biop, X1, X2; kernel)
        ext.assembleblock_body_gpu!(biop, X1, test_ids, X2, trial_ids, ctx, store; kernel)
    end
end
