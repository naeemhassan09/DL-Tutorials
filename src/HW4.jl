#!/usr/bin/env julia
# =============================================================================
#  Task 1 – Fashion‑MNIST (local CSV) · Lux.jl
# =============================================================================
using Lux, MLUtils, Optimisers, OneHotArrays, Random, Statistics, Printf, Zygote
using CSV, DataFrames, Plots

# ----------------------------------------------------------------------------
#  CSV → DataLoader ----------------------------------------------------------
# ----------------------------------------------------------------------------
flatten_last(x) = reshape(x, :, size(x, ndims(x)))

function meanpool2(x)
    # Custom 2x2 mean pooling without Flux or NNlib
    N = size(x, 4)
    pooled = Array{Float32}(undef, 14, 14, 1, N)
    for n in 1:N
        for i in 1:14, j in 1:14
            patch = x[2i-1:2i, 2j-1:2j, 1, n]
            pooled[i, j, 1, n] = mean(patch)
        end
    end
    return pooled
end

function mnistloader(df::DataFrame, batch_size::Int)
    xmat = Matrix{Float32}(select(df, Not(:label)))            # (N, 784)
    N = size(xmat, 1)
    x4d = reshape(permutedims(xmat), 28, 28, 1, N)            # (28,28,1,N)
    x4d = reverse(x4d, dims = 1)
    x4d .*= 1 / 255f0
    x4d = meanpool2(x4d)
    x   = flatten_last(x4d)                                    # (196,N)
    y   = onehotbatch(Vector(df.label), 0:9)
    return DataLoader((x, y); batchsize = batch_size, shuffle = true)
end

# ----------------------------------------------------------------------------
#  Load Fashion-MNIST CSVs --------------------------------------------------
# ----------------------------------------------------------------------------
train_df = CSV.read("./FashionMNIST/fashion-mnist_train.csv", DataFrame; header=1)
test_df  = CSV.read("./FashionMNIST/fashion-mnist_test.csv",  DataFrame; header=1)

# ----------------------------------------------------------------------------
#  Model / loss / accuracy --------------------------------------------------
# ----------------------------------------------------------------------------
make_mlp(h) = Chain(Dense(196 => h, relu), Dense(h => 10))
const LOSS = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, loader)
    st_test = Lux.testmode(st)
    corr = tot = 0
    for (x,y) in loader
        pred = onecold(Array(first(model(x, ps, st_test))), 0:9)
        targ = onecold(y, 0:9)
        corr += sum(pred .== targ); tot += length(targ)
    end
    return corr / tot
end

# ----------------------------------------------------------------------------
#  Train one hidden size ----------------------------------------------------
# ----------------------------------------------------------------------------
function train_hidden(h; epochs=10, rng=Xoshiro(1))
    train_dl = mnistloader(train_df, 512)
    test_dl  = mnistloader(test_df, 10000)

    model = make_mlp(h)
    ps, st = Lux.setup(rng, model)
    state  = Training.TrainState(model, ps, st, AdamW(lambda=3e-4))

    for _ in 1:epochs
       vjp = AutoZygote()
        for batch in train_dl
            _,_,_,state = Training.single_train_step!(vjp, LOSS, batch, state)
        end

        # Optionally, you can print the training loss here
        # println("Epoch completed")
    end

    return accuracy(model, state.parameters, state.states, test_dl)
end

# ----------------------------------------------------------------------------
#  Main experiment -----------------------------------------------------------
# ----------------------------------------------------------------------------
function main()
    hidden_sizes = [10, 20, 40, 50, 100, 300]
    accs = Float64[]
    @info "Task 1: one-hidden-layer sweep on local Fashion-MNIST CSVs"
    for h in hidden_sizes
        acc = train_hidden(h)
        push!(accs, acc)
        @info(@sprintf("hidden=%3d → acc=%.4f", h, acc))
    end

    plot(hidden_sizes, accs .* 100; xlabel="Hidden size", ylabel="Test accuracy (%)",
         marker=:circle, title="Fashion-MNIST CSV : accuracy vs hidden size")
    savefig("hidden_vs_accuracy.png")
    @info "Plot saved to hidden_vs_accuracy.png"
end

main()