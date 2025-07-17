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
#  Task 1 experiment -----------------------------------------------------------
# ----------------------------------------------------------------------------
function Task1()
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


# ----------------------------------------------------------------------------
#  Task 2: Random initialisation variability analysis ------------------------
# ----------------------------------------------------------------------------
function Task2()
    @info "Task 2: Random initialisation test with hidden layer size = 30"

    accs = Float64[]
    seeds = 1:10
    for seed in seeds
        rng = Xoshiro(seed)
        acc = train_hidden(30; epochs=10, rng=rng)
        push!(accs, acc)
        @info(@sprintf("Run %2d → acc = %.4f", seed, acc))
    end

    μ = mean(accs)
    σ = std(accs)
    @info(@sprintf("Mean accuracy: %.4f", μ))
    @info(@sprintf("Std deviation: %.4f", σ))

    scatter(seeds, accs .* 100;
        xlabel="Run #",
        ylabel="Test accuracy (%)",
        title="Impact of Random Initialization (Hidden=30)",
        legend=false,
        marker=:diamond)

    hline!([μ * 100], label="Mean", linestyle=:dash)
    savefig("random_init_accuracy.png")
    @info "Plot saved to random_init_accuracy.png"
end


# ----------------------------------------------------------------------------
#  Task 3: Training with decaying learning rate -------------------------------
# ----------------------------------------------------------------------------
function Task3()
    @info "Task 3: Batch size = 32, Epochs = 25, LR decay every 5 epochs"

    batch_size = 32
    total_epochs = 25
    init_lr = 3e-3
    decay_rate = 0.5

    # Load data
    train_dl = mnistloader(train_df, batch_size)
    test_dl  = mnistloader(test_df, 10000)

    # Model and parameters
    model = make_mlp(30)
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model)

    # Initial optimizer
    η = init_lr
    opt = AdamW(η)
    state = Training.TrainState(model, ps, st, opt)

    accs = Float64[]

    for epoch in 1:total_epochs
        # Adjust learning rate every 5 epochs
        if epoch % 5 == 0
            η *= decay_rate
            state = Training.TrainState(model, state.parameters, state.states, AdamW(η))
            @info(@sprintf("Epoch %2d → Decayed LR: %.5f", epoch, η))
        end

        vjp = AutoZygote()
        for batch in train_dl
            _,_,_,state = Training.single_train_step!(vjp, LOSS, batch, state)
        end

        acc = accuracy(model, state.parameters, state.states, test_dl)
        push!(accs, acc)
        @info(@sprintf("Epoch %2d → Test accuracy: %.4f", epoch, acc))
    end

    plot(1:total_epochs, accs .* 100;
         xlabel="Epoch",
         ylabel="Test accuracy (%)",
         title="Task 3: Accuracy with LR Decay",
         marker=:circle)

    savefig("task3_lr_decay.png")
    @info "Plot saved to task3_lr_decay.png"
end

# ----------------------------------------------------------------------------
#  Task 4: Grid Search (batch size × learning rate) ---------------------------
# ----------------------------------------------------------------------------
function Task4()
    @info "Task 4: Grid search for batch size × learning rate schedule"

    batch_sizes = [32, 64, 128]
    learning_rates = [1e-3, 3e-3, 1e-2]
    epochs = 15
    decay_rate = 0.5

    acc_matrix = zeros(length(batch_sizes), length(learning_rates))

    for (i, batch_size) in enumerate(batch_sizes)
        for (j, lr_init) in enumerate(learning_rates)
            @info(@sprintf("Batch size: %d | Init LR: %.4f", batch_size, lr_init))

            # Prepare data loader
            train_dl = mnistloader(train_df, batch_size)
            test_dl  = mnistloader(test_df, 10000)

            # Model
            model = make_mlp(30)
            rng = Xoshiro(1234)
            ps, st = Lux.setup(rng, model)

            # Optimizer state
            η = lr_init
            opt = AdamW(η)
            state = Training.TrainState(model, ps, st, opt)

            for epoch in 1:epochs
                if epoch % 5 == 0
                    η *= decay_rate
                    state = Training.TrainState(model, state.parameters, state.states, AdamW(η))
                end
                for batch in train_dl
                    _, _, _, state = Training.single_train_step!(AutoZygote(), LOSS, batch, state)
                end
            end

            acc = accuracy(model, state.parameters, state.states, test_dl)
            acc_matrix[i, j] = acc
            @info(@sprintf("→ Final Accuracy: %.4f", acc))
        end
    end

    heatmap(
        string.(learning_rates),
        string.(batch_sizes),
        acc_matrix .* 100,
        xlabel = "Initial Learning Rate",
        ylabel = "Batch Size",
        title = "Grid Search Accuracy (%)",
        c = :viridis
    )

    savefig("task4_grid_search.png")
    @info "Saved heatmap to task4_grid_search.png"
end

# ----------------------------------------------------------------------------
#  Task 5: Retrain using best hyperparameters from Task 4 ---------------------
# ----------------------------------------------------------------------------
function Task5()
    @info "Task 5: Retrain with best parameters from grid search (compare with Task 3)"

    # Set best parameters manually (from Task 4 heatmap)
    # Update these based on your best observed values
    best_batch_size = 64
    best_init_lr    = 3e-3
    decay_rate      = 0.5
    epochs          = 25

    # Load data
    train_dl = mnistloader(train_df, best_batch_size)
    test_dl  = mnistloader(test_df, 10000)

    # Model
    model = make_mlp(30)
    rng = Xoshiro(9876)
    ps, st = Lux.setup(rng, model)
    η = best_init_lr
    opt = AdamW(η)
    state = Training.TrainState(model, ps, st, opt)

    # Train
    for epoch in 1:epochs
        if epoch % 5 == 0
            η *= decay_rate
            state = Training.TrainState(model, state.parameters, state.states, AdamW(η))
        end
        for batch in train_dl
            _, _, _, state = Training.single_train_step!(AutoZygote(), LOSS, batch, state)
        end
        @info(@sprintf("Epoch %2d complete", epoch))
    end

    # Evaluate final test accuracy
    final_acc = accuracy(model, state.parameters, state.states, test_dl)
    @info(@sprintf("Task 5: Final accuracy with optimized params = %.4f", final_acc))

    return final_acc
end
# Run Task 1
Task1()

# Run Task 2
Task2()

# Run Task 3
Task3() 

# Run Task 4
Task4()

# Run Task 5
final_acc = Task5()
@info(@sprintf("Task 5: Final accuracy with best hyperparameters = %.4f", final_acc))