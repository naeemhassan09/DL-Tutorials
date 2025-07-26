###############################################################
# Homework 5 – CIFAR-10 with LeNet Variants (Flux.jl)
# -------------------------------------------------------------

# Run:
#   julia --project -e 'using Pkg; Pkg.instantiate()'
#   julia hw5_cifar10_flux.jl           # runs all tasks & saves results under ./cifar10_results
#
###############################################################

using Random
Random.seed!(42)

using Flux
using MLDatasets
using JLD2
using Statistics
using Plots

# Optional deps (only if you need CSV/DataFrame outputs)
# using CSV, DataFrames

# -------------------------------------------------------------
# Device helpers (GPU if available)
# -------------------------------------------------------------
const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

dev(x) = HAS_CUDA ? gpu(x) : x
cpu_if_needed(x) = HAS_CUDA ? cpu(x) : x

# -------------------------------------------------------------
# General utilities
# -------------------------------------------------------------
struct TrainHistory
    epoch::Int
    train_loss::Float32
    train_acc::Float32
    test_loss::Float32
    test_acc::Float32
end

"""
    get_class_names()
Return CIFAR-10 class names in canonical order.
"""
get_class_names() = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]

# -------------------------------------------------------------
# Data loading
# -------------------------------------------------------------
function load_cifar10_data()
    train_x, train_y = CIFAR10(split = :train)[:]
    test_x,  test_y  = CIFAR10(split = :test)[:]

    train_x = Float32.(train_x) ./ 255f0
    test_x  = Float32.(test_x)  ./ 255f0

    train_y_hot = Flux.onehotbatch(train_y, 0:9)
    test_y_hot  = Flux.onehotbatch(test_y, 0:9)

    return (train_x, train_y_hot), (test_x, test_y_hot)
end

function create_dataloader(x, y; batchsize = 128, shuffle = true)
    Flux.DataLoader((x, y); batchsize = batchsize, shuffle = shuffle)
end

function create_subset_data(train_x, train_y, n_samples)
    idxs = Random.randperm(size(train_x, 4))[1:n_samples]
    return train_x[:, :, :, idxs], train_y[:, idxs]
end

# -------------------------------------------------------------
# Model definitions
# -------------------------------------------------------------
"""
    create_lenet(filter_size::Int=5)
Return a LeNet-like Chain where the convolutional filter size is configurable.
The dense input size is auto-computed to avoid hardcoding.
"""
function create_lenet(filter_size::Int = 5)
    conv1 = Conv((filter_size, filter_size), 3 => 6, relu)
    pool1 = MeanPool((2, 2))
    conv2 = Conv((filter_size, filter_size), 6 => 16, relu)
    pool2 = MeanPool((2, 2))
    flat  = Flux.flatten

    # compute dense input size using a dummy forward pass
    dummy = rand(Float32, 32, 32, 3, 1)
    dummy_out = flat(pool2(conv2(pool1(conv1(dummy)))))
    dense_in = length(dummy_out)

    return Chain(
        conv1,
        pool1,
        conv2,
        pool2,
        flat,
        Dense(dense_in => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end

# -------------------------------------------------------------
# Training & evaluation
# -------------------------------------------------------------
function loss_and_accuracy(model, data_loader; device = dev)
    total_loss = 0f0
    total_correct = 0
    total_samples = 0

    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = model(x)
        total_loss += Flux.logitcrossentropy(ŷ, y)
        total_correct += sum(Flux.onecold(ŷ, 0:9) .== Flux.onecold(y, 0:9))
        total_samples += size(y, 2)
    end

    avg_loss = total_loss / length(data_loader)
    acc = 100f0 * total_correct / total_samples
    return (avg_loss, acc)
end

function train!(model, train_loader, nepochs; opt = AdamW(1e-3, (0.9, 0.999), 1e-4),
                test_loader = nothing, log_every = 5, device = dev)
    opt_state = Flux.setup(opt, model)
    history = Vector{TrainHistory}()

    for epoch in 1:nepochs
        for (x, y) in train_loader
            x, y = device(x), device(y)
            gs = Flux.gradient(model) do m
                Flux.logitcrossentropy(m(x), y)
            end
            Flux.update!(opt_state, model, gs)
        end

        if (epoch % log_every == 0) || epoch == nepochs
            tl, ta = loss_and_accuracy(model, train_loader; device = device)
            if test_loader !== nothing
                vl, va = loss_and_accuracy(model, test_loader; device = device)
            else
                vl, va = NaN32, NaN32
            end
            push!(history, TrainHistory(epoch, Float32(tl), Float32(ta), Float32(vl), Float32(va)))
            @info "Epoch $(epoch)/$(nepochs)  TrainAcc=$(round(ta, digits=2))%  TestAcc=$(round(va, digits=2))%"
        end
    end
    return history
end

# -------------------------------------------------------------
# Task 1: LeNet5 on full CIFAR-10
# -------------------------------------------------------------
function task1_lenet5_cifar10(train_x, train_y, test_x, test_y; epochs = 20)
    println("Task 1: Training LeNet5 on CIFAR-10")
    train_loader = create_dataloader(train_x, train_y; batchsize = 128)
    test_loader  = create_dataloader(test_x,  test_y;  batchsize = 128, shuffle = false)

    model = dev(create_lenet(5))
    history = train!(model, train_loader, epochs;
                     test_loader = test_loader,
                     log_every = max(1, epochs ÷ 10))

    return model, history
end

# -------------------------------------------------------------
# Task 2: Dataset size vs repeated exposure
# -------------------------------------------------------------
function task2_dataset_size_experiment(train_x, train_y, test_x, test_y)
    println("Task 2: Dataset Size Experiment")
    test_loader = create_dataloader(test_x, test_y; batchsize = 128, shuffle = false)

    experiments = [
        (n_samples = 10_000, epochs = 6),
        (n_samples = 20_000, epochs = 3),
        (n_samples = 30_000, epochs = 2)
    ]

    results = Vector{NamedTuple}()
    for (i, exp) in enumerate(experiments)
        println("Experiment $i: $(exp.n_samples) samples, $(exp.epochs) epochs")
        subset_x, subset_y = create_subset_data(train_x, train_y, exp.n_samples)
        train_loader = create_dataloader(subset_x, subset_y; batchsize = 128)

        model = dev(create_lenet(5))
        history = train!(model, train_loader, exp.epochs;
                         test_loader = test_loader,
                         log_every = max(1, exp.epochs))

        final_test_acc = history[end].test_acc
        push!(results, (n_samples = exp.n_samples,
                        epochs = exp.epochs,
                        final_test_acc = final_test_acc,
                        history = history))
        println("Final test accuracy: $(round(final_test_acc, digits=2))%")
    end

    # Plot
    sample_sizes = [r.n_samples for r in results]
    test_accs    = [r.final_test_acc for r in results]

    p = plot(sample_sizes, test_accs,
             marker = :circle, linewidth = 2, markersize = 8,
             xlabel = "Number of Training Samples",
             ylabel = "Final Test Accuracy (%)",
             title = "Effect of Dataset Size on Performance",
             legend = false, grid = true)

    return results, p
end

# -------------------------------------------------------------
# Task 3: Filter size comparison (3x3, 5x5, 7x7)
# -------------------------------------------------------------
function task3_filter_size_comparison(train_x, train_y, test_x, test_y; epochs = 15)
    println("Task 3: Filter Size Comparison")
    train_loader = create_dataloader(train_x, train_y; batchsize = 128)
    test_loader  = create_dataloader(test_x,  test_y;  batchsize = 128, shuffle = false)

    filter_sizes = [3, 5, 7]
    results = Vector{NamedTuple}()

    for fs in filter_sizes
        println("Training LeNet$(fs)...")
        model = dev(create_lenet(fs))
        history = train!(model, train_loader, epochs;
                         test_loader = test_loader,
                         log_every = max(1, epochs ÷ 5))
        final_test_acc = history[end].test_acc
        push!(results, (filter_size = fs, final_test_acc = final_test_acc,
                        history = history, model = cpu_if_needed(model)))
        println("LeNet$(fs) final test accuracy: $(round(final_test_acc, digits=2))%")
    end

    f_sizes  = [r.filter_size for r in results]
    test_acc = [r.final_test_acc for r in results]

    p = plot(f_sizes, test_acc,
             marker = :circle, linewidth = 2, markersize = 8,
             xlabel = "Filter Size",
             ylabel = "Final Test Accuracy (%)",
             title = "Effect of Filter Size on LeNet Performance",
             legend = false, grid = true, xticks = f_sizes)

    return results, p
end

# -------------------------------------------------------------
# Task 4: Feature visualization for LeNet3
# -------------------------------------------------------------
function task4_feature_investigation(train_x, train_y, test_x, test_y;
                                     epochs = 10, sample_indices = [1, 100, 500])
    println("Task 4: Feature Investigation (LeNet3)")

    train_loader = create_dataloader(train_x, train_y; batchsize = 128)
    test_loader  = create_dataloader(test_x,  test_y;  batchsize = 128, shuffle = false)

    model = dev(create_lenet(3))
    history = train!(model, train_loader, epochs;
                     test_loader = test_loader,
                     log_every = max(1, epochs ÷ 5))

    class_names = get_class_names()
    # Convert one-hot test labels to indices 1..10
    test_indices = Flux.onecold(test_y, 0:9) .+ 1
    labels = class_names[test_indices]

    plots_list = []

    for (i, sidx) in enumerate(sample_indices)
        original = test_x[:, :, :, sidx]
        lbl = labels[sidx]

        # Forward passes to intermediate stages
        x = reshape(original, 32, 32, 3, 1)
        conv1_out = model[1](dev(x)) |> cpu_if_needed
        pool1_out = model[1:2](dev(x)) |> cpu_if_needed
        conv2_out = model[1:3](dev(x)) |> cpu_if_needed

        # Select a few feature maps to display
        f1_idx = 1
        f2_idx = min(2, size(conv1_out, 3))
        f3_idx = 1
        f4_idx = min(5, size(conv2_out, 3))

        p1 = heatmap(original[:, :, 1]', title = "Original ($lbl)", aspect_ratio = :equal, color = :gray)
        p2 = heatmap(conv1_out[:, :, f1_idx, 1]', title = "Conv1 - F$(f1_idx)", aspect_ratio = :equal)
        p3 = heatmap(conv1_out[:, :, f2_idx, 1]', title = "Conv1 - F$(f2_idx)", aspect_ratio = :equal)
        p4 = heatmap(pool1_out[:, :, f1_idx, 1]', title = "After Pool1 - F$(f1_idx)", aspect_ratio = :equal)
        p5 = heatmap(conv2_out[:, :, f3_idx, 1]', title = "Conv2 - F$(f3_idx)", aspect_ratio = :equal)
        p6 = heatmap(conv2_out[:, :, f4_idx, 1]', title = "Conv2 - F$(f4_idx)", aspect_ratio = :equal)

        sample_plot = plot(p1, p2, p3, p4, p5, p6, layout = (2, 3),
                           plot_title = "Sample $(i): $lbl")
        push!(plots_list, sample_plot)
    end

    return model, history, plots_list
end

# -------------------------------------------------------------
# Saving helpers
# -------------------------------------------------------------
function save_everything(folder::AbstractString; kwargs...)
    isdir(folder) || mkdir(folder)
    JLD2.jldsave(joinpath(folder, "all_results.jld2"); kwargs...)
end

# -------------------------------------------------------------
# Main orchestration
# -------------------------------------------------------------
function run_all_tasks()
    println("Starting Homework 5: CNN Analysis on CIFAR-10")
    folder = "cifar10_results"
    isdir(folder) || mkdir(folder)

    # Load once
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()

    println("\nExecuting Task 1...")
    model1, hist1 = task1_lenet5_cifar10(train_x, train_y, test_x, test_y; epochs = 20)

    println("\nExecuting Task 2...")
    results2, plot2 = task2_dataset_size_experiment(train_x, train_y, test_x, test_y)
    savefig(plot2, joinpath(folder, "task2_dataset_size_effect.png"))

    println("\nExecuting Task 3...")
    results3, plot3 = task3_filter_size_comparison(train_x, train_y, test_x, test_y; epochs = 15)
    savefig(plot3, joinpath(folder, "task3_filter_size_comparison.png"))

    println("\nExecuting Task 4...")
    model4, hist4, feature_plots = task4_feature_investigation(train_x, train_y, test_x, test_y;
                                                               epochs = 10, sample_indices = [1, 100, 500])
    for (i, p) in enumerate(feature_plots)
        savefig(p, joinpath(folder, "task4_features_sample_$(i).png"))
    end

    println("\nAll tasks completed!")

    # Save JLD2 bundle
    save_everything(folder;
        model1_state = Flux.state(cpu_if_needed(model1)), history1 = hist1,
        task2_results = results2,
        task3_results = results3,
        model4_state = Flux.state(cpu_if_needed(model4)), history4 = hist4)

    return (model1, hist1), (results2, plot2), (results3, plot3), (model4, hist4, feature_plots)
end


run_all_tasks()
