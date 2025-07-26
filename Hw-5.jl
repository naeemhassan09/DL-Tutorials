###############################################################
# Homework 5 – CIFAR-10 with LeNet Variants (Flux.jl) 
# -------------------------------------------------------------
###############################################################

using Random; Random.seed!(42)
using Flux
using MLDatasets
using JLD2
using Statistics
using Plots

# -------------------------------------------------------------
# ⚙️ Global speed/debug configuration
# -------------------------------------------------------------
const GLOBAL_TRAIN_LIMIT = 0   # 0 = full train set
const GLOBAL_TEST_LIMIT  = 0   # 0 = full test set
const AUTO_SHORT_EPOCHS  = true   # cut epochs when limiting data
const GLOBAL_LOG_EVERY   = 5      # evaluate/log every N epochs

# -------------------------------------------------------------
# Types & helpers
# -------------------------------------------------------------
struct TrainHistory
    epoch::Int
    train_loss::Float32
    train_acc::Float32
    test_loss::Float32
    test_acc::Float32
end

get_class_names() = ["airplane","automobile","bird","cat","deer",
                     "dog","frog","horse","ship","truck"]

# Pretty summaries ------------------------------------------------
function print_task1_summary(hist::Vector{TrainHistory})
    last = hist[end]
    println("Task 1 ► TestAcc=$(round(last.test_acc,digits=2))%  TrainAcc=$(round(last.train_acc,digits=2))%  Ep=$(last.epoch)")
end

function print_task2_summary(results)
    println("Task 2 ► Dataset size vs epochs (same steps)")
    for (i,r) in enumerate(results)
        println("  [$i] n=$(r.n_samples), ep=$(r.epochs) → TestAcc=$(round(r.final_test_acc,digits=2))%")
    end
end

function print_task3_summary(results)
    println("Task 3 ► Filter size comparison")
    for r in results
        println("  fs=$(r.filter_size) → TestAcc=$(round(r.final_test_acc,digits=2))%")
    end
end

function print_task4_summary(hist::Vector{TrainHistory})
    last = hist[end]
    println("Task 4 ► LeNet3 features: TestAcc=$(round(last.test_acc,digits=2))%  Ep=$(last.epoch)")
end

# -------------------------------------------------------------
# Data
# -------------------------------------------------------------
function maybe_limit(x, y, limit)
    if limit > 0
        limit = min(limit, size(x,4))
        return x[:,:,:,1:limit], y[:,1:limit]
    end
    return x, y
end

function load_cifar10_data()
    # Load data
    train_data = CIFAR10(split=:train)
    test_data = CIFAR10(split=:test)
    
    train_x = train_data.features
    train_y = train_data.targets
    test_x = test_data.features  
    test_y = test_data.targets

    # Convert to Float32 and normalize
    train_x = Float32.(train_x) ./ 255f0
    test_x  = Float32.(test_x)  ./ 255f0

    # One-hot encode labels
    train_y_hot = Flux.onehotbatch(train_y, 0:9)
    test_y_hot  = Flux.onehotbatch(test_y, 0:9)

    # Apply limits if specified
    train_x, train_y_hot = maybe_limit(train_x, train_y_hot, GLOBAL_TRAIN_LIMIT)
    test_x,  test_y_hot  = maybe_limit(test_x,  test_y_hot,  GLOBAL_TEST_LIMIT)

    return (train_x, train_y_hot), (test_x, test_y_hot)
end

create_dataloader(x,y; batchsize=128, shuffle=true) =
    Flux.DataLoader((x,y); batchsize=batchsize, shuffle=shuffle)

create_subset_data(x,y,n_samples) = begin
    n = min(n_samples, size(x,4))
    idxs = Random.randperm(size(x,4))[1:n]
    x[:,:,:,idxs], y[:,idxs]
end

# -------------------------------------------------------------
# Model
# -------------------------------------------------------------
function create_lenet(filter_size::Int=5)
    conv1 = Conv((filter_size,filter_size), 3=>6, relu)
    pool1 = MeanPool((2,2))
    conv2 = Conv((filter_size,filter_size), 6=>16, relu)
    pool2 = MeanPool((2,2))
    flat  = Flux.flatten

    # Calculate dense layer input size
    dummy = rand(Float32,32,32,3,1)
    dense_in = length(flat(pool2(conv2(pool1(conv1(dummy))))))

    Chain(
        conv1, pool1,
        conv2, pool2,
        flat,
        Dense(dense_in=>120, relu),
        Dense(120=>84, relu),
        Dense(84=>10)
    )
end

# -------------------------------------------------------------
# Training / eval (FIXED)
# -------------------------------------------------------------
function loss_and_accuracy(model, dl)
    tot_loss = 0f0; correct = 0; n = 0
    for (x,y) in dl
        ŷ = model(x)
        tot_loss += Flux.logitcrossentropy(ŷ, y)
        correct  += sum(Flux.onecold(ŷ,0:9) .== Flux.onecold(y,0:9))
        n += size(y,2)
    end
    (tot_loss/length(dl), 100f0*correct/n)
end

function train!(model, train_loader, nepochs;
                opt=AdamW(1e-3,(0.9,0.999),1e-4),
                test_loader=nothing,
                log_every=GLOBAL_LOG_EVERY)
    
   
    opt_state = Flux.setup(opt, model)
    hist = TrainHistory[]

    for epoch in 1:nepochs
        for (x,y) in train_loader
            
            loss, grads = Flux.withgradient(model) do m
                Flux.logitcrossentropy(m(x), y)
            end
            Flux.update!(opt_state, model, grads[1])
        end
        
        if (epoch % log_every == 0) || epoch == nepochs
            tl,ta = loss_and_accuracy(model, train_loader)
            vl,va = test_loader === nothing ? (Float32(NaN), Float32(NaN)) :
                                           loss_and_accuracy(model, test_loader)
            push!(hist, TrainHistory(epoch, Float32(tl), Float32(ta), Float32(vl), Float32(va)))
            @info "Epoch $(epoch)/$(nepochs)  TrainAcc=$(round(ta,digits=2))%  TestAcc=$(round(va,digits=2))%"
        end
    end
    hist
end

# -------------------------------------------------------------
# Tasks
# -------------------------------------------------------------
function task1_lenet5_cifar10(tx,ty, vx,vy; epochs=20)
    println("Task 1: LeNet5")
    epochs = (AUTO_SHORT_EPOCHS && GLOBAL_TRAIN_LIMIT>0) ? min(epochs,3) : epochs
    tr_loader = create_dataloader(tx,ty)
    te_loader = create_dataloader(vx,vy; shuffle=false)
    model = create_lenet(5)
    hist  = train!(model, tr_loader, epochs; test_loader=te_loader)
    return model, hist
end

function task2_dataset_size_experiment(tx,ty, vx,vy)
    println("Task 2: Dataset size experiment")
    te_loader = create_dataloader(vx,vy; shuffle=false)
    exps = [(10_000,6),(20_000,3),(30_000,2)]
    if GLOBAL_TRAIN_LIMIT>0
        exps = [(min(n,GLOBAL_TRAIN_LIMIT), ep) for (n,ep) in exps]
    end
    results = NamedTuple[]
    for (i,(n,ep)) in enumerate(exps)
        println("  Exp $i → n=$n, ep=$ep")
        sub_x, sub_y = create_subset_data(tx,ty,n)
        tr_loader = create_dataloader(sub_x, sub_y)
        model = create_lenet(5)
        hist  = train!(model, tr_loader, ep; test_loader=te_loader, log_every=max(1,ep))
        push!(results, (n_samples=n, epochs=ep, final_test_acc=hist[end].test_acc, history=hist))
    end
    xs = [r.n_samples for r in results]
    ys = [r.final_test_acc for r in results]
    p = plot(xs, ys, marker=:circle, linewidth=2, markersize=8,
             xlabel="Number of Training Samples", ylabel="Final Test Accuracy (%)",
             title="Effect of Dataset Size on Performance", legend=false, grid=true)
    return results, p
end

function task3_filter_size_comparison(tx,ty, vx,vy; epochs=15)
    println("Task 3: Filter sizes")
    epochs = (AUTO_SHORT_EPOCHS && GLOBAL_TRAIN_LIMIT>0) ? min(epochs,4) : epochs
    tr_loader = create_dataloader(tx,ty)
    te_loader = create_dataloader(vx,vy; shuffle=false)

    fss = [3,5,7]
    results = NamedTuple[]
    for fs in fss
        println("  fs=$fs")
        model = create_lenet(fs)
        hist  = train!(model, tr_loader, epochs; test_loader=te_loader, log_every=max(1,epochs÷5))
        push!(results, (filter_size=fs, final_test_acc=hist[end].test_acc, history=hist, model=model))
    end
    xs = [r.filter_size for r in results]
    ys = [r.final_test_acc for r in results]
    p = plot(xs, ys, marker=:circle, linewidth=2, markersize=8,
             xlabel="Filter Size", ylabel="Final Test Accuracy (%)",
             title="Effect of Filter Size on LeNet Performance", legend=false, grid=true, xticks=xs)
    return results, p
end

function task4_feature_investigation(tx,ty, vx,vy; epochs=10, sample_indices=[1,100,500])
    println("Task 4: Feature viz (LeNet3)")
    epochs = (AUTO_SHORT_EPOCHS && GLOBAL_TRAIN_LIMIT>0) ? min(epochs,3) : epochs
    tr_loader = create_dataloader(tx,ty)
    te_loader = create_dataloader(vx,vy; shuffle=false)

    model = create_lenet(3)
    hist  = train!(model, tr_loader, epochs; test_loader=te_loader, log_every=max(1,epochs÷5))

    class_names = get_class_names()
    labels = class_names[Flux.onecold(vy,0:9) .+ 1]

    plots_list = Plots.Plot[]
    for (i,sidx) in enumerate(sample_indices)
        # Ensure sidx is within bounds
        sidx = min(sidx, size(vx,4))
        original = vx[:,:,:,sidx]
        lbl = labels[sidx]
        x = reshape(original,32,32,3,1)

        # Extract features from different layers
        conv1_out = model[1](x)
        pool1_out = model[2](conv1_out)
        conv2_out = model[3](pool1_out)

        # Select feature maps to visualize
        f1,f2 = 1, min(2,size(conv1_out,3))
        f3,f4 = 1, min(5,size(conv2_out,3))

        p1 = heatmap(original[:,:,1]',  title="Original ($lbl)", aspect_ratio=:equal, color=:gray)
        p2 = heatmap(conv1_out[:,:,f1,1]', title="Conv1 - F$f1", aspect_ratio=:equal)
        p3 = heatmap(conv1_out[:,:,f2,1]', title="Conv1 - F$f2", aspect_ratio=:equal)
        p4 = heatmap(pool1_out[:,:,f1,1]', title="Pool1 - F$f1", aspect_ratio=:equal)
        p5 = heatmap(conv2_out[:,:,f3,1]', title="Conv2 - F$f3", aspect_ratio=:equal)
        p6 = heatmap(conv2_out[:,:,f4,1]', title="Conv2 - F$f4", aspect_ratio=:equal)

        push!(plots_list, plot(p1,p2,p3,p4,p5,p6, layout=(2,3), plot_title="Sample $(i): $lbl"))
    end
    return model, hist, plots_list
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
    println("=== Starting Homework 5: CNN Analysis on CIFAR-10 ===")
    folder = "cifar10_results"; isdir(folder) || mkdir(folder)

    (train_x, train_y), (test_x, test_y) = load_cifar10_data()

    println("\nExecuting Task 1...")
    t1 = task1_lenet5_cifar10(train_x, train_y, test_x, test_y; epochs=20)

    println("\nExecuting Task 2...")
    t2 = task2_dataset_size_experiment(train_x, train_y, test_x, test_y)
    savefig(t2[2], joinpath(folder, "task2_dataset_size_effect.png"))

    println("\nExecuting Task 3...")
    t3 = task3_filter_size_comparison(train_x, train_y, test_x, test_y; epochs=15)
    savefig(t3[2], joinpath(folder, "task3_filter_size_comparison.png"))

    println("\nExecuting Task 4...")
    t4 = task4_feature_investigation(train_x, train_y, test_x, test_y; epochs=10, sample_indices=[1,100,500])
    for (i,p) in enumerate(t4[3])
        savefig(p, joinpath(folder, "task4_features_sample_$(i).png"))
    end

    println("\n================ SUMMARY =================")
    print_task1_summary(t1[2])
    print_task2_summary(t2[1])
    print_task3_summary(t3[1])
    print_task4_summary(t4[2])
    println("========================================\n")

    println("All tasks completed!\n")

    save_everything(folder;
        model1_state = Flux.state(t1[1]), history1 = t1[2],
        task2_results = t2[1],
        task3_results = t3[1],
        model4_state = Flux.state(t4[1]), history4 = t4[2])

    return nothing
end


# run all tasks
run_all_tasks()

# Results after running


# === Starting Homework 5: CNN Analysis on CIFAR-10 ===

# Executing Task 1...
# Task 1: LeNet5

# [ Info: Epoch 5/20  TrainAcc=39.02%  TestAcc=39.22%
# [ Info: Epoch 10/20  TrainAcc=45.2%  TestAcc=45.46%
# [ Info: Epoch 15/20  TrainAcc=47.84%  TestAcc=47.05%
# [ Info: Epoch 20/20  TrainAcc=50.81%  TestAcc=49.81%

# Executing Task 2...
# Task 2: Dataset size experiment
#   Exp 1 → n=10000, ep=6
# [ Info: Epoch 6/6  TrainAcc=27.64%  TestAcc=27.4%
#   Exp 2 → n=20000, ep=3
# [ Info: Epoch 3/3  TrainAcc=24.7%  TestAcc=25.13%
#   Exp 3 → n=30000, ep=2
# [ Info: Epoch 2/2  TrainAcc=25.84%  TestAcc=26.93%

# Executing Task 3...
# Task 3: Filter sizes
#   fs=3
# [ Info: Epoch 3/15  TrainAcc=31.25%  TestAcc=31.66%
# [ Info: Epoch 6/15  TrainAcc=37.11%  TestAcc=37.3%
# [ Info: Epoch 9/15  TrainAcc=40.52%  TestAcc=41.14%
# [ Info: Epoch 12/15  TrainAcc=42.68%  TestAcc=42.61%
# [ Info: Epoch 15/15  TrainAcc=44.49%  TestAcc=44.33%
#   fs=5
# [ Info: Epoch 3/15  TrainAcc=29.58%  TestAcc=30.35%
# [ Info: Epoch 6/15  TrainAcc=36.33%  TestAcc=36.34%
# [ Info: Epoch 9/15  TrainAcc=39.6%  TestAcc=39.8%
# [ Info: Epoch 12/15  TrainAcc=41.09%  TestAcc=41.28%
# [ Info: Epoch 15/15  TrainAcc=43.28%  TestAcc=43.21%
#   fs=7
# [ Info: Epoch 3/15  TrainAcc=31.18%  TestAcc=31.88%
# [ Info: Epoch 6/15  TrainAcc=36.67%  TestAcc=36.93%
# [ Info: Epoch 9/15  TrainAcc=40.53%  TestAcc=40.59%
# [ Info: Epoch 12/15  TrainAcc=42.77%  TestAcc=42.98%
# [ Info: Epoch 15/15  TrainAcc=44.16%  TestAcc=43.58%

# Executing Task 4...
# Task 4: Feature viz (LeNet3)
# [ Info: Epoch 2/10  TrainAcc=27.97%  TestAcc=28.46%
# [ Info: Epoch 4/10  TrainAcc=31.91%  TestAcc=32.46%
# [ Info: Epoch 6/10  TrainAcc=35.26%  TestAcc=35.37%
# [ Info: Epoch 8/10  TrainAcc=38.75%  TestAcc=38.61%
# [ Info: Epoch 10/10  TrainAcc=40.06%  TestAcc=40.69%

# ================ SUMMARY =================
# Task 1 ► TestAcc=49.81%  TrainAcc=50.81%  Ep=20
# Task 2 ► Dataset size vs epochs (same steps)
#   [1] n=10000, ep=6 → TestAcc=27.4%
#   [2] n=20000, ep=3 → TestAcc=25.13%
#   [3] n=30000, ep=2 → TestAcc=26.93%
# Task 3 ► Filter size comparison
#   fs=3 → TestAcc=44.33%
#   fs=5 → TestAcc=43.21%
#   fs=7 → TestAcc=43.58%
# Task 4 ► LeNet3 features: TestAcc=40.69%  Ep=10
# ========================================

# All tasks completed!