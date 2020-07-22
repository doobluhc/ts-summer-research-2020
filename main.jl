using algorithms_module
using PyPlot
using PyCall
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: predict
using LsqFit
using Statistics
@sk_import linear_model: LinearRegression
@pyimport matplotlib.pyplot as pyplt


function main(a::Float64,b::Float64,T::Int64,N::Int64)
    avg_cost = [zeros(T) for n = 1:N]
    cost = [zeros(T) for n = 1:N]
    gain = [zeros(T) for n = 1:N]
    regret = [zeros(T) for n = 1:N]
    regret_moving_average = [zeros(T) for n = 1:N]
    avg_regret = zeros(T)
    mov_avg_regret = zeros(T)
    algo_name = ""

    println("choose the algorithm: enter 1 for CE, 2 for SG, 3 for TS and 4 for TSDE")
    algo_type = parse(UInt8, readline())

    if algo_type == 1
        algo = CE(a,b,T)
    elseif algo_type == 2
        algo = SG(a,b,T)
    elseif algo_type == 3
        algo = TS(a,b,T)
    elseif algo_type == 4
        algo = TSDE(a,b,T)
    else
        println("Invalid input")
        return
    end

    for n = 1:N
        cost[n],avg_cost[n], gain[n], regret[n], algo_name= algorithms_module.sample(algo)
        # t = 1
        # window_size = 100
        # while t < length(regret[n]) - window_size
        #     this_window = regret[n][t:t + window_size]
        #     window_average = sum(this_window)/window_size
        #     regret_moving_average[n][t] = window_average
        #     t = t+1
        # end
        if n == 1
            println("Running $algo_name")
        end
    end

    for t = 1:T
        temp1 = zeros(N)
        # temp2 = zeros(N)
        for n = 1:N
            temp1[n] = regret[n][t]
            # temp2[n] = regret_moving_average[n][t]
        end
        avg_regret[t] = mean(temp1)
        # mov_avg_regret[t] = mean(temp2)
    end

    println("Data ready.")
    println("enter 1 to plot avg regret vs t")
    println("enter 2 to plot avg cost vs t")
    println("enter 3 to plot log avg regret vs log t")
    println("enter 4 to plot avg regret vs sqrt t")
    plot_type = parse(UInt8, readline())


    if plot_type == 1
        plot_avg_regret_vs_t(avg_regret,T,algo_name)
    elseif plot_type == 2
        plot_avg_cost_vs_t(avg_cost,T,N,algo_name)
    elseif plot_type == 3
        plot_log_avg_regret_vs_log_t(avg_regret,T,algo_name)
    elseif plot_type == 4
        plot_avg_regret_vs_sqrt_t(avg_regret,T,algo_name)
    else
        println("Invalid input")
        return
    end


end

function plot_avg_regret_vs_t(data,T,algo_name)
        pyplt.clf()
        X = [t for t = 10:(T-20)]
        Y = data[10:(T-20)]
        @. model(x,p) = p[1] * x^p[2]
        p0 = [0.5,0.5]
        fit = curve_fit(model,X,Y,p0)
        y_fit = [fit.param[1] * x^fit.param[2] for x in X]
        pyplt.plot(X, Y, color ="blue")
        pyplt.plot(X, y_fit, color ="red")
        pyplt.xlabel("t")
        pyplt.ylabel("average regret")
        pyplt.title("average regret vs t for $algo_name")
        pyplt.savefig("average regret vs t $algo_name.png")
        println(fit.param[1])
        println(fit.param[2])
end

function plot_avg_cost_vs_t(data,T,N,algo_name)
    pyplt.clf()
    for n in 1:N
        pyplt.plot([t for t = 1:T],data[n])
        pyplt.axis([0,T,0,10])
        pyplt.xlabel("t")
        pyplt.ylabel("cost/t")
        pyplt.title("cost/t vs t ")
        pyplt.savefig("average cost vs t for $algo_name.png")
    end

end

function plot_log_avg_regret_vs_log_t(data,T,algo_name)
    pyplt.clf()
    X = reshape(log.([t for t = 100:T]),(T-99),1)
    Y = reshape(log.(data[100:T]),(T-99),1)
    regr = LinearRegression()
    fit!(regr,X,Y)
    y_pred = predict(regr,X)
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)
    pyplt.plot(X, Y, color ="blue")
    pyplt.plot(X, y_pred, color ="red")
    pyplt.xlabel("logt")
    pyplt.ylabel("log(average regret)")
    pyplt.title("log(average regret) vs log(t) for $algo_name(slope = $slope)")
    pyplt.savefig("log average regret vs log t $algo_name.png")

end

function plot_avg_regret_vs_sqrt_t(data,T,algo_name)
    pyplt.clf()
    X = reshape([sqrt(t) for t = 100:T],(T-99),1)
    Y = reshape(data[100:T],(T-99),1)
    regr = LinearRegression()
    fit!(regr,X,Y)
    y_pred = predict(regr,X)
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)
    # pyplt.fill_between([sqrt(t) for t = 100:T],bot[100:T],top[100:T],color="gray")
    pyplt.plot(X, Y, color ="blue")
    pyplt.plot(X, y_pred, color ="red")
    pyplt.xlabel("sqrtt")
    pyplt.ylabel("average regret")
    pyplt.title("average regret vs sqrt t for $algo_name(slope = $slope)")
    pyplt.savefig("average regret vs sqrt t $algo_name.png")
end
