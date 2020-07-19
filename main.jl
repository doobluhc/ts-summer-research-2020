using algorithms_module
using PyPlot
using PyCall
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: predict
using LsqFit
using Statistics
@pyimport matplotlib.pyplot as pyplt


function get_data(a::Float64,b::Float64,T::Int64,N::Int64)
    avg_cost = [zeros(T) for n = 1:N]
    cost = [zeros(T) for n = 1:N]
    gain = [zeros(T) for n = 1:N]
    regret = [zeros(T) for n = 1:N]
    regret_moving_average = [zeros(T) for n = 1:N]
    avg_regret = zeros(T)
    mov_avg_regret = zeros(T)

    ce = CE(a,b,T)

    for n = 1:N
        cost[n],avg_cost[n], gain[n], regret[n] = sample(ce)
        t = 1
        window_size = 10
        while t < length(regret[n]) - window_size
            this_window = regret[n][t:t + window_size]
            window_average = sum(this_window)/window_size
            regret_moving_average[n][t] = window_average
            t = t+1
        end
    end

    for t = 1:T
        temp1 = zeros(N)
        temp2 = zeros(N)
        for n = 1:N
            temp1[n] = regret[n][t]
            temp2[n] = regret_moving_average[n][t]
        end
        avg_regret[t] = mean(temp1)
        mov_avg_regret[t] = mean(temp2)
    end
      plot_avg_regret_vs_t(mov_avg_regret,T)

end

function plot_avg_regret_vs_t(data,T)
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
        pyplt.title("average regret vs t) for CE")
        pyplt.savefig("average regret vs t CE.png")
        println(fit.param[1])
        println(fit.param[2])
end
