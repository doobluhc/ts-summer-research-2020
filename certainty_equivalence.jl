
using ControlSystems
using LinearAlgebra
using Statistics


abstract type Algorithm end


mutable struct CE <: Algorithm
    a::Float64
    b::Float64
    T::Int64
end


function sample(ce::CE)

    a = ce.a
    b = ce.b
    T = ce.T
    cost = zeros(T)
    regret = zeros(T)
    avg_cost = zeros(T)
    gain = zeros(T)
    x = 0.0
    u = 0.0
    w = 0.0
    j_optimal = 1
    θ̂ = [0.0;1.0]
    Σ = [1.0 0.0; 0.0 1.0]
    â = zeros(T)
    b̂ = zeros(T)

    for t = 1:T
        if t == 1
            cost[t] = x * x
            regret[t] = x*x - j_optimal
        else
            cost[t] = cost[t-1] + x * x
            regret[t] = regret[t-1] + x*x - j_optimal
        end
        avg_cost[t] = cost[t] / t
        # println(x*x)



        â = θ̂[1]
        b̂ = θ̂[2]
        gain[t] = θ̂[1] / θ̂[2]
        u = -gain[t] * x
        w = randn()
        z = [x;u]

        x = a * x + b * u + w


        θ̂ = θ̂ + Σ * z * (x - z' * θ̂) / (1 + z' * Σ * z)
        Σ = Σ - Σ * z * z' * Σ / (1 + z' * Σ * z)
    end

    algo_name = "CE"
    return cost, avg_cost, gain, regret,algo_name

end

# function simulation(ce::CE)
#     a = ce.a
#     b = ce.b
#     T = ce.T
#     N = ce.N
#     avg_cost = [zeros(T) for n = 1:N]
#     cost = [zeros(T) for n = 1:N]
#     gain = [zeros(T) for n = 1:N]
#     regret = [zeros(T) for n = 1:N]
#     avg_regret = zeros(T)
#     mov_avg_regret = zeros(T)
#     bot = zeros(T)
#     top = zeros(T)
#     regret_moving_average = [zeros(T) for n = 1:N]
#     pyplt.clf()
#     c = 1
#     α = 1
#     for n = 1:N
#         cost[n],avg_cost[n], gain[n], regret[n] = sample(ce)
#         t = 1
#         window_size = 10
#         while t < length(regret[n]) - window_size
#             this_window = regret[n][t:t + window_size]
#             window_average = sum(this_window)/window_size
#             regret_moving_average[n][t] = window_average
#             t = t+1
#         end
#         # println(n)
#         # pyplt.plot([t for t = 1:T],cost[n])
#         # pyplt.axis([0,T,0,10])
#         # pyplt.xlabel("t")
#         # pyplt.ylabel("cost/t")
#         # pyplt.title("cost function value/t vs t ")
#         # pyplt.savefig("average cost vs t for CE.png")
#     end
#
#
#     # for t = 1:T
#     #     temp1 = zeros(N)
#     #     temp2 = zeros(N)
#     #     for n = 1:N
#     #         temp1[n] = regret[n][t]
#     #         temp2[n] = regret_moving_average[n][t]
#     #     end
#     #     avg_regret[t] = mean(temp1)
#     #     mov_avg_regret[t] = mean(temp2)
#         # bot[t] = quantile!(temp,0.25)
#         # top[t] = quantile!(temp,0.75)
#     # end
#     #plot average regret vs t
#     # pyplt.clf()
#     # X = [t for t = 10:(T-20)]
#     # Y = mov_avg_regret[10:(T-20)]
#     # @. model(x,p) = p[1] * x^p[2]
#     # p0 = [0.5,0.5]
#     # fit = curve_fit(model,X,Y,p0)
#     # y_fit = [fit.param[1] * x^fit.param[2] for x in X]
#     # pyplt.plot(X, Y, color ="blue")
#     # pyplt.plot(X, y_fit, color ="red")
#     # pyplt.xlabel("t")
#     # pyplt.ylabel("average regret")
#     # pyplt.title("average regret vs t) for CE")
#     # pyplt.savefig("average regret vs t CE.png")
#     # println(fit.param[1])
#     # println(fit.param[2])
#     #plot average regret vs sqrt t
#     # pyplt.clf()
#     # X = reshape([sqrt(t) for t = 10:T],(T-9),1)
#     # Y = reshape(avg_regret[10:T],(T-9),1)
#     # regr = LinearRegression()
#     # fit!(regr,X,Y)
#     # y_pred = predict(regr,X)
#     # slope = float(regr.coef_)
#     # intercept = float(regr.intercept_)
#     # pyplt.fill_between([sqrt(t) for t = 10:T],bot[10:T],top[10:T],color="gray")
#     # pyplt.plot([sqrt(t) for t = 10:T], avg_regret[10:T], color ="blue")
#     # pyplt.plot(X, y_pred, color ="red")
#     # print(slope)
#     # pyplt.xlabel("sqrtt")
#     # pyplt.ylabel("average regret")
#     # pyplt.title("average regret vs sqrt t) for CE(slope = $slope)")
#     # pyplt.savefig("average regret vs sqrt t CE.png")
#
#     #plot log(average regret) vs log(t)
#     # pyplt.clf()
#     # X = reshape(log.([t for t = 10:T]),(T-9),1)
#     # Y = reshape(log.(avg_regret[10:T]),(T-9),1)
#     # regr = LinearRegression()
#     # fit!(regr,X,Y)
#     # slope = float(regr.coef_)
#     # intercept = float(regr.intercept_)
#     # y_pred = predict(regr,X)
#     # pyplt.plot(log.([t for t = 10:T]), log.(avg_regret[10:T]), color ="blue")
#     # pyplt.plot(X, y_pred, color ="red")
#     # pyplt.xlabel("logt")
#     # pyplt.ylabel("log(average regret)")
#     # pyplt.title("log(average regret) vs log(t) for CE(slope = $slope)")
#     # pyplt.savefig("log average regret vs log t CE.png")
#
#
#
# end
