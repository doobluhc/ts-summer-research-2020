
using ControlSystems
using LinearAlgebra
using Statistics
using PyPlot
using PyCall
@pyimport matplotlib.pyplot as pyplt


abstract type Algorithm end


mutable struct CE <: Algorithm
    A::Float64
    B::Float64
    Q::Float64
    R::Float64
end


function sample(A::Array{Float64},B::Array{Float64},Q::Array{Float64},R::Array{Float64},T::Int64)

    # A = ce.A
    # B = ce.B
    # Q = ce.Q
    # R = ce.R

    if size(B) == (size(B)[1],)
        B = reshape(B,size(B)[1],1)
    end
    if size(R) == (size(R)[1],)
        R = reshape(R,size(R)[1],1)
    end
    cost = zeros(T)
    regret = zeros(T)
    avg_cost = zeros(T)
    x = zeros(Float64,size(A)[1],1)
    u = zeros(Float64,size(B)[2],1)
    w = zeros(Float64,size(A)[1],1)
    θ̂ = vcat(zeros(Float64,size(A)[1],size(A)[1]),ones(Float64,size(B)[2],size(B)[1]))
    Σ = Matrix{Float64}(I,size(B)[1]+size(B)[2],size(B)[1]+size(B)[2])
    # println("A", size(A))
    # println("B", size(B))
    # println("Q", size(Q))
    # println("R", size(R))
    # println("x", size(x))
    # println("u", size(u))
    S = dare(A,B,Q,R)
    j_optimal = tr(S)
    println(j_optimal)

    for t = 1:T
        if t == 1
            cost[t] = (x'*Q*x + u'*R*u)[1]
            regret[t] = (x'*Q*x + u'*R*u)[1] - j_optimal[1]
        else
            cost[t] = cost[t-1] + (x'*Q*x + u'*R*u)[1]
            regret[t] = regret[t-1] + (x'*Q*x + u'*R*u)[1] - j_optimal[1]
        end
        avg_cost[t] = cost[t] / t


        Â = θ̂[1:size(A)[1],:]
        B̂ = θ̂[size(A)[1]+1:size(A)[1]+size(B)[2],:]'

        Ŝ = dare(Â,B̂,Q,R)
        gain = inv(R + B̂'*Ŝ*B̂)*B̂'*Ŝ*Â
        u = -gain * x
        w = Array{Float64}(undef,size(A)[1],1)
        fill!(w,randn())
        z = vcat(x,u)

        x = A * x + B * u + w


        θ̂ = θ̂ + (Σ * z * (x - θ̂'*z)') / (1 + (z' * Σ * z)[1])
        Σ = Σ - Σ * z * z' * Σ / (1 + (z' * Σ * z)[1])
    end

    algo_name = "CE"
    pyplt.clf()
    pyplt.plot([t for t = 1:T],avg_cost)
    pyplt.axis([0,T,0,10])
    pyplt.xlabel("t")
    pyplt.ylabel("cost/t")
    pyplt.title("cost/t vs t ")
    pyplt.savefig("average cost vs t for CE2.png")
    # return cost, avg_cost, regret, algo_name

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
