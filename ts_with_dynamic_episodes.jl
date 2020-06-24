using PyPlot
using PyCall
using Distributions
using Statistics
using LinearAlgebra
using LinearAlgebra: inv
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: predict
using ControlSystems
@sk_import linear_model: LinearRegression
@pyimport matplotlib.pyplot as pyplt


function sample(a::Float64,b::Float64;T = 40000,J = 300)
    cost = Float64[]
    avg_cost = Float64[]
    regret = Float64[]
    j_optimal = tr(dare(a, b, 1.0, 1.0))
    θ̃ = [0.0;0.0]
    θ̂ = [a;b]
    Σ = Symmetric([1.0 0.0;0.0 1.0])
    Σⱼ = Symmetric([1.0 0.0;0.0 1.0])
    u = 0.0
    x = 0.0
    t = 1
    Gⱼ = 0.0
    tⱼ = 0


    for j in 1:J
        #println("j = $j ##############################")
        Tⱼ₋₁ = t - tⱼ
        tⱼ = t
        #println("Tⱼ₋₁ = $Tⱼ₋₁")
        #println("tⱼ = $tⱼ")

        θ̃ = rand(MvNormal(θ̂,Σ))
        # println("θ̃ = $θ̃")

        S = float(dare(θ̃[1],θ̃[2],1.0,1.0))
        Gⱼ = -(1.0+b'*S[1]*b)^-1*b'*S[1]*a
        Σⱼ = Σ
        det_sigj = det(Σⱼ)
        # println("det sigj = $det_sigj")

        while t <= (tⱼ + Tⱼ₋₁) && det(Σ) >= 0.5*det(Σⱼ)
            # println("t = $t #############")
            if t == 1
                push!(cost,x*x)
                push!(regret,cost[t] - j_optimal)
                push!(avg_cost,cost[t])
            else
                push!(cost,cost[t-1] + x*x)
                push!(regret,regret[t-1]+cost[t]-j_optimal)
                push!(avg_cost,cost[t]/t)
            end

            u = Gⱼ * x
            w = randn()
            x = a*x+b*u+w
            z = [x;u]
            θ̂ = θ̂ + (Σ*z*(x-z'*θ̂))/(1+z'*Σ*z)
            Σ = Σ - Symmetric((Σ*z*z'*Σ))/(1+z'*Σ*z)
            det_sig = det(Σ)
            # println("det sig = $det_sig")
            t += 1

        end
    end
    return avg_cost[1:T],regret[1:T]

end

function simulation(a::Float64,b::Float64;T = 40000 , N = 100)
    avg_cost  = [ zeros(T) for n = 1:N ]
    regret = [ zeros(T) for n = 1:N ]
    avg_regret = zeros(T)
    pyplt.clf()
    for n in 1:N
        avg_cost[n],regret[n]= sample(a,b)
        println(length(regret[n]))
    end



    #plot log(average regret) vs log t
    for t = 1:T
        temp = zeros(N)
        for n = 1
            temp[n] = regret[n][t]
        end
        if mean(temp) < 0
            avg_regret[t] = 0
        else
            avg_regret[t] = log(10,mean(temp))
        end
    end

    X = reshape([log(t) for t = 100:T],39901,1)
    Y = reshape(avg_regret[100:T],39901,1)
    regr = LinearRegression()
    fit!(regr,X,Y)
    y_pred = predict(regr,X)
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)
    pyplt.scatter(X, Y, color ="blue")
    pyplt.plot(X, y_pred, color ="red")
    pyplt.text(5,3.5,"slope = $slope")
    pyplt.text(5,4.5,"intercept = $intercept")
    pyplt.xlabel("logt")
    pyplt.ylabel("log(average regret)")
    pyplt.title("log(average regret) vs log(t) for TSDE")
    pyplt.savefig("average  cost vs log t TSDE.png")


end