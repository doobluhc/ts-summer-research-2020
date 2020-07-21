module algorithms_module
export CE, TS, SG, TSDE, sample
include("certainty_equivalence.jl")
include("thompson_sampling.jl")
include("stochastic_gradient_algorithm.jl")
include("ts_with_dynamic_episodes.jl")
end
