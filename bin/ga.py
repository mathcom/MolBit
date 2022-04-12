import math
import numpy as np

class GAOptimizer(object):
    def __init__(self, population_size, chromosome_length, fitness_ft, mutation_prob=0.2):
        self.psize = population_size
        self.csize = chromosome_length
        self.fitness_ft = fitness_ft
        self.mutation_prob = mutation_prob
        
    def get_random_population(self):
        chromosomes = np.random.randint(2, size=(self.psize, self.csize)).astype(np.float32)
        scores = np.zeros(self.psize)
        return chromosomes, scores # chromosomes.shape = (psize, csize)
        
    def partial_fit(self, chromosomes, scores, topk=10, skipk=None): # chromosomes.shape = (psize, csize), scores.shape = (psize,)
        if skipk is None:
            skipk = topk
        ## score calculation
        for i in range(skipk, self.psize):
            scores[i] = self.fitness_ft(chromosomes[i])
        ## parents selection
        idx_parents = np.argsort(scores)[-topk:]
        parents = chromosomes[idx_parents]
        parents_scores = scores[idx_parents]
        ## crossover
        new_chromosomes = np.zeros_like(chromosomes)
        new_scores = np.zeros_like(scores)
        for k in range(self.psize):
            if k < topk:
                new_chromosomes[k] = parents[k]
                new_scores[k] = parents_scores[k]
            else:
                i, j = np.random.choice(topk, 2, replace=False)
                t = np.random.choice(self.csize-2) + 1 # trim start(0) and end(csize-1)
                new_chromosomes[k][:t] = parents[i][:t]
                new_chromosomes[k][t:] = parents[j][t:]
        ## mutation
        for k in range(topk, self.psize):
            p = np.random.rand()
            if p < self.mutation_prob:
                t = np.random.choice(self.csize)
                new_chromosomes[k][t] = 1. - new_chromosomes[k][t]
        return new_chromosomes, new_scores
        


if __name__=="__main__":     
    class FitnessScorer(object):
        def __init__(self, sample_size=10):
            self.sample_size=sample_size
            
        def __call__(self, x):
            score = 0.
            for i, j in enumerate(x):
                score += i * j
            return score
    
    population_size = 20
    chromosome_length = 20
    fitness_ft = FitnessScorer()
    
    optimizer = GAOptimizer(population_size, chromosome_length, fitness_ft)
    
    chromosomes, scores = optimizer.get_random_population()
    chromosomes, scores = optimizer.partial_fit(chromosomes, scores, skipk=0)
    print(0, scores)
    
    num_generations = 20
    for g in range(1, num_generations):
        chromosomes, scores = optimizer.partial_fit(chromosomes, scores)
        print(g, scores)