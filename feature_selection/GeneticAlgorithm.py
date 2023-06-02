import numpy as np
from deap import tools

#FeatureSelection by GA
def runGA(toolbox,X):
    population_size = 100
    generations = 100
    pop = toolbox.population(n=population_size)

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}")

        # Valutazione della popolazione
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Stampa delle metriche della popolazione attuale
        rmse_values = [ind.fitness.values[0] for ind in pop]
        r2_values = [ind.fitness.values[1] for ind in pop]
        mre_values = [ind.fitness.values[2] for ind in pop]
        print(f"RMSE values: {rmse_values}")
        print(f"R^2 values: {r2_values}")
        print(f"MRE values: {mre_values}")

        # Stampa della popolazione attuale
        print(f"\nPopulation of Generation {gen + 1}:")
        for i, ind in enumerate(pop):
            print(f"Individual {i + 1}: {ind}")

            # Selezione degli individui migliori
        offspring = toolbox.select(pop, len(pop))

        # Clonazione degli individui selezionati
        offspring = list(map(toolbox.clone, offspring))

        # Applicazione degli operatori di crossover e mutazione
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Valutazione degli individui appena generati
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Sostituzione della popolazione con la nuova generazione
        pop[:] = offspring

    # Selezione dell'individuo migliore
    best_individual = tools.selBest(pop, k=1)[0]
    selected_features = [feature for feature, mask in zip(X.columns, best_individual) if mask]

    #Best Individual fitness metrics
    print("\n--Selected features--")
    print(selected_features)
    print("--------------------------")
    print("--Best individual fitness--")
    print("RMSE:", best_individual.fitness.values[0])
    print("R^2 score:", best_individual.fitness.values[1])
    print("MRE:", best_individual.fitness.values[2])
    print("--------------------------")
