from numpy import exp, array, random, dot


# Modéliser un seul neurone, avec 3 entrée et 1 sortie.
class ReseauxdeNeurone():
    def __init__(self):
        # Attribuer des poids aléatoires à une matrice 3 x 1,
        # avec des valeurs comprises entre -1 et 1
        self.poids_synaptiques = 2 * random.random((3, 1)) - 1

    # La fonction Sigmoïde, qui décrit une courbe en forme de S.
    # Passer la somme pondérée des entrées par cette fonction pour 
    # les normaliser entre 0 et 1.
    def __sigmoide(self, x):
        return 1 / (1 + exp(-x))

    # La dérivée de la fonction sigmoïde.
    # C'est le gradient de la courbe sigmoïde.
    # Elle indique le degré de confiance que nous avons dans le poids existant.
    def __derivation_sigmoide(self, x):
        return x * (1 - x)

    # Former le réseau de neurone par un processus d'essais et d'erreurs.
    # Ajuster les poids synaptiques à chaque fois.
    def apprentisage(self, echantillon_entree, echantillon_sortie, nombre_iterations_entraînement):
        for iteration in range(nombre_iterations_entraînement):
            # Faire passer l'ensemble d'apprentissage par le réseau de neurone (un seul neurone).
            valeur_predite = self.propagation(echantillon_entree)

            # Calculer la perte/l'erreur (la différence entre la sortie cible et la sortie prédite).
            perte = echantillon_sortie - valeur_predite

            # Rétropropagation 
            # Multiplier la perte par l'entrée et à nouveau par le gradient de la courbe sigmoïde.
            # Cela signifie que les poids les moins sûrs sont davantage ajustés.
            # Cela signifie que les entrées, qui sont nulles, n'entraînent pas de changements dans les poids.
            ajustement = dot(echantillon_entree.T, perte * self.__derivation_sigmoide(valeur_predite))

            # Ajuster les poids.
            self.poids_synaptiques += ajustement

    # Propagation avant
    # Le réseau de neurone réfléchit.
    def propagation(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoide(dot(inputs, self.poids_synaptiques))


if __name__ == "__main__":

    #Initialiser un réseau de neurone à un seul neurone.
    reseaux_neurone = ReseauxdeNeurone()

    print("Poids synaptiques initiaux aléatoires : ")
    print(reseaux_neurone.poids_synaptiques)

    # L'ensemble d'apprentissage comprend 4 exemples, 
    # chacun composé de 3 valeurs d'entrée et d'une valeur de sortie.
    echantillon_entree = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]])
    echantillon_sortie = array([[0, 1, 1, 0]]).T

    # Entraînez le réseau de neurone à l'aide d'un ensemble d'entraînement.
    # Répétez l'opération 10000 fois et procédez à de petits ajustements à chaque fois.
    reseaux_neurone.apprentisage(echantillon_entree, echantillon_sortie, 10000)

    print("Nouveaux poids synaptiques après l'entraînement : ")
    print(reseaux_neurone.poids_synaptiques)

    # Tester le réseau de neurone avec une nouvelle situation.
    print("La valeur prédite pour la nouvelle situation [1, 0, 0] -> ? ")
    print(reseaux_neurone.propagation(array([1, 0, 0])))

