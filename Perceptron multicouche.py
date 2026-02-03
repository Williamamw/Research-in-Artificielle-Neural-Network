from numpy import exp, array, random, dot

# Créer une classe pour des couches cachée
class CoucheNeuron():
    def __init__(self, nombre_de_neurones, nombre_entrées_par_neurone):
        self.poids_synaptiques = 2 * random.random((nombre_entrées_par_neurone, nombre_de_neurones)) - 1


class ReseauxdeNeurone():
    def __init__(self, couche1, couche2):
        self.couche1 = couche1
        self.couche2 = couche2

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
            # Faire passer l'ensemble d'apprentissage par le réseau de neurone.
            predite_de_la_couche_1, predite_de_la_couche_2 = self.propagation(echantillon_entree)

            # Calculer la perte/l'erreur pour la couche 2 (la différence entre 
            # la sortie cible et la sortie prédite).
            couche2_perte = echantillon_sortie - predite_de_la_couche_2
            couche2_delta = couche2_perte * self.__derivation_sigmoide(predite_de_la_couche_2)

            # Rétropropagation 
            # Calculer la perte pour la couche 1 (En examinant les poids 
            # de la couche 1, le réseau de neurones peut déterminer dans quelle 
            # mesure la couche 1 a contribué à la perte de la couche 2).).
            couche1_perte = couche2_delta.dot(self.couche2.poids_synaptiques.T)
            couche1_delta = couche1_perte * self.__derivation_sigmoide(predite_de_la_couche_1)

            # Calculer de combien il faut ajuster les poids
            couche1_ajustement = echantillon_entree.T.dot(couche1_delta)
            couche2_ajustement = predite_de_la_couche_1.T.dot(couche2_delta)

            # Ajuster les poids.
            self.couche1.poids_synaptiques += couche1_ajustement
            self.couche2.poids_synaptiques += couche2_ajustement

    # Propagation avant
    # Le réseau de neurone réfléchit.
    def propagation(self, entrees):
        predite_de_la_couche1 = self.__sigmoide(dot(entrees, self.couche1.poids_synaptiques))
        predite_de_la_couche2 = self.__sigmoide(dot(predite_de_la_couche1, self.couche2.poids_synaptiques))
        return predite_de_la_couche1, predite_de_la_couche2

    # Print les poids du réseau de neurone
    def print_poids(self):
        print ("    Couche 1 (4 neurones, chacun avec 3 entrées) : ")
        print (self.couche1.poids_synaptiques)
        print ("    Couche 2 (1 neurone, avec 4 entrées) :")
        print (self.couche2.poids_synaptiques)

if __name__ == "__main__":
    # Créer la couche 1 (4 neurones, chacun avec 3 entrées)
    couche1 = CoucheNeuron(4, 3)

    # Créer la couche 2 (un seul neurone avec 4 entrées)
    couche2 = CoucheNeuron(1, 4)

    # Combine the layers to create a neural network
    reseaux_neurone = ReseauxdeNeurone(couche1, couche2)

    print ("Etage 1) Poids synaptiques initiaux aléatoires : ")
    reseaux_neurone.print_poids()

    # L'ensemble d'apprentissage comprend 7 exemples, 
    # chacun composé de 3 valeurs d'entrée et d'une valeur cible.
    echantillon_entree = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    echantillon_sortie = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Entraînez le réseau de neurone à l'aide d'un ensemble d'entraînement.
    # Répétez l'opération 90000 fois et procédez à de petits ajustements à chaque fois.
    reseaux_neurone.apprentisage(echantillon_entree, echantillon_sortie, 90000)

    print ("Etage 2) Nouveaux poids synaptiques après l'entraînement : ")
    reseaux_neurone.print_poids()

    # Tester le réseau de neurone avec une nouvelle situation.
    print ("Etage 3) La valeur prédite pour la nouvelle situation [1, 1, 0] -> ?: ")
    valeur_couche_cachée, valeur_predite = reseaux_neurone.propagation(array([1, 1, 0]))
    print (valeur_predite)