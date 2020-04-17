import numpy as np 

class NeuralNetwork():

    def __init__(self):

        #Pour initialiser les poids, 4 chiffres random dans une matice 4x1

        self.poids_synaptiques = 2 * np.random.random((4,1)) - 1 

    def sigmoide(self, x):

        #Fonction sigmoide permet de normaliser les résultats entre -1 et 1

        return 1 / (1 + np.exp(-x))

    def sigmoide_der(self, x):

        """Calculer les ajustements à faire sur les poids"""

        return x * (1 - x)

    def train(self, train_inputs, train_outputs, iterations):

        """A chaque itération ajuster le poids synaptique pour avoir un résultat plus précis"""

        for i in range(iterations):

            output = self.process(train_inputs)

            error = train_outputs - output
            
            adjustments = np.dot(train_inputs.T, error * self.sigmoide_der(output))
            self.poids_synaptiques += adjustments


    def process(self, inputs):

        """Passer les inputs pour obtenir des outputs"""

        inputs = inputs.astype(float)
        output = self.sigmoide(np.dot(inputs, self.poids_synaptiques))

        return output


if __name__ == "__main__": 

    #Le neurone est un objet

    nn = NeuralNetwork()

    print("Ce programme prédit le résultat d'une suite de 4 inputs (0 ou 1) en fonction d'une de ces 2 règles (que le programme devine): \n Règle 1 : L'output doit être le même chiffre que le premier input \n Règle 2 : L'output doit être le même chiffre que le dernier input")


    #Choisir le nombre d'itérations (100,000 minimum recommandé)
    iterations = int(input("Nb d'itérations ?"))
    choix_output = int(input("Quelle règle ? \n 1. Première valeur = résultat \n 2. Dernière valeur = résultat"))

    print("Poids synaptiques initiaux : ")
    print(nn.poids_synaptiques)

    #4 inputs pour entrainer le neurone 

    train_inputs = np.array([[0,1,1,1],
                            [1,0,1,1],
                            [1,0,0,0],
                            [0,1,1,0]])


    #4 outputs transposés pour fit l'array de 4x4

    train_outputs_premier = np.array([[0,1,1,0]]).T
    train_outputs_dernier = np.array([[1,1,0,0]]).T



    if choix_output == 1 : 
        train_outputs = train_outputs_premier
    elif choix_output == 2 : 
        train_outputs = train_outputs_dernier

    nn.train(train_inputs, train_outputs, iterations)

    print("Poids synaptiques après training : ")
    print(nn.poids_synaptiques)

    a = str(input("Input 1 : "))
    b = str(input("Input 2 : "))
    c = str(input("Input 3 : "))
    d = str(input("Input 4 : "))

    print("Input : ", a, b, c, d)
    print("Résultat prédit :")
    print(nn.process(np.array([a,b,c,d])))
    print("Résultat arrondi : ")
    print(np.round(nn.process(np.array([a,b,c,d]))))

